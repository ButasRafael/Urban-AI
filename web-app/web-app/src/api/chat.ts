import client from "./client";

export interface ChatRequest {
  message: string;
  session_id?: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  streaming?: boolean;
}

export interface SessionSummary {
  id: number;
  title?: string;
  created_at: string;
  last_message_at: string;
}

export interface RagChunk {
  id: number;
  media_id: number;
  chunk: string;
  image_url?: string;
}

export async function listSessions(): Promise<SessionSummary[]> {
  const { data } = await client.get<SessionSummary[]>("/chat/sessions");
  return data;
}

export async function getSessionHistory(
  sessionId: number
): Promise<ChatMessage[]> {
  const { data } = await client.get<{ messages: ChatMessage[] }>(
    `/chat/sessions/${sessionId}`
  );
  return data.messages;
}

export async function deleteSession(sessionId: number): Promise<void> {
  await client.delete(`/chat/sessions/${sessionId}`);
}

export async function getRagChunk(id: number): Promise<RagChunk> {
  const { data } = await client.get<RagChunk>(`/rag/chunk/${id}`);
  return data;
}

export async function updateSessionTitle(sessionId: number, title: string): Promise<void> {
  await client.patch(`/chat/sessions/${sessionId}/title`, null, {
    params: { title }
  });
}

export async function sendChatStream(
  body: ChatRequest,
  onDelta: (chunk: string) => void,
  onSession?: (sid: number) => void
): Promise<{ session_id: number }> {
  const base = (client.defaults.baseURL ?? "").replace(/\/+$/, "");
  const url = `${base}/chat/stream`;

  const token = localStorage.getItem("accessToken");

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const resp = await fetch(url, {
    method: "POST",
    headers,
    credentials: "omit",
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let sessionId: number | undefined;

  const isRecord = (v: unknown): v is Record<string, unknown> =>
    typeof v === "object" && v !== null;

  const parseSSEFrame = (frame: string): { event: string | null; data: unknown } => {
    const lines = frame.split(/\r?\n/);
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of lines) {
      if (!line) continue;
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    const raw = dataLines.join("\n");
    try { return { event, data: raw ? JSON.parse(raw) : "" }; }
    catch { return { event, data: raw }; }
  };

  const handleFrame = (frame: string) => {
    const { event, data } = parseSSEFrame(frame);
    if (!event) return;

    if (event === "session" && isRecord(data) && typeof data.session_id === "number") {
      sessionId = data.session_id;
      onSession?.(data.session_id);
      return;
    }
    if (event === "delta") {
      if (typeof data === "string") onDelta(data);
      else if (isRecord(data) && typeof data.text === "string") onDelta(data.text);
      return;
    }
    if (event === "error") {
      const msg =
        typeof data === "string"
          ? data
          : isRecord(data) && typeof data.message === "string"
          ? data.message
          : "stream error";
      throw new Error(msg);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split(/\n\n/);
    buf = parts.pop() ?? "";
    for (const part of parts) if (part.trim()) handleFrame(part);
  }
  if (buf.trim()) handleFrame(buf);

  const finalSessionId = sessionId ?? body.session_id;
  if (typeof finalSessionId !== "number") {
    throw new Error("No session_id received or provided.");
  }
  return { session_id: finalSessionId };
}