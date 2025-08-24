import client from "./client";

export interface ChatRequest {
  message: string;
  session_id?: number;
}

export interface ChatResponse {
  session_id: number;
  answer: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
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

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const { data } = await client.post<ChatResponse>("/chat", req);
  return data;
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
