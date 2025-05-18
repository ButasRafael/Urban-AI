import client from "./client";

export interface ChatRequest {
  message: string;
  session_id?: number;
  latitude?: number;
  longitude?: number;
  radius_km?: number;
}
export interface ChatResponse {
  session_id: number;
  answer: string;
}
export interface ChatMessage { role: "user" | "assistant"; content: string; }

export async function sendChat(req: ChatRequest) {
  const { data } = await client.post<ChatResponse>("/chat", req);
  return data;
}
