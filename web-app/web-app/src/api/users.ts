import client from "./client";

export type Role = "user" | "authority" | "admin";

export interface UserLite {
  username: string;
  role: Role;
}

export async function searchUsers(
  q: string,
  limit = 8,
  role: Role | undefined = "authority"
): Promise<UserLite[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (role) params.set("role", role);
  const { data } = await client.get<UserLite[]>("/users", { params });
  return data ?? [];
}
