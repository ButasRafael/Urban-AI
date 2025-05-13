import client from "./client";

export interface DayStat {
  date: string;
  count: number;
}

export interface UserStat {
  user: string;
  count: number;
}

export async function uploadsByDay() {
  const { data } = await client.get<DayStat[]>("/analytics/uploads-by-day");
  return data;
}

export async function uploadsByUser() {
  const { data } = await client.get<UserStat[]>("/analytics/uploads-by-user");
  return data;
}