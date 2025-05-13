import client from "./client";

export interface Problem {
  media_id: number;
  address?: string;
  latitude?: number;
  longitude?: number;
  user_username: string;
  media_type: "image" | "video";
  annotated_image_url?: string;
  annotated_video_url?: string;
  created_at: string;
  predicted_classes: string[];
}

export async function getProblems(opts: { media_type?: string; klass?: string } = {}) {
  const params = new URLSearchParams();
  if (opts.media_type) params.append('media_type', opts.media_type);
  if (opts.klass) params.append('klass', opts.klass.toLowerCase()); // Make search case-insensitive
  const { data } = await client.get<Problem[]>("/problems", { params });
  return data;
}
