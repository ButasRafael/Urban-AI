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
  descriptions?: string[];
  solutions?: string[];
}

interface ProblemDTO {
  media_id: number;
  address?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  user_username: string | null;
  media_type: "image" | "video";
  annotated_image_url?: string | null;
  annotated_video_url?: string | null;
  created_at: string;
  predicted_classes: string[] | null;
  descriptions?: (string | null)[] | null;
  solutions?: (string | null)[] | null;
}

function normalize(p: ProblemDTO): Problem {
  return {
    media_id: p.media_id,
    address: p.address ?? undefined,
    latitude: p.latitude ?? undefined,
    longitude: p.longitude ?? undefined,
    user_username: p.user_username ?? "",
    media_type: p.media_type,
    annotated_image_url: p.annotated_image_url ?? undefined,
    annotated_video_url: p.annotated_video_url ?? undefined,
    created_at: p.created_at,
    predicted_classes: p.predicted_classes ?? [],
    descriptions: p.descriptions?.filter(Boolean) as string[] | undefined,
    solutions: p.solutions?.filter(Boolean) as string[] | undefined,
  };
}

export async function getProblems(
  opts: { media_type?: "image" | "video"; klass?: string } = {}
) {
  const params = new URLSearchParams();
  if (opts.media_type) params.set("media_type", opts.media_type);
  if (opts.klass && opts.klass.trim()) params.set("klass", opts.klass.trim());

  const { data } = await client.get<ProblemDTO[]>("/problems", { params });
  return (data ?? []).map(normalize);
}
