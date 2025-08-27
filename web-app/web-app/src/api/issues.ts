import client from "./client";

export type IssueSeverity = "low" | "medium" | "high";
export type IssueStatus   = "open" | "resolved" | "ignored";
export type IssueSource   = "yolo" | "gpt_dino" | "sam_fallback";
export type MediaType     = "image" | "video";

export interface Issue {
  id: number;
  media_id: number;
  frame_id: number;
  created_at: string;
  class_name: string;
  confidence?: number | null;
  bbox: [number, number, number, number];
  description?: string | null;
  solution?: string | null;
  severity: IssueSeverity;
  status: IssueStatus;
  source: IssueSource;
  annotated_image_url?: string | null;
  annotated_video_url?: string | null;
  address?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  assigned_to?: string | null;
  verified_by?: string | null;
  verified_at?: string | null;
}

export type IssuesSummary =
  Partial<Record<IssueSeverity, Partial<Record<IssueStatus, number>>>>;

export async function listIssues(params?: {
  media_id?: number;
  media_type?: MediaType;
  klass?: string;
  severity?: IssueSeverity;
  status?: IssueStatus;
  source?: IssueSource;
  assigned_to?: string;
  limit?: number;
  offset?: number;
}) {
  const q = new URLSearchParams();
  if (!params) params = {};
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") q.append(k, String(v));
  });
  const { data } = await client.get<Issue[]>("/problems/issues", { params: q });
  return data ?? [];
}

export async function issuesSummaryByMedia(mediaIds: number[]) {  // <-- NEW
  const q = new URLSearchParams();
  mediaIds.forEach((id) => q.append("media_ids", String(id)));
  const { data } = await client.get<Record<number, IssuesSummary>>(
    "/problems/issues/summary_by_media",
    { params: q }
  );
  return data ?? {};
}

export async function issuesSummary(media_type?: MediaType) {
  const q = new URLSearchParams();
  if (media_type) q.set("media_type", media_type);
  const { data } = await client.get<IssuesSummary>("/problems/issues/summary", { params: q });
  return data ?? {};
}

export async function getIssue(id: number) {
  const { data } = await client.get(`/issues/${id}`);
  return data as Issue;
}

export async function updateIssueStatus(id: number, status: IssueStatus) {
  const { data } = await client.patch(`/issues/${id}/status`, { status });
  return data as { id: number; old_status?: IssueStatus; new_status?: IssueStatus; resolved_at?: string | null };
}

export async function updateIssueSeverity(id: number, severity: IssueSeverity) {
  const { data } = await client.patch(`/issues/${id}/severity`, { severity });
  return data as { id: number; severity: IssueSeverity };
}

export async function assignIssue(id: number, assigned_to?: string | null) {
  const { data } = await client.patch(`/issues/${id}/assign`, { assigned_to: assigned_to ?? null });
  return data as { id: number; assigned_to: string | null };
}

export async function verifyIssue(id: number, verified: boolean) {
  const { data } = await client.patch(`/issues/${id}/verify`, { verified });
  return data as { id: number; verified_by: string | null; verified_at: string | null };
}

export async function bulkUpdateStatus(mediaId: number, status: "resolved" | "ignored") {
  const { data } = await client.patch("/problems/issues/bulk_status", {
    media_id: mediaId,
    status
  });
  return data as {
    updated_count: number;
    media_id: number;
    new_status: string;
    message: string;
  };
}

import type { Problem } from "./problems";

export function issueToProblemLike(i: Issue): Problem {
  return {
    media_id: i.media_id,
    address: i.address ?? undefined,
    latitude: i.latitude ?? undefined,
    longitude: i.longitude ?? undefined,
    user_username: "", // not present on IssueOut
    media_type: i.annotated_image_url ? "image" : "video",
    annotated_image_url: i.annotated_image_url ?? undefined,
    annotated_video_url: i.annotated_video_url ?? undefined,
    created_at: i.created_at,
    predicted_classes: [i.class_name],
    descriptions: i.description ? [i.description] : [],
    solutions: i.solution ? [i.solution] : [],

  };
}
