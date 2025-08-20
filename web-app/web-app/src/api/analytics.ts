import client from "./client";

export interface DayStat { date: string; count: number; }
export interface UserStat { user: string; count: number; }

export interface KPIResponse {
  window: { start: string; end: string };
  uploads: { total: number; images: number; videos: number };
  latency_ms: { avg: number; p95: number };
  detections: { total: number };
}

export interface LatencyByDay { date: string; avg_ms: number; p95_ms: number; count: number; }
export interface SeverityByDay { date: string; low: number; medium: number; high: number; }
export interface SourceBreakdown { source: "yolo"|"gpt_dino"|"sam_fallback"; count: number; }
export interface StatusBreakdown { status: "open"|"resolved"|"ignored"; count: number; }
export interface TopClass { class_name: string; count: number; }
export interface ConfidenceSummary { avg: number; p05: number; p50: number; p95: number; }
export interface HeatBucket { geohash6: string; count: number; latest: string; }
export interface TTRow { severity: "low"|"medium"|"high"; avg_hours: number; p95_hours: number; count: number; }
export interface Hotspot {
  geohash: string;
  precision: number;
  count: number;
  prev_count: number;
  trend_pct: number;
  latest: string | null;
  uploaders: number;
  severity: { low: number; medium: number; high: number };
  top_classes: { class_name: string; count: number }[];
  lat: number | null;
  lon: number | null;
  bbox: [number, number, number, number] | null;
}

export interface AgingBySeverityRow {
  "0-24h": number;
  "1-3d": number;
  "3-7d": number;
  "7-30d": number;
  ">30d": number;
  total: number;
}

export interface AgingByAssigneeRow {
  assignee: string | null;
  severity: "low" | "medium" | "high";
  buckets: Partial<Record<AgingBucketKey, number>>;
  total: number;
}

export interface AgingResponse {
  window: { start: string; end: string };
  sla_hours: { low: number; medium: number; high: number };
  by_severity: Partial<Record<"low" | "medium" | "high", AgingBySeverityRow>>;
  by_assignee: AgingByAssigneeRow[];
  sla_breach_open: Partial<Record<"low" | "medium" | "high", number>>;
  sla_breach_rate: Partial<Record<"low" | "medium" | "high", number>>;
  open_counts: Partial<Record<"low" | "medium" | "high", number>>;
}

export type MediaTypeFilter = "image" | "video" | undefined;
export type RangeParams = { days?: number; media_type?: MediaTypeFilter };
export type AgingBucketKey = "0-24h" | "1-3d" | "3-7d" | "7-30d" | ">30d";
export async function uploadsByDay(params?: RangeParams) {
  const { data } = await client.get<DayStat[]>("/analytics/uploads-by-day", { params });
  return data;
}

export async function uploadsByUser(params?: RangeParams) {
  const { data } = await client.get<UserStat[]>("/analytics/uploads-by-user", { params });
  return data;
}

export async function getKPIs(params?: RangeParams) {
  const { data } = await client.get<KPIResponse>("/analytics/kpis", { params });
  return data;
}

export async function getLatencyByDay(params?: RangeParams) {
  const { data } = await client.get<LatencyByDay[]>("/analytics/latency-by-day", { params });
  return data;
}

export async function getSeverityByDay(params?: RangeParams) {
  const { data } = await client.get<SeverityByDay[]>("/analytics/detections/severity-by-day", { params });
  return data;
}

export async function getSourceBreakdown(params?: RangeParams) {
  const { data } = await client.get<SourceBreakdown[]>("/analytics/detections/source-breakdown", { params });
  return data;
}

export async function getStatusBreakdown(params?: RangeParams) {
  const { data } = await client.get<StatusBreakdown[]>("/analytics/detections/status-breakdown", { params });
  return data;
}

export async function getTopClasses(params?: RangeParams & { limit?: number }) {
  const { data } = await client.get<TopClass[]>("/analytics/detections/top-classes", { params });
  return data;
}

export async function getConfidenceSummary(params?: RangeParams) {
  const { data } = await client.get<ConfidenceSummary>("/analytics/detections/confidence-summary", { params });
  return data;
}

export async function getGeoHeatmap(params?: RangeParams & { min_count?: number }) {
  const { data } = await client.get<HeatBucket[]>("/analytics/geo/heatmap", { params });
  return data;
}

export async function getTimeToResolution(params?: RangeParams) {
  const { data } = await client.get<TTRow[]>("/analytics/detections/time-to-resolution", { params });
  return data;
}

export async function getGeoHotspots(
  params?: RangeParams & { precision?: number; min_count?: number; limit?: number }
) {
  const { data } = await client.get<Hotspot[]>("/analytics/geo/hotspots", { params });
  return data;
}

export async function getIssuesAgingBuckets(params?: RangeParams & {
  include_unassigned?: boolean;
  sla_high_h?: number;
  sla_medium_h?: number;
  sla_low_h?: number;
}) {
  const { data } = await client.get<AgingResponse>("/analytics/issues/aging-buckets", { params });
  return data;
}
