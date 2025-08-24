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

// New analytics types
export interface ConfidenceByClass {
  class_name: string;
  avg_confidence: number;
  std_confidence: number;
  p25_confidence: number;
  p75_confidence: number;
  count: number;
  reliability_score: number;
}

export interface DailyHealthMetric {
  date: string;
  uploads: number;
  detections: number;
  avg_processing_ms: number;
  p95_processing_ms: number;
  avg_confidence: number;
  high_confidence_rate: number;
  detections_per_upload: number;
}

export interface UserEngagement {
  user: string;
  total_uploads: number;
  active_days: number;
  activity_rate_pct: number;
  uploads_per_active_day: number;
  avg_hours_between_uploads: number;
  engagement_score: number;
  first_upload: string | null;
  last_upload: string | null;
}

export interface TemporalPatterns {
  hourly: {
    hour: number;
    uploads: number;
    avg_processing_ms: number;
  }[];
  daily: {
    day_of_week: number;
    day_name: string;
    uploads: number;
    avg_processing_ms: number;
  }[];
}

// New high-value analytics types
export interface ResolutionPerformance {
  assignee_performance: {
    assignee: string;
    total_assigned: number;
    resolved_count: number;
    resolution_rate_pct: number;
    avg_resolution_hours: number;
    p95_resolution_hours: number;
  }[];
  verification_performance: {
    verifier: string;
    verified_count: number;
    avg_verification_lag_hours: number;
  }[];
  workload_distribution: {
    assignee: string;
    total_assigned: number;
    open: number;
    resolved: number;
    ignored: number;
    completion_rate: number;
  }[];
}

export interface DetectionQuality {
  overview: {
    total_detections: number;
    false_positive_rate: number;
    verification_rate: number;
    avg_confidence: number;
    avg_ignored_confidence: number;
    avg_resolved_confidence: number;
  };
  by_confidence_range: {
    confidence_range: string;
    total_count: number;
    ignored_count: number;
    resolved_count: number;
    false_positive_rate: number;
  }[];
  by_class: {
    class_name: string;
    total_count: number;
    verified_count: number;
    ignored_count: number;
    avg_confidence: number;
    false_positive_rate: number;
    verification_rate: number;
  }[];
  daily_trends: {
    date: string;
    total_detections: number;
    ignored_count: number;
    false_positive_rate: number;
    avg_confidence: number;
  }[];
}

export interface GeographicPatterns {
  geographic_clusters: {
    geohash: string;
    total_issues: number;
    unique_locations: number;
    issue_density: number;
    avg_resolution_hours: number;
    resolution_rate: number;
    severity_distribution: {
      high: number;
      medium: number;
      low: number;
    };
    latest_issue: string | null;
    lat: number | null;
    lon: number | null;
  }[];
  recurring_patterns: {
    geohash: string;
    class_name: string;
    issue_count: number;
    unique_days: number;
    recurrence_rate: number;
    first_occurrence: string | null;
    latest_occurrence: string | null;
    avg_confidence: number;
  }[];
  summary: {
    total_problem_areas: number;
    total_recurring_patterns: number;
    avg_issues_per_area: number;
  };
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

// New API functions
export async function getConfidenceByClass(params?: RangeParams & { min_detections?: number }) {
  const { data } = await client.get<ConfidenceByClass[]>("/analytics/detections/confidence-by-class", { params });
  return data;
}

export async function getDailyHealthMetrics(params?: RangeParams) {
  const { data } = await client.get<DailyHealthMetric[]>("/analytics/system/daily-health", { params });
  return data;
}

export async function getUserEngagementMetrics(params?: RangeParams) {
  const { data } = await client.get<UserEngagement[]>("/analytics/activity/user-engagement", { params });
  return data;
}

export async function getTemporalPatterns(params?: RangeParams) {
  const { data } = await client.get<TemporalPatterns>("/analytics/activity/temporal-patterns", { params });
  return data;
}

// High-value analytics API functions
export async function getResolutionPerformance(params?: RangeParams) {
  const { data } = await client.get<ResolutionPerformance>("/analytics/performance/resolution-efficiency", { params });
  return data;
}

export async function getDetectionQuality(params?: RangeParams) {
  const { data } = await client.get<DetectionQuality>("/analytics/quality/detection-quality", { params });
  return data;
}

export async function getGeographicPatterns(params?: RangeParams & { precision?: number }) {
  const { data } = await client.get<GeographicPatterns>("/analytics/geography/issue-patterns", { params });
  return data;
}
