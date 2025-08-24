import { useEffect, useMemo, useRef, useState } from 'react';
import {
  uploadsByDay, uploadsByUser,
  getKPIs, getLatencyByDay, getSeverityByDay,
  getSourceBreakdown, getStatusBreakdown, getTopClasses,
  getConfidenceSummary, getGeoHeatmap, getTimeToResolution,
  getGeoHotspots, getIssuesAgingBuckets,
  getConfidenceByClass, getDailyHealthMetrics, getUserEngagementMetrics, getTemporalPatterns,
  getResolutionPerformance, getDetectionQuality, getGeographicPatterns,
  type DayStat, type UserStat, type KPIResponse, type LatencyByDay,
  type SeverityByDay, type SourceBreakdown, type StatusBreakdown,
  type TopClass, type ConfidenceSummary, type HeatBucket, type TTRow,
  type MediaTypeFilter, type Hotspot, type AgingResponse, type AgingByAssigneeRow, type AgingBucketKey,
  type ConfidenceByClass, type DailyHealthMetric, type UserEngagement, type TemporalPatterns,
  type ResolutionPerformance, type DetectionQuality, type GeographicPatterns
} from '../api/analytics';
import {
  ResponsiveContainer,
  AreaChart, Area, LineChart, Line,
  CartesianGrid, XAxis, YAxis, Tooltip, BarChart, Bar, Cell,
  ComposedChart, PieChart, Pie, Legend
} from 'recharts';

import { notify } from '../lib/notify';
import '../styles/analytics.css';

type LoadState = 'idle'|'loading'|'ready'|'error';

const SLA_DEFAULTS = { low: 168, medium: 72, high: 24 };

function fmtDayLabel(iso: string) {
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
function isoDaysRange(days = 7): string[] {
  const out: string[] = [];
  const end = new Date();
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(end);
    d.setDate(end.getDate() - i);
    out.push(d.toISOString().slice(0,10));
  }
  return out;
}
function fillMissingDays(raw: DayStat[], days = 7): DayStat[] {
  const map = new Map(raw.map(r => [r.date, r.count]));
  return isoDaysRange(days).map(date => ({ date, count: map.get(date) ?? 0 }));
}
const sum = (arr: number[]) => arr.reduce((a,b)=>a+b,0);

function useCountUp(value: number, duration = 600) {
  const [display, setDisplay] = useState(0);
  const raf = useRef<number | null>(null);
  const startTs = useRef<number>(0);
  const from = useRef(0);

  useEffect(() => {
    cancelAnimationFrame(raf.current ?? 0);
    startTs.current = 0;
    from.current = display;

    const step = (ts: number) => {
      if (!startTs.current) startTs.current = ts;
      const t = Math.min(1, (ts - startTs.current) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(from.current + (value - from.current) * eased);
      if (t < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current ?? 0);
  }, [value]);

  return display;
}

function getErrorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === 'string') return e;
  try { return JSON.stringify(e); } catch { return 'Unknown error'; }
}

function timeAgo(iso: string) {
  const ms = Date.now() - new Date(iso).getTime();
  if (!isFinite(ms)) return '—';
  const mins = Math.floor(ms / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

type RangeOpt = 7 | 30 | 90;

const BUCKETS: AgingBucketKey[] = ["0-24h","1-3d","3-7d","7-30d",">30d"];

function sevOrder(s: "low"|"medium"|"high"): number {
  return s === "high" ? 0 : s === "medium" ? 1 : 2;
}


export default function AnalyticsPage() {
  const [range, setRange] = useState<RangeOpt>(7);
  const [media, setMedia] = useState<"all" | MediaTypeFilter>("all");
  const mediaParam: MediaTypeFilter = media === "all" ? undefined : media;

  const [state, setState] = useState<LoadState>('idle');
  
  const [daily, setDaily] = useState<DayStat[]>([]);
  const [byUser, setByUser] = useState<UserStat[]>([]);
  const [kpis, setKPIs] = useState<KPIResponse | null>(null);
  const [latency, setLatency] = useState<LatencyByDay[]>([]);
  const [sevByDay, setSevByDay] = useState<SeverityByDay[]>([]);
  const [srcBreak, setSrcBreak] = useState<SourceBreakdown[]>([]);
  const [stBreak, setStBreak] = useState<StatusBreakdown[]>([]);
  const [topClasses, setTopClasses] = useState<TopClass[]>([]);
  const [conf, setConf] = useState<ConfidenceSummary | null>(null);
  const [heat, setHeat] = useState<HeatBucket[]>([]);
  const [tTRows, setTTRows] = useState<TTRow[]>([]);
  
  const [precision, setPrecision] = useState<4 | 5 | 6>(6);
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);

  const [aging, setAging] = useState<AgingResponse | null>(null);

  const [confidenceByClass, setConfidenceByClass] = useState<ConfidenceByClass[]>([]);
  const [healthMetrics, setHealthMetrics] = useState<DailyHealthMetric[]>([]);
  const [userEngagement, setUserEngagement] = useState<UserEngagement[]>([]);
  const [timePatterns, setTimePatterns] = useState<TemporalPatterns | null>(null);

  const [resolutionPerf, setResolutionPerf] = useState<ResolutionPerformance | null>(null);
  const [detectionQuality, setDetectionQuality] = useState<DetectionQuality | null>(null);
  const [geoPatterns, setGeoPatterns] = useState<GeographicPatterns | null>(null);
  
  async function load() {
    setState('loading');
    try {
      const params = { days: range, media_type: mediaParam };
      const [
        d, u, kp, lat, sev, src, st, top, cnf, gh, tt, hs, ag, cbcl, hm, ue, tp, rp, dq, gp
      ] = await Promise.all([
        uploadsByDay(params),
        uploadsByUser(params),
        getKPIs(params),
        getLatencyByDay(params),
        getSeverityByDay(params),
        getSourceBreakdown(params),
        getStatusBreakdown(params),
        getTopClasses({ ...params, limit: 10 }),
        getConfidenceSummary(params),
        getGeoHeatmap({ ...params, min_count: 1 }),
        getTimeToResolution(params),
        getGeoHotspots({ ...params, precision, min_count: 1, limit: 300 }),
        getIssuesAgingBuckets({ ...params }),
        getConfidenceByClass({ ...params, min_detections: 2 }),
        getDailyHealthMetrics(params),
        getUserEngagementMetrics(params),
        getTemporalPatterns(params),
        getResolutionPerformance(params),
        getDetectionQuality(params),
        getGeographicPatterns({ ...params, precision: 5 })
      ]);
      setDaily(fillMissingDays(d, range));
      setByUser(u.slice().sort((a,b)=>b.count-a.count));
      setKPIs(kp);
      setLatency(lat);
      setSevByDay(sev);
      setSrcBreak(src);
      setStBreak(st);
      setTopClasses(top);
      setConf(cnf);
      setHeat(gh);
      setTTRows(tt);
      setHotspots(hs);
      setAging(ag);
      setConfidenceByClass(cbcl);
      setHealthMetrics(hm);
      setUserEngagement(ue);
      setTimePatterns(tp);
      setResolutionPerf(rp);
      setDetectionQuality(dq);
      setGeoPatterns(gp);
      setState('ready');
    } catch (e: unknown) {
      notify.error('Failed to load analytics', getErrorMessage(e));
      setState('error');
    }
  }

  useEffect(() => { load();   }, [range, media, precision]);
  
  const { total7, avgDay, top, trendPct } = useMemo(() => {
    const total7 = sum(daily.map(d => d.count));
    const avgDay = total7 / Math.max(daily.length, 1);
    const top = byUser[0] ?? { user: '—', count: 0 };
    const lastN = 3;
    const last3 = sum(daily.slice(-lastN).map(d=>d.count));
    const prev3 = sum(daily.slice(-2*lastN, -lastN).map(d=>d.count));
    const trendPct = prev3 === 0 ? (last3 > 0 ? 100 : 0) : ((last3 - prev3) / prev3) * 100;
    return { total7, avgDay, top, trendPct };
  }, [daily, byUser]);
  
  const sevTotals = useMemo(() => {
    return sevByDay.reduce(
      (a, d) => ({ low: a.low + d.low, medium: a.medium + d.medium, high: a.high + d.high }),
      { low: 0, medium: 0, high: 0 }
    );
  }, [sevByDay]);
  const sevSum = Math.max(1, sevTotals.low + sevTotals.medium + sevTotals.high);
  const sevPct = {
    low: Math.round((100 * sevTotals.low) / sevSum),
    med: Math.round((100 * sevTotals.medium) / sevSum),
    high: Math.round((100 * sevTotals.high) / sevSum),
  };
  
  const imgCount = kpis?.uploads.images ?? 0;
  const vidCount = kpis?.uploads.videos ?? 0;
  const upTot = Math.max(1, imgCount + vidCount);
  const imgPct = Math.round((100 * imgCount) / upTot);
  const vidPct = 100 - imgPct;

  const totalAnim = Math.round(useCountUp(total7));
  const avgAnim = useCountUp(avgDay);
  const topShare = total7 > 0 ? Math.round((100 * top.count) / total7) : 0;

  const isEmpty = state === 'ready'
    && daily.every(d=>d.count===0)
    && (kpis?.detections.total ?? 0) === 0;

  const maxUser = byUser.reduce((m,u)=>Math.max(m,u.count),0);

  function downloadCSV() {
    const push = (title: string, rows: string[]) => [`# ${title}`, ...rows, ''].join('\n');
    const dailyRows = ['date,count', ...daily.map(d => `${d.date},${d.count}`)];
    const userRows  = ['user,count', ...byUser.map(u => `${u.user},${u.count}`)];
    const latencyRows = ['date,avg_ms,p95_ms,count', ...latency.map(r=>`${r.date},${r.avg_ms},${r.p95_ms},${r.count}`)];
    const sevRows = ['date,low,medium,high', ...sevByDay.map(r=>`${r.date},${r.low},${r.medium},${r.high}`)];
    const srcRows = ['source,count', ...srcBreak.map(r=>`${r.source},${r.count}`)];
    const stRows  = ['status,count', ...stBreak.map(r=>`${r.status},${r.count}`)];
    const topRows = ['class_name,count', ...topClasses.map(r=>`${r.class_name},${r.count}`)];
    const confRows = conf ? ['metric,value', `avg,${conf.avg}`, `p05,${conf.p05}`, `p50,${conf.p50}`, `p95,${conf.p95}`] : [];
    const ttrRows = ['severity,avg_hours,p95_hours,count', ...tTRows.map(r=>`${r.severity},${r.avg_hours},${r.p95_hours},${r.count}`)];
    const heatRows = ['geohash6,count,latest', ...heat.map(h=>`${h.geohash6},${h.count},${h.latest}`)];
    const hotspotRows = ['geohash,precision,count,prev_count,trend_pct,latest,uploaders,low,medium,high,top1,top1_count,top2,top2_count,top3,top3_count',
      ...hotspots.map(h=>{
        const [t1,t2,t3] = (h.top_classes ?? []);
        return [
          h.geohash, h.precision, h.count, h.prev_count, Math.round(h.trend_pct),
          h.latest ?? '', h.uploaders,
          h.severity.low, h.severity.medium, h.severity.high,
          t1?.class_name ?? '', t1?.count ?? '',
          t2?.class_name ?? '', t2?.count ?? '',
          t3?.class_name ?? '', t3?.count ?? '',
        ].join(',');
      })
    ];

    const content = [
      push('uploads-by-day', dailyRows),
      push('uploads-by-user', userRows),
      push('latency-by-day', latencyRows),
      push('severity-by-day', sevRows),
      push('source-breakdown', srcRows),
      push('status-breakdown', stRows),
      push('top-classes', topRows),
      confRows.length ? push('confidence-summary', confRows) : '',
      push('time-to-resolution', ttrRows),
      push('geo-heatmap', heatRows),
      push('geo-hotspots', hotspotRows),
    ].filter(Boolean).join('\n');

    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics-${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="analytics">
      {/* Header */}
      <header className="analytics__header">
        <div className="title-stack">
          <div className="eyebrow">Dashboard</div>
          <h1 className="grad-title">Analytics</h1>
          <p className="muted">Trends, quality, and processing across your inference pipeline</p>
        </div>

        <div className="header-actions">
          {/* Range */}
          <div className="segmented" role="group" aria-label="Time range">
            {[7,30,90].map((d)=>(
              <button
                key={d}
                type="button"
                className={range===d ? 'active' : ''}
                onClick={()=>setRange(d as RangeOpt)}
              >{d}d</button>
            ))}
          </div>

          {/* Media filter */}
          <div className="segmented" role="group" aria-label="Media type">
            {(['all','image','video'] as const).map((m)=>(
              <button
                key={m}
                type="button"
                className={media===m ? 'active' : ''}
                onClick={()=>setMedia(m)}
              >{m[0].toUpperCase()+m.slice(1)}</button>
            ))}
          </div>

          <button className="btn btn-ghost" onClick={load} title="Refresh">↻ Refresh</button>
          <button className="btn btn-ghost" onClick={downloadCSV} title="Export CSV">
            <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden><path d="M12 3v12m0 0 4-4m-4 4-4-4M4 21h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            Export CSV
          </button>
        </div>
      </header>

      {/* Window chip */}
      <div className="window-chip-wrap">
        <span className="badge">
          Window:&nbsp;
          {kpis ? new Date(kpis.window.start).toLocaleDateString() : '—'}
          &nbsp;→&nbsp;
          {kpis ? new Date(kpis.window.end).toLocaleDateString() : '—'}
        </span>
      </div>

      {/* KPI tiles */}
      <section className="kpis">
        {['loading','idle'].includes(state) ? (
          Array.from({length:6}).map((_,i)=>(
            <div key={i} className="kpi kpi--skeleton" aria-hidden>
              <div className="sk-line" />
              <div className="sk-num" />
            </div>
          ))
        ) : (
          <>
            {/* Uploads */}
            <div className="kpi kpi--accent">
              <div className="kpi__top">
                <span className="kpi__label">Uploads ({range}d)</span>
                <TrendChip value={trendPct}/>
              </div>
              <span className="kpi__value" aria-live="polite">{totalAnim}</span>
              <MiniSparkline data={daily.map(d=>d.count)} />
            </div>

            {/* Images / Videos with split bar */}
            <div className="kpi">
              <span className="kpi__label">Images / Videos</span>
              <span className="kpi__value">
                {kpis ? `${imgCount} / ${vidCount}` : '—'}
              </span>
              <div className="split-bar" title={`${imgCount} images / ${vidCount} videos`} aria-label="Images vs Videos">
                <i className="seg seg--img" style={{ width: `${imgPct}%` }} />
                <i className="seg seg--vid" style={{ width: `${vidPct}%` }} />
              </div>
              <small className="muted">{imgPct}% images · {vidPct}% videos</small>
            </div>

            {/* Avg/day */}
            <div className="kpi">
              <span className="kpi__label">Avg uploads / day</span>
              <span className="kpi__value" aria-live="polite">{avgAnim.toFixed(1)}</span>
              <small className="muted">based on last {range} days</small>
            </div>

            {/* Latency with sparkline */}
            <div className="kpi">
              <div className="kpi__top">
                <span className="kpi__label">Processing avg / p95</span>
              </div>
              <span className="kpi__value">
                {kpis ? `${Math.round(kpis.latency_ms.avg)} / ${Math.round(kpis.latency_ms.p95)} ms` : '—'}
              </span>
              <MiniSparkline data={latency.map(l => l.avg_ms)} colorVar="var(--secondary-500)" />
            </div>

            {/* Detections with mini severity mix */}
            <div className="kpi">
              <span className="kpi__label">Detections</span>
              <span className="kpi__value">{kpis?.detections.total ?? '—'}</span>
              <div className="sev-bar sev-bar--tiny" title={`low ${sevTotals.low} / medium ${sevTotals.medium} / high ${sevTotals.high}`}>
                <i className="sev s-low"  style={{ width: `${sevPct.low}%` }} />
                <i className="sev s-med"  style={{ width: `${sevPct.med}%` }} />
                <i className="sev s-high" style={{ width: `${sevPct.high}%` }} />
              </div>
              <small className="muted">{sevPct.low}% low · {sevPct.med}% med · {sevPct.high}% high</small>
            </div>

            {/* Top uploader */}
            <div className="kpi">
              <span className="kpi__label">Top uploader</span>
              <span className="kpi__value">{(byUser[0]?.user) ?? '—'}</span>
              <small className="chip chip--soft">{topShare}% of uploads</small>
            </div>
          </>
        )}
      </section>

      {state === 'error' && (
        <div className="card error-banner" role="alert">
          Couldn’t load analytics. Try again later.
        </div>
      )}

      {isEmpty && (
        <div className="card empty">
          <svg width="36" height="36" viewBox="0 0 24 24" aria-hidden>
            <path d="M3 7h18M5 7v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7M9 7V5a3 3 0 0 1 6 0v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <p>No data yet for the selected range.</p>
        </div>
      )}

      {!isEmpty && (
        <>
          {/* Uploads over time */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Uploads — last {range} days</h2>
              <span className="muted">Daily volume</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? <div className="chart-skeleton" /> : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={daily} margin={{ left: 8, right: 8, top: 6, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradPrimary" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--primary-500)" stopOpacity={0.45}/>
                        <stop offset="100%" stopColor="var(--primary-500)" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={fmtDayLabel}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }} height={36}/>
                    <YAxis allowDecimals={false} width={36}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }}/>
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{
                        borderRadius: 12, border: '1px solid var(--surface-200)',
                        background: 'var(--surface-0)', boxShadow: 'var(--shadow-md)', color: 'var(--on-surface)'
                      }}
                      labelStyle={{ color: 'var(--on-surface)', fontWeight: 700 }}
                      itemStyle={{ color: 'var(--on-surface)' }}
                      labelFormatter={(l)=>`📅 ${fmtDayLabel(String(l))}`}
                    />
                    <Area type="monotone" dataKey="count" stroke="var(--primary-600)" strokeWidth={2} fill="url(#gradPrimary)" activeDot={{ r: 5 }}/>
                    <Line type="monotone" dataKey="count" stroke="var(--primary-700)" strokeWidth={1.25} dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>

          {/* Issue aging + SLA */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Issue aging &amp; SLA</h2>
              <span className="muted">
      Buckets by age (open issues). SLA (h): L {aging?.sla_hours.low ?? SLA_DEFAULTS.low}
                &nbsp;· M {aging?.sla_hours.medium ?? SLA_DEFAULTS.medium}
                &nbsp;· H {aging?.sla_hours.high ?? SLA_DEFAULTS.high}
    </span>

            </div>


            {/* Table: by severity */}
            <div className="aging-scroll">
              <div className="aging-table">
                <div className="aging-row aging-head">
                  <span>Severity</span>
                  {BUCKETS.map(b => <span key={b} className="right">{b}</span>)}
                  <span className="right">Open</span>
                  <span className="right">SLA breach</span>
                  <span className="right">% breach</span>
                </div>

                {(["high", "medium", "low"] as const).map(sev => {
                  const r = aging?.by_severity?.[sev];
                  const open = r?.total ?? 0;
                  const breach = aging?.sla_breach_open?.[sev] ?? 0;
                  const rate = aging ? Math.round((aging.sla_breach_rate?.[sev] ?? 0)) : 0;

                  // NEW: clamp & style for the meter fill
                  const clamped = open ? Math.min(100, Math.max(0, rate)) : 0;
                  const ratioStyle: React.CSSProperties & { ['--ratio']?: string } = { '--ratio': `${clamped}%` };

                  return (
                    <div key={sev} className="aging-row">
                      <span className={`sev-label sev-${sev}`}>{sev}</span>
                      {BUCKETS.map(b => {
                        const v = r ? r[b] : 0;
                        return <span key={b} className={`right num ${v === 0 ? 'muted-0' : ''}`}>{v}</span>;
                      })}
                      <span className={`right num ${open === 0 ? 'muted-0' : ''}`}><strong>{open}</strong></span>
                      <span className={`right num ${breach === 0 ? 'muted-0' : ''}`}>{breach}</span>

                      {/* NEW: % breach meter pill */}
                      <span className="right">
                        <span className="rate-pill" style={ratioStyle}>
                          <span>{open ? `${Math.round(rate)}%` : '—'}</span>
                        </span>
                      </span>
                    </div>
                  );
                })}

              </div>
            </div>

            {/* Table: by assignee & severity */}
            <div className="section-subhead" style={{marginTop: 12}}>
              <span className="muted">Split by assignee × severity</span>
            </div>
            <div className="aging-scroll">
              <div className="aging-table assignee">
                <div className="aging-row aging-head">
                  <span>Assignee</span>
                  <span>Severity</span>
                  {BUCKETS.map(b => <span key={b} className="right">{b}</span>)}
                  <span className="right">Total</span>
                </div>

                {(aging?.by_assignee ?? [])
                    .slice()
                    .sort((a, b) => (a.assignee ?? "").localeCompare(b.assignee ?? "") || sevOrder(a.severity) - sevOrder(b.severity))
                    .map((row: AgingByAssigneeRow, i: number) => (
                        <div key={`${row.assignee ?? '(unassigned)'}:${row.severity}:${i}`} className="aging-row">
                          <span className="mono">{row.assignee ?? "(unassigned)"}</span>
                          <span className={`sev-label sev-${row.severity}`}>{row.severity}</span>
                          {BUCKETS.map(b => {
                            const v = row.buckets[b] ?? 0;
                            return <span key={b} className={`right num ${v === 0 ? 'muted-0' : ''}`}>{v}</span>;
                          })}
                          <span className={`right num ${row.total === 0 ? 'muted-0' : ''}`}><strong>{row.total}</strong></span>

                        </div>
                    ))
                }

                {(!aging || aging.by_assignee.length === 0) && (
                    <div className="muted" style={{padding: 6}}>No open issues in this window.</div>
                )}
              </div>
            </div>
          </section>


          {/* Latency + Severity over time */}
          <section className="grid-2">
            <div className="card card--glow">
              <div className="section-head">
                <h2>Processing latency</h2>
                <span className="muted">Average & p95 (ms)</span>
              </div>
              <div className="chart-wrap">
                {state !== 'ready' ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={latency}>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                      <XAxis dataKey="date" tickFormatter={fmtDayLabel}
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                        axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }} height={36}/>
                      <YAxis width={42} tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                        axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }}/>
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                        labelFormatter={(l)=>`📅 ${fmtDayLabel(String(l))}`}
                      />
                      <Line type="monotone" dataKey="avg_ms" stroke="var(--secondary-500)" strokeWidth={2} dot={false} name="avg"/>
                      <Line type="monotone" dataKey="p95_ms" stroke="var(--primary-600)" strokeWidth={2} dot={false} name="p95"/>
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="card card--glow">
              <div className="section-head">
                <h2>Severity over time</h2>
                <span className="muted">Detections per day</span>
              </div>
              <div className="chart-wrap">
                {state !== 'ready' ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={sevByDay}>
                      <defs>
                        <linearGradient id="gradLow" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--sev-low)" stopOpacity={0.55}/>
                          <stop offset="100%" stopColor="var(--sev-low)" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="gradMed" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--sev-medium)" stopOpacity={0.55}/>
                          <stop offset="100%" stopColor="var(--sev-medium)" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="gradHigh" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--sev-high)" stopOpacity={0.55}/>
                          <stop offset="100%" stopColor="var(--sev-high)" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={fmtDayLabel}
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                        axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }} height={36}/>
                      <YAxis width={44}
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                        axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }}/>
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                        labelFormatter={(l)=>`📅 ${fmtDayLabel(String(l))}`}
                      />
                      <Area type="monotone" dataKey="low"    stackId="1" stroke="var(--sev-low)"    fill="url(#gradLow)" name="low"/>
                      <Area type="monotone" dataKey="medium" stackId="1" stroke="var(--sev-medium)" fill="url(#gradMed)" name="medium"/>
                      <Area type="monotone" dataKey="high"   stackId="1" stroke="var(--sev-high)"   fill="url(#gradHigh)" name="high"/>
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
              <div className="legend">
                <span><i className="dot dot--low" />Low</span>
                <span><i className="dot dot--med" />Medium</span>
                <span><i className="dot dot--high" />High</span>
              </div>
            </div>
          </section>

          {/* Pies + Top classes */}
          <section className="grid-3">
            <div className="card card--glow">
              <div className="section-head"><h2>Sources</h2><span className="muted">Model provenance</span></div>
              <div className="chart-wrap sm">
                {state !== 'ready' ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={srcBreak} dataKey="count" nameKey="source" innerRadius={48} outerRadius={78} paddingAngle={2}>
                        {srcBreak.map((s, i) => (
                          <Cell key={i} fill={
                            s.source === 'yolo' ? 'var(--secondary-500)' :
                            s.source === 'gpt_dino' ? 'var(--primary-600)' :
                            'var(--secondary-700)'
                          }/>
                        ))}
                      </Pie>
                      <Legend />
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)', color: 'var(--on-surface)' }}
                        labelStyle={{ color: 'var(--on-surface)' }}
                        itemStyle={{ color: 'var(--on-surface)' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="card card--glow">
              <div className="section-head"><h2>Status</h2><span className="muted">Lifecycle</span></div>
              <div className="chart-wrap sm">
                {state !== 'ready' ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={stBreak} dataKey="count" nameKey="status" innerRadius={48} outerRadius={78} paddingAngle={2}>
                        {stBreak.map((s, i) => (
                          <Cell key={i} fill={
                            s.status === 'open' ? 'var(--info)' :
                            s.status === 'resolved' ? 'var(--success)' :
                            'var(--warning)'
                          }/>
                        ))}
                      </Pie>
                      <Legend />
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)', color: 'var(--on-surface)' }}
                        labelStyle={{ color: 'var(--on-surface)' }}
                        itemStyle={{ color: 'var(--on-surface)' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="card card--glow">
              <div className="section-head"><h2>Top classes</h2><span className="muted">Most frequent</span></div>
              <div className="chart-wrap">
                {state !== 'ready' ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[...topClasses].reverse()} layout="vertical" margin={{ left: 80, right: 8, top: 6, bottom: 0 }}>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                      <XAxis type="number" tick={{ fill: 'var(--chart-axis)', fontSize: 12 }} />
                      <YAxis type="category" dataKey="class_name" width={80} tick={{ fill: 'var(--chart-axis)', fontSize: 12 }} />
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)', color: 'var(--on-surface)' }}
                        labelStyle={{ color: 'var(--on-surface)' }}
                        itemStyle={{ color: 'var(--on-surface)' }}
                      />
                      <Bar dataKey="count" radius={[8,8,8,8]} fill="url(#barGradPrimary)">
                        {topClasses.map((_, i) => <Cell key={i} fill="url(#barGradPrimary)" />)}
                      </Bar>
                      <defs>
                        <linearGradient id="barGradPrimary" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--primary-500)" stopOpacity="1" />
                          <stop offset="100%" stopColor="var(--primary-500)" stopOpacity="0.65" />
                        </linearGradient>
                      </defs>
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </section>

          {/* Confidence + Time to resolution */}
          <section className="grid-2">
            <div className="card card--glow">
              <div className="section-head"><h2>Detection confidence</h2><span className="muted">Avg / p05 / p50 / p95</span></div>
              <div className="confidence-grid">
                {conf ? (
                  <>
                    <div className="conf-pill"><span>avg</span><strong>{conf.avg.toFixed(2)}</strong></div>
                    <div className="conf-pill"><span>p05</span><strong>{conf.p05.toFixed(2)}</strong></div>
                    <div className="conf-pill"><span>p50</span><strong>{conf.p50.toFixed(2)}</strong></div>
                    <div className="conf-pill"><span>p95</span><strong>{conf.p95.toFixed(2)}</strong></div>
                  </>
                ) : <div className="chart-skeleton" style={{ height: 120 }} />}
              </div>
            </div>

            <div className="card card--glow">
              <div className="section-head"><h2>Time to resolution</h2><span className="muted">Avg / p95 (hours)</span></div>
              <div className="chart-wrap">
                {state !== 'ready' ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={tTRows}>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                      <XAxis dataKey="severity" tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <YAxis width={42} tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)', color: 'var(--on-surface)' }}
                        labelStyle={{ color: 'var(--on-surface)' }}
                        itemStyle={{ color: 'var(--on-surface)' }}
                      />
                      <Bar dataKey="avg_hours" name="avg" radius={[8,8,0,0]} fill="var(--secondary-500)"/>
                      <Line dataKey="p95_hours" name="p95" type="monotone" stroke="var(--primary-600)" strokeWidth={2} dot />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </section>

          {/* Contributors */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Top contributors</h2>
              <span className="muted">Uploads by user</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? <div className="chart-skeleton" /> : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={byUser}>
                    <defs>
                      <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--secondary-500)" stopOpacity="1" />
                        <stop offset="100%" stopColor="var(--secondary-500)" stopOpacity="0.65" />
                      </linearGradient>
                      <linearGradient id="barGradPrimary" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--primary-500)" stopOpacity="1" />
                        <stop offset="100%" stopColor="var(--primary-500)" stopOpacity="0.65" />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                    <XAxis dataKey="user" interval={0} angle={-15} textAnchor="end" height={52}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }}/>
                    <YAxis allowDecimals={false} width={36}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }} tickLine={{ stroke: 'var(--chart-axis)' }}/>
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                      labelStyle={{ color: 'var(--on-surface)', fontWeight: 700 }}
                      itemStyle={{ color: 'var(--on-surface)' }}
                      formatter={(v)=>[v,'Uploads']}
                    />
                    <Bar dataKey="count" radius={[10,10,0,0]}>
                      {byUser.map((u) => (
                        <Cell key={u.user} fill={u.count === maxUser ? 'url(#barGradPrimary)' : 'url(#barGrad)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>

          {/* Hotspots */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Hotspots</h2>
              <div className="muted" style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <span>Aggregated by</span>
                <div className="segmented" role="group" aria-label="Geohash precision">
                  {([4,5,6] as const).map((p)=>(
                    <button
                      key={p}
                      type="button"
                      className={precision===p ? 'active' : ''}
                      onClick={()=>setPrecision(p)}
                    >gh{p}</button>
                  ))}
                </div>
              </div>
            </div>

            <div className="hotspot-table">
              <div className="hotspot-row hotspot-head">
                <span>Area (geohash)</span>
                <span className="right">Count</span>
                <span className="right">Trend</span>
                <span className="grow">Severity mix</span>
                <span className="grow">Top classes</span>
                <span className="right">Latest</span>
                <span></span>
              </div>

              {(hotspots ?? []).map(h => {
                const totalSev = Math.max(1, h.severity.low + h.severity.medium + h.severity.high);
                const lowPct = (100 * h.severity.low) / totalSev;
                const medPct = (100 * h.severity.medium) / totalSev;
                const highPct = (100 * h.severity.high) / totalSev;

                const trend = isFinite(h.trend_pct) ? h.trend_pct : 0;
                const trendLabel = trend === 0 ? '—' : `${trend > 0 ? '▲' : '▼'} ${Math.abs(trend).toFixed(0)}%`;

                const bboxQ = h.bbox ? `${h.bbox[0].toFixed(6)},${h.bbox[1].toFixed(6)},${h.bbox[2].toFixed(6)},${h.bbox[3].toFixed(6)}` : '';
                const mapHref = h.bbox ? `/map?bbox=${encodeURIComponent(bboxQ)}` : `/map?geohash=${h.geohash}`;

                return (
                  <div key={`${h.precision}:${h.geohash}`} className="hotspot-row">
                    <span className="mono">{h.geohash}</span>
                    <span className="right"><strong>{h.count}</strong></span>
                    <span className={`right chip ${trend===0 ? 'chip--neutral' : (trend>0?'chip--up':'chip--down')}`}>{trendLabel}</span>

                    {/* centered severity bar */}
                    <span className="grow sev-cell">
                      <div className="sev-bar" title={`low ${h.severity.low} / medium ${h.severity.medium} / high ${h.severity.high}`}>
                        <i style={{ width: `${lowPct}%` }}  className="sev s-low" />
                        <i style={{ width: `${medPct}%` }}  className="sev s-med" />
                        <i style={{ width: `${highPct}%` }} className="sev s-high" />
                      </div>
                    </span>

                    <span className="grow hotspot-tags">
                      {(h.top_classes ?? []).map(tc => (
                        <span key={tc.class_name} className="tag">{tc.class_name} ×{tc.count}</span>
                      ))}
                      {(h.top_classes ?? []).length === 0 && <span className="muted">—</span>}
                    </span>

                    <span className="right">{h.latest ? timeAgo(h.latest) : '—'}</span>

                    <span>
                      <a className="btn btn-ghost" href={mapHref} title="View on map">View</a>
                    </span>
                  </div>
                );
              })}

              {(!hotspots || hotspots.length === 0) && (
                <div className="muted" style={{ padding: 6 }}>No hotspots for this filter.</div>
              )}
            </div>
          </section>

          {/* NEW: Model Performance & Insights */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Model Performance by Class</h2>
              <span className="muted">Detection confidence & reliability analysis</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? <div className="chart-skeleton" /> : (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={confidenceByClass.slice(0, 10)}>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                    <XAxis
                      dataKey="class_name"
                      angle={-45}
                      textAnchor="end"
                      height={60}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}
                    />
                    <YAxis yAxisId="conf" orientation="left" domain={[0, 1]}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <YAxis yAxisId="count" orientation="right"
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                      formatter={(value, name) => {
                        if (name === 'avg_confidence') return [Number(value).toFixed(3), 'Avg Confidence'];
                        if (name === 'reliability_score') return [Number(value).toFixed(3), 'Reliability Score'];
                        return [value, name];
                      }}
                    />
                    <Bar yAxisId="count" dataKey="count" fill="var(--surface-300)" name="Count" opacity={0.3} />
                    <Line yAxisId="conf" type="monotone" dataKey="avg_confidence" stroke="var(--primary-600)" strokeWidth={2} dot />
                    <Line yAxisId="conf" type="monotone" dataKey="reliability_score" stroke="var(--success)" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="legend">
              <span><i className="dot" style={{backgroundColor: 'var(--primary-600)'}} />Avg Confidence</span>
              <span><i className="dot" style={{backgroundColor: 'var(--success)'}} />Reliability</span>
              <span><i className="dot" style={{backgroundColor: 'var(--surface-300)'}} />Count</span>
            </div>
          </section>

          {/* NEW: System Health Trends */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Daily System Health</h2>
              <span className="muted">Processing performance and detection quality over time</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? <div className="chart-skeleton" /> : (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={healthMetrics.slice(-14)}>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                    <XAxis
                      dataKey="date"
                      tickFormatter={fmtDayLabel}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}
                    />
                    <YAxis yAxisId="uploads" orientation="left"
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <YAxis yAxisId="confidence" orientation="right" domain={[0, 1]}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                      formatter={(value, name) => {
                        if (name === 'avg_confidence') return [Number(value).toFixed(3), 'Avg Confidence'];
                        if (name === 'high_confidence_rate') return [Math.round(Number(value)) + '%', 'High Confidence Rate'];
                        if (name === 'avg_processing_ms') return [Math.round(Number(value)) + 'ms', 'Avg Processing Time'];
                        if (name === 'detections_per_upload') return [Number(value).toFixed(2), 'Detections/Upload'];
                        return [value, name];
                      }}
                      labelFormatter={(l)=>`📅 ${fmtDayLabel(String(l))}`}
                    />
                    <Bar yAxisId="uploads" dataKey="uploads" fill="var(--primary-300)" name="Daily Uploads" opacity={0.4} />
                    <Line yAxisId="confidence" type="monotone" dataKey="avg_confidence" stroke="var(--primary-600)" strokeWidth={2} dot />
                    <Line yAxisId="confidence" type="monotone" dataKey="high_confidence_rate" stroke="var(--success)" strokeWidth={2} dot />
                    <Line yAxisId="uploads" type="monotone" dataKey="avg_processing_ms" stroke="var(--warning)" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="legend">
              <span><i className="dot" style={{backgroundColor: 'var(--primary-300)'}} />Daily Uploads</span>
              <span><i className="dot" style={{backgroundColor: 'var(--primary-600)'}} />Avg Confidence</span>
              <span><i className="dot" style={{backgroundColor: 'var(--success)'}} />High Confidence Rate</span>
              <span><i className="dot" style={{backgroundColor: 'var(--warning)'}} />Avg Processing Time</span>
            </div>
          </section>

          {/* NEW: Activity Patterns */}
          <section className="grid-2">
            <div className="card card--glow">
              <div className="section-head">
                <h2>Daily Activity Pattern</h2>
                <span className="muted">Upload trends by day of week</span>
              </div>
              <div className="chart-wrap">
                {state !== 'ready' || !timePatterns ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={timePatterns.daily}>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                      <XAxis dataKey="day_name"
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <YAxis tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)', color: 'var(--on-surface)' }}
                        labelStyle={{ color: 'var(--on-surface)' }}
                        itemStyle={{ color: 'var(--on-surface)' }}
                      />
                      <Bar dataKey="uploads" fill="var(--primary-500)" radius={[4,4,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            <div className="card card--glow">
              <div className="section-head">
                <h2>Hourly Activity Pattern</h2>
                <span className="muted">Upload distribution by hour</span>
              </div>
              <div className="chart-wrap">
                {state !== 'ready' || !timePatterns ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timePatterns.hourly}>
                      <defs>
                        <linearGradient id="hourlyGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--secondary-500)" stopOpacity={0.6}/>
                          <stop offset="100%" stopColor="var(--secondary-500)" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                      <XAxis dataKey="hour"
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                        tickFormatter={(hour) => `${hour}:00`}/>
                      <YAxis tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                        labelFormatter={(hour) => `${hour}:00`}
                      />
                      <Area type="monotone" dataKey="uploads" stroke="var(--secondary-600)" strokeWidth={2} fill="url(#hourlyGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </section>

          {/* NEW: User Engagement */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>User Engagement Analysis</h2>
              <span className="muted">Activity patterns and engagement scores</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? <div className="chart-skeleton" /> : (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={userEngagement.slice(0, 15)}>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                    <XAxis dataKey="user" angle={-45} textAnchor="end" height={60}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}/>
                    <YAxis yAxisId="uploads" orientation="left"
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <YAxis yAxisId="score" orientation="right" domain={[0, 100]}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                      formatter={(value, name) => {
                        if (name === 'engagement_score') return [Math.round(Number(value)), 'Engagement Score'];
                        if (name === 'activity_rate_pct') return [Math.round(Number(value)) + '%', 'Activity Rate'];
                        return [value, name];
                      }}
                    />
                    <Bar yAxisId="uploads" dataKey="total_uploads" fill="var(--primary-500)" opacity={0.7} />
                    <Line yAxisId="score" type="monotone" dataKey="engagement_score" stroke="var(--success)" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="legend">
              <span><i className="dot" style={{backgroundColor: 'var(--primary-500)'}} />Total Uploads</span>
              <span><i className="dot" style={{backgroundColor: 'var(--success)'}} />Engagement Score</span>
            </div>
          </section>

          {/* NEW: Resolution Performance Analytics */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Team Resolution Performance</h2>
              <span className="muted">Assignee efficiency and workload distribution</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' || !resolutionPerf ? <div className="chart-skeleton" /> : (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={resolutionPerf.assignee_performance.slice(0, 10)}>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                    <XAxis dataKey="assignee" angle={-45} textAnchor="end" height={60}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 11 }}/>
                    <YAxis yAxisId="count" orientation="left"
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <YAxis yAxisId="rate" orientation="right" domain={[0, 100]}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                      formatter={(value, name) => {
                        if (name === 'resolution_rate_pct') return [Math.round(Number(value)) + '%', 'Resolution Rate'];
                        if (name === 'avg_resolution_hours') return [Math.round(Number(value)) + 'h', 'Avg Resolution Time'];
                        return [value, name];
                      }}
                    />
                    <Bar yAxisId="count" dataKey="total_assigned" fill="var(--surface-300)" name="Total Assigned" opacity={0.4} />
                    <Line yAxisId="rate" type="monotone" dataKey="resolution_rate_pct" stroke="var(--success)" strokeWidth={2} dot />
                    <Line yAxisId="count" type="monotone" dataKey="avg_resolution_hours" stroke="var(--warning)" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="legend">
              <span><i className="dot" style={{backgroundColor: 'var(--surface-300)'}} />Total Assigned</span>
              <span><i className="dot" style={{backgroundColor: 'var(--success)'}} />Resolution Rate %</span>
              <span><i className="dot" style={{backgroundColor: 'var(--warning)'}} />Avg Resolution Hours</span>
            </div>
          </section>

          {/* NEW: Detection Quality Analysis */}
          <section className="grid-2">
            <div className="card card--glow">
              <div className="section-head">
                <h2>Detection Quality Overview</h2>
                <span className="muted">False positive rates and verification metrics</span>
              </div>
              <div className="confidence-grid">
                {!detectionQuality ? <div className="chart-skeleton" style={{ height: 120 }} /> : (
                  <>
                    <div className="conf-pill">
                      <span>False Positive Rate</span>
                      <strong>{detectionQuality.overview.false_positive_rate.toFixed(1)}%</strong>
                    </div>
                    <div className="conf-pill">
                      <span>Verification Rate</span>
                      <strong>{detectionQuality.overview.verification_rate.toFixed(1)}%</strong>
                    </div>
                    <div className="conf-pill">
                      <span>Avg Confidence</span>
                      <strong>{detectionQuality.overview.avg_confidence.toFixed(3)}</strong>
                    </div>
                    <div className="conf-pill">
                      <span>Total Detections</span>
                      <strong>{detectionQuality.overview.total_detections}</strong>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="card card--glow">
              <div className="section-head">
                <h2>Quality by Confidence Range</h2>
                <span className="muted">False positive rates across confidence levels</span>
              </div>
              <div className="chart-wrap">
                {state !== 'ready' || !detectionQuality ? <div className="chart-skeleton" /> : (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={detectionQuality.by_confidence_range}>
                      <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/>
                      <XAxis dataKey="confidence_range"
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <YAxis yAxisId="count" orientation="left"
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <YAxis yAxisId="rate" orientation="right" domain={[0, 100]}
                        tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}/>
                      <Tooltip
                        wrapperStyle={{ outline: 'none' }}
                        contentStyle={{ borderRadius: 12, border: '1px solid var(--surface-200)', background: 'var(--surface-0)' }}
                        formatter={(value, name) => {
                          if (name === 'false_positive_rate') return [Number(value).toFixed(1) + '%', 'False Positive Rate'];
                          return [value, name];
                        }}
                      />
                      <Bar yAxisId="count" dataKey="total_count" fill="var(--primary-300)" name="Total Count" opacity={0.5} />
                      <Line yAxisId="rate" type="monotone" dataKey="false_positive_rate" stroke="var(--error)" strokeWidth={3} dot connectNulls={true} />
                    </ComposedChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </section>

          {/* NEW: Geographic Issue Patterns */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Geographic Problem Areas</h2>
              <span className="muted">Issue clustering and recurring patterns by location</span>
            </div>
            <div className="hotspot-table">
              <div className="hotspot-row hotspot-head">
                <span>Area (geohash)</span>
                <span className="right">Total Issues</span>
                <span className="right">Issue Density</span>
                <span className="right">Resolution Rate</span>
                <span className="grow">Severity Distribution</span>
                <span className="right">Latest Issue</span>
              </div>

              {!geoPatterns ? <div className="chart-skeleton" style={{ height: 200 }} /> :
                geoPatterns.geographic_clusters.slice(0, 10).map((cluster) => {
                  const totalSev = Math.max(1, cluster.severity_distribution.low + cluster.severity_distribution.medium + cluster.severity_distribution.high);
                  const lowPct = (100 * cluster.severity_distribution.low) / totalSev;
                  const medPct = (100 * cluster.severity_distribution.medium) / totalSev;
                  const highPct = (100 * cluster.severity_distribution.high) / totalSev;

                  return (
                    <div key={cluster.geohash} className="hotspot-row">
                      <span className="mono">{cluster.geohash}</span>
                      <span className="right"><strong>{cluster.total_issues}</strong></span>
                      <span className="right">{cluster.issue_density}</span>
                      <span className="right">{Math.round(cluster.resolution_rate)}%</span>

                      <span className="grow sev-cell">
                        <div className="sev-bar" title={`low ${cluster.severity_distribution.low} / medium ${cluster.severity_distribution.medium} / high ${cluster.severity_distribution.high}`}>
                          <i style={{ width: `${lowPct}%` }} className="sev s-low" />
                          <i style={{ width: `${medPct}%` }} className="sev s-med" />
                          <i style={{ width: `${highPct}%` }} className="sev s-high" />
                        </div>
                      </span>

                      <span className="right">{cluster.latest_issue ? timeAgo(cluster.latest_issue) : '—'}</span>
                    </div>
                  );
                })
              }

              {(!geoPatterns || geoPatterns.geographic_clusters.length === 0) && (
                <div className="muted" style={{ padding: 6 }}>No geographic patterns found for this period.</div>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function TrendChip({ value }: { value: number }) {
  const up = value > 0;
  const flat = value === 0 || !isFinite(value);
  const label = flat ? '—' : `${up ? '▲' : '▼'} ${Math.abs(value).toFixed(0)}%`;
  return (
    <span className={`chip ${flat ? 'chip--neutral' : up ? 'chip--up' : 'chip--down'}`}>
      {label}
    </span>
  );
}

function MiniSparkline({ data, colorVar = 'var(--primary-600)' }: { data: number[]; colorVar?: string }) {
  const spark = data.map((v, i) => ({ i, v }));
  return (
    <div className="sparkline">
      <ResponsiveContainer width="100%" height={42}>
        <LineChart data={spark} margin={{ left: 0, right: 0, top: 6, bottom: 0 }}>
          <Line type="monotone" dataKey="v" stroke={colorVar} strokeWidth={1.75} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}