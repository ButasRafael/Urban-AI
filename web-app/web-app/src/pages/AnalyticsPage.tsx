import { useEffect, useMemo, useRef, useState } from 'react';
import {
  uploadsByDay,
  uploadsByUser,
  type DayStat,
  type UserStat,
} from '../api/analytics';
import {
  ResponsiveContainer,
  AreaChart, Area, Line, LineChart,
  XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, Cell,
} from 'recharts';
import { notify } from '../lib/notify';
import '../styles/analytics.css';

type LoadState = 'idle'|'loading'|'error'|'ready';

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

// tiny count-up hook for KPIs
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
      const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
      setDisplay(from.current + (value - from.current) * eased);
      if (t < 1) raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current ?? 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return display;
}

export default function AnalyticsPage() {
  const [daily, setDaily] = useState<DayStat[]>([]);
  const [byUser, setByUser] = useState<UserStat[]>([]);
  const [state, setState] = useState<LoadState>('idle');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setState('loading');
        const [d, u] = await Promise.all([uploadsByDay(), uploadsByUser()]);
        if (cancelled) return;
        setDaily(fillMissingDays(d, 7));
        setByUser(u.slice().sort((a,b)=>b.count-a.count));
        setState('ready');
      } catch (e: any) {
        notify.error('Failed to load analytics', e?.message ?? 'Unknown error');
        setState('error');
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // KPIs (derived)
  const { total7, avgDay, peak, activeUsers, top, trendPct } = useMemo(() => {
    const total7 = sum(daily.map(d => d.count));
    const avgDay = total7 / Math.max(daily.length, 1);
    const peak = daily.reduce((m, d) => d.count > m.count ? d : m, { date: '', count: -1 });
    const activeUsers = byUser.length;
    const top = byUser[0] ?? { user: '—', count: 0 };
    const last3 = sum(daily.slice(-3).map(d=>d.count));
    const prev3 = sum(daily.slice(-6, -3).map(d=>d.count));
    const trendPct = prev3 === 0 ? (last3 > 0 ? 100 : 0) : ((last3 - prev3) / prev3) * 100;
    return { total7, avgDay, peak, activeUsers, top, trendPct };
  }, [daily, byUser]);

  const total7Anim = Math.round(useCountUp(total7));
  const avgDayAnim = useCountUp(avgDay);
  const topShare = total7 > 0 ? Math.round((100 * top.count) / total7) : 0;

  function downloadCSV() {
    const dailyRows = ['date,count', ...daily.map(d => `${d.date},${d.count}`)].join('\n');
    const userRows  = ['user,count', ...byUser.map(u => `${u.user},${u.count}`)].join('\n');
    const blob = new Blob([`# uploads-by-day\n${dailyRows}\n\n# uploads-by-user\n${userRows}\n`], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics-${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const isEmpty = state === 'ready' && daily.every(d=>d.count===0) && byUser.length === 0;
  const maxUser = byUser.reduce((m,u)=>Math.max(m,u.count),0);

  return (
    <div className="analytics">
      <header className="analytics__header">
        <div className="title-stack">
          <div className="eyebrow">Dashboard</div>
          <h1 className="grad-title">Analytics</h1>
          <p className="muted">Last 7 days overview and top contributors</p>
        </div>
        <div className="header-actions">
          <div className="segmented" role="group" aria-label="Time range">
            <button className="active" type="button">7d</button>
            <button type="button" disabled>30d</button>
            <button type="button" disabled>90d</button>
          </div>
          <button className="btn btn-ghost" onClick={downloadCSV}>
            <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden><path d="M12 3v12m0 0 4-4m-4 4-4-4M4 21h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            Export CSV
          </button>
        </div>
      </header>

      {/* KPI tiles */}
      <section className="kpis">
        {['loading','idle'].includes(state) ? (
          Array.from({length:5}).map((_,i)=>(
            <div key={i} className="kpi kpi--skeleton" aria-hidden>
              <div className="sk-line" />
              <div className="sk-num" />
            </div>
          ))
        ) : (
          <>
            <div className="kpi kpi--accent">
              <div className="kpi__top">
                <span className="kpi__label">Uploads (7d)</span>
                <TrendChip value={trendPct}/>
              </div>
              <span className="kpi__value" aria-live="polite">{total7Anim}</span>
              <MiniSparkline data={daily.map(d=>d.count)} />
            </div>

            <div className="kpi">
              <span className="kpi__label">Avg / day</span>
              <span className="kpi__value" aria-live="polite">{avgDayAnim.toFixed(1)}</span>
              <small className="muted">based on last 7 days</small>
            </div>

            <div className="kpi">
              <span className="kpi__label">Peak day</span>
              <span className="kpi__value">
                {peak.count >= 0 ? (
                  <>
                    {fmtDayLabel(peak.date)} <small className="muted">({peak.count})</small>
                  </>
                ) : '—'}
              </span>
            </div>

            <div className="kpi">
              <span className="kpi__label">Active uploaders</span>
              <span className="kpi__value">{activeUsers}</span>
            </div>

            <div className="kpi">
              <span className="kpi__label">Top uploader</span>
              <span className="kpi__value">{top.user}</span>
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
          <p>No uploads yet in the last 7 days.</p>
        </div>
      )}

      {!isEmpty && (
        <>
          {/* Area chart: uploads last 7 days */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Uploads — last 7 days</h2>
              <span className="muted">Daily volume</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? (
                <div className="chart-skeleton" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={daily} margin={{ left: 8, right: 8, top: 6, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gradPrimary" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--primary-500)" stopOpacity={0.45}/>
                        <stop offset="100%" stopColor="var(--primary-500)" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={fmtDayLabel}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }}
                      tickLine={{ stroke: 'var(--chart-axis)' }}
                      height={36}
                    />
                    <YAxis
                      allowDecimals={false}
                      width={36}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }}
                      tickLine={{ stroke: 'var(--chart-axis)' }}
                    />
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{
                        borderRadius: 12,
                        border: '1px solid var(--surface-200)',
                        background: 'var(--surface-0)',
                        boxShadow: 'var(--shadow-md)',
                        color: 'var(--on-surface)',
                      }}
                      labelStyle={{ color: 'var(--on-surface)', fontWeight: 700 }}
                      itemStyle={{ color: 'var(--on-surface)' }}
                      labelFormatter={(l)=>`📅 ${fmtDayLabel(String(l))}`}
                    />
                    <Area
                      type="monotone"
                      dataKey="count"
                      stroke="var(--primary-600)"
                      strokeWidth={2}
                      fill="url(#gradPrimary)"
                      activeDot={{ r: 5 }}
                    />
                    <Line type="monotone" dataKey="count" stroke="var(--primary-700)" strokeWidth={1.25} dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>

          {/* Bar chart: top contributors */}
          <section className="card card--glow">
            <div className="section-head">
              <h2>Top contributors</h2>
              <span className="muted">Uploads by user</span>
            </div>
            <div className="chart-wrap">
              {state !== 'ready' ? (
                <div className="chart-skeleton" />
              ) : (
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
                    <XAxis
                      dataKey="user"
                      interval={0}
                      angle={-15}
                      textAnchor="end"
                      height={52}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }}
                      tickLine={{ stroke: 'var(--chart-axis)' }}
                    />
                    <YAxis
                      allowDecimals={false}
                      width={36}
                      tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                      axisLine={{ stroke: 'var(--chart-axis)' }}
                      tickLine={{ stroke: 'var(--chart-axis)' }}
                    />
                    <Tooltip
                      wrapperStyle={{ outline: 'none' }}
                      contentStyle={{
                        borderRadius: 12,
                        border: '1px solid var(--surface-200)',
                        background: 'var(--surface-0)',
                        boxShadow: 'var(--shadow-md)',
                        color: 'var(--on-surface)',
                      }}
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

function MiniSparkline({ data }: { data: number[] }) {
  const spark = data.map((v, i) => ({ i, v }));
  return (
    <div className="sparkline">
      <ResponsiveContainer width="100%" height={42}>
        <LineChart data={spark} margin={{ left: 0, right: 0, top: 6, bottom: 0 }}>
          <Line type="monotone" dataKey="v" stroke="var(--primary-600)" strokeWidth={1.75} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
