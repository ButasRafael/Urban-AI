import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import Input from "../components/Input";
import Button from "../components/Button";
import { notify } from "../lib/notify";
import { useAuth } from "../auth/useAuth";
import { searchUsers, type UserLite } from "../api/users";
import {
  listIssues,
  updateIssueStatus,
  updateIssueSeverity,
  assignIssue,
  verifyIssue,
  type Issue,
  type IssueStatus,
  type IssueSeverity,
  type MediaType,
  type IssueSource,
} from "../api/issues";
import "../styles/issues-page.css";

function useDebounced<T>(value: T, ms = 240) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}
function absUrl(path?: string | null): string | null {
  if (!path) return null;
  try {
    const base = import.meta.env.VITE_API_BASE as string | undefined;
    return base ? new URL(path, base).toString() : path;
  } catch {
    return path;
  }
}
type LoadState = "idle" | "loading" | "ready" | "error";

export default function IssuesPage() {
  const { user } = useAuth() as {
    user: { username?: string | null; role: "user" | "authority" | "admin" } | null;
  };

  const me = user?.username ?? "";
  const isAdmin = user?.role === "admin";
  const isAuthority = user?.role === "authority";

  const sp0 = new URLSearchParams(location.search);
  const initialMedia = sp0.get("type");
  const initialStatus = sp0.get("status");
  const initialSeverity = sp0.get("severity");
  const initialSource = sp0.get("source");
  const initialAssigneeMe = sp0.get("assignee") === "me";
  const initialMediaId = (() => {
  const v = new URLSearchParams(location.search).get("media_id");
  const n = v ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
})();

  const [mediaType, setMediaType] = useState<"all" | MediaType>(
    initialMedia === "image" || initialMedia === "video" ? (initialMedia as MediaType) : "all"
  );

  const [klassInput, setKlassInput] = useState(
    sp0.get("q") ?? sp0.get("klass_like") ?? sp0.get("klass") ?? ""
  );
  const klassLike = useDebounced(klassInput.trim(), 240);

  const [status, setStatus] = useState<"all" | IssueStatus>(
    initialStatus === "open" || initialStatus === "resolved" || initialStatus === "ignored"
      ? (initialStatus as IssueStatus)
      : "all"
  );
  const [severity, setSeverity] = useState<"all" | IssueSeverity>(
    initialSeverity === "low" || initialSeverity === "medium" || initialSeverity === "high"
      ? (initialSeverity as IssueSeverity)
      : "all"
  );
  const [source, setSource] = useState<"all" | IssueSource>(
    initialSource === "yolo" || initialSource === "gpt_dino" || initialSource === "sam_fallback"
      ? (initialSource as IssueSource)
      : "all"
  );

  const [assignedMine, setAssignedMine] = useState<boolean>(initialAssigneeMe && !!me);

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  const [allRows, setAllRows] = useState<Issue[]>([]);

  const [state, setState] = useState<LoadState>("idle");
  const [error, setError] = useState<string | null>(null);

  const searchRef = useRef<HTMLInputElement | null>(null);

  const [assignTarget, setAssignTarget] = useState<Issue | null>(null);
  const [verifyTarget, setVerifyTarget] = useState<Issue | null>(null);

  const [onlyMedia, setOnlyMedia] = useState<number | null>(initialMediaId);

  useEffect(() => {
  const sp = new URLSearchParams(location.search);
  if (klassLike) { sp.set("q", klassLike); sp.delete("klass_like"); sp.delete("klass"); }
  else { sp.delete("q"); sp.delete("klass_like"); sp.delete("klass"); }

  if (mediaType !== "all") sp.set("type", mediaType); else sp.delete("type");
  if (status   !== "all") sp.set("status", status);   else sp.delete("status");
  if (severity !== "all") sp.set("severity", severity); else sp.delete("severity");
  if (source   !== "all") sp.set("source", source);   else sp.delete("source");
  if (assignedMine && me) sp.set("assignee", "me"); else sp.delete("assignee");

  if (onlyMedia) sp.set("media_id", String(onlyMedia)); else sp.delete("media_id");

  history.replaceState(null, "", `${location.pathname}?${sp.toString()}`);
}, [klassLike, mediaType, status, severity, source, assignedMine, me, onlyMedia]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName?.toLowerCase();
      const inField = tag === "input" || tag === "textarea" || (el?.isContentEditable ?? false);
      if (e.key === "/" && !e.shiftKey && !inField) {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    setError(null);

    (async () => {
      try {
        const limit = 200;
        let offset = 0;
        const acc: Issue[] = [];

        for (;;) {
          const batch = await listIssues({
            media_type: mediaType === "all" ? undefined : mediaType,
            limit,
            offset,
          });
          if (cancelled) return;
          acc.push(...batch);
          if (batch.length < limit) break;
          offset += batch.length;

          if (acc.length > 10000) break;
        }

        setAllRows(acc);
        setState("ready");
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "Failed to load issues";
        setError(msg);
        notify.error("Failed to load issues", msg);
        setState("error");
      }
    })();

    return () => { cancelled = true; };
  }, [mediaType]);

  const filtered = useMemo(() => {
  const n = klassLike.toLowerCase();
  return allRows.filter((r) =>
    (!klassLike || r.class_name.toLowerCase().includes(n)) &&
    (status   === "all" || r.status   === status) &&
    (severity === "all" || r.severity === severity) &&
    (source   === "all" || r.source   === source) &&
    (!assignedMine || (me && r.assigned_to === me)) &&
    (!onlyMedia || r.media_id === onlyMedia) // <-- NEW
  );
}, [allRows, klassLike, status, severity, source, assignedMine, me, onlyMedia]);

  useEffect(() => { setPage(1); }, [mediaType, klassLike, status, severity, source, assignedMine, pageSize]);

  const statusCounts = useMemo(() => {
    let open = 0, resolved = 0, ignored = 0;
    for (const r of filtered) {
      if (r.status === "open") open++;
      else if (r.status === "resolved") resolved++;
      else ignored++;
    }
    return { open, resolved, ignored };
  }, [filtered]);

  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const pageRows = useMemo(
    () => filtered.slice((page - 1) * pageSize, page * pageSize),
    [filtered, page, pageSize]
  );
  const empty = state === "ready" && total === 0;
  const hasPrev = page > 1;
  const hasNext = page < totalPages;

  const changeStatus = useCallback(async (r: Issue, next: IssueStatus) => {
    if (r.status === next) return;
    if (next === "resolved" && r.assigned_to !== (user?.username ?? "")) {
      notify.error("Only the assignee can resolve this issue.");
      return;
    }
    const prev = r.status;
    setAllRows((arr) => arr.map((x) => (x.id === r.id ? { ...x, status: next } : x)));
    try {
      await updateIssueStatus(r.id, next);
      notify.success(`Status → ${next}`);
    } catch {
      setAllRows((arr) => arr.map((x) => (x.id === r.id ? { ...x, status: prev } : x)));
      notify.error("Failed to update status");
    }
  }, [user?.username]);

  const changeSeverity = useCallback(async (r: Issue, next: IssueSeverity) => {
    if (r.severity === next) return;
    const prev = r.severity;
    setAllRows((arr) => arr.map((x) => (x.id === r.id ? { ...x, severity: next } : x)));
    try {
      await updateIssueSeverity(r.id, next);
      notify.success(`Severity → ${next}`);
    } catch {
      setAllRows((arr) => arr.map((x) => (x.id === r.id ? { ...x, severity: prev } : x)));
      notify.error("Failed to update severity");
    }
  }, []);

  const openAssign = useCallback((r: Issue) => setAssignTarget(r), []);
  const closeAssign = useCallback(() => setAssignTarget(null), []);
  const applyAssign = useCallback(async (r: Issue, username: string | null) => {
    try {
      const { assigned_to } = await assignIssue(r.id, username);
      setAllRows((arr) => {
        const next = arr.map((x) => (x.id === r.id ? { ...x, assigned_to } : x));
        if (assignedMine && me && assigned_to !== me) {
          return next.filter((x) => x.id !== r.id);
        }
        return next;
      });
      notify.success(assigned_to ? `Assigned to @${assigned_to}` : "Unassigned");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to assign";
      notify.error(msg);
    }
  }, [assignedMine, me]);

  const openVerify = useCallback((r: Issue) => setVerifyTarget(r), []);
  const closeVerify = useCallback(() => setVerifyTarget(null), []);
  const applyVerify = useCallback(async (r: Issue) => {
    try {
      const resp = await verifyIssue(r.id, true);
      setAllRows((arr) =>
        arr.map((x) =>
          x.id === r.id ? { ...x, verified_by: resp.verified_by, verified_at: resp.verified_at } : x
        )
      );
      notify.success("Marked as verified");
    } catch {
      notify.error("Failed to verify");
    }
  }, []);

const clearFilters = useCallback(() => {
  setKlassInput("");
  setMediaType("all");
  setStatus("all");
  setSeverity("all");
  setSource("all");
  setAssignedMine(false);
  setOnlyMedia(null);
}, []);

  return (
    <div className="page issues-page">
      {/* Toolbar */}
      <div className="toolbar">
        <div className="left">
          <Input
            ref={searchRef}
            placeholder='Filter by class…'
            value={klassInput}
            onChange={(e) => setKlassInput(e.target.value)}
            stopGlobalKeys
            prefixIcon={<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></svg>}
            clearable
          />

          <div className="selectWrap">
            <select className="select" value={mediaType} onChange={(e) => setMediaType(e.target.value as "all" | MediaType)} title="Media type" aria-label="Media type">
              <option value="all">All media</option><option value="image">Images</option><option value="video">Videos</option>
            </select>
          </div>

          <div className="selectWrap">
            <select className="select" value={status} onChange={(e) => setStatus(e.target.value as "all" | IssueStatus)} title="Status" aria-label="Status">
              <option value="all">All status</option><option value="open">Open</option><option value="resolved">Resolved</option><option value="ignored">Ignored</option>
            </select>
          </div>

          <div className="selectWrap">
            <select className="select" value={severity} onChange={(e) => setSeverity(e.target.value as "all" | IssueSeverity)} title="Severity" aria-label="Severity">
              <option value="all">All severity</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option>
            </select>
          </div>

          <div className="selectWrap">
            <select className="select" value={source} onChange={(e) => setSource(e.target.value as "all" | IssueSource)} title="Source" aria-label="Source">
              <option value="all">All sources</option><option value="yolo">YOLO</option><option value="gpt_dino">GPT-DINO</option><option value="sam_fallback">SAM</option>
            </select>
          </div>

{/* Active filters (like Map/List) */}
{(klassInput.trim() || mediaType !== "all" || onlyMedia) && (
  <div className="activeFilters">
    {klassLike && <span className="pill">class: “{klassLike}”</span>}
    {mediaType !== "all" && <span className="pill">{mediaType}</span>}

    {/* NEW: media pill + clear */}
    {onlyMedia && <span className="pill">media #{onlyMedia}</span>}
    {onlyMedia && (
      <Button variant="ghost" size="sm" onClick={() => setOnlyMedia(null)}>
        Clear media
      </Button>
    )}

    <Button
      variant="ghost"
      size="sm"
      onClick={clearFilters}
      leftIcon={
        <svg
          viewBox="0 0 24 24"
          width="14"
          height="14"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      }
    >
      Clear
    </Button>
  </div>
)}


{isAuthority && (
  <button
    className={`togglePill ${assignedMine ? "active" : ""}`}
    onClick={() => setAssignedMine((v) => !v)}
    title="Show only issues assigned to me"
    aria-pressed={assignedMine}
  >
    Assigned to me
  </button>
)}
{(assignedMine && isAuthority) && <span className="pill">@{me}</span>}

        </div>

        <div className="right">
          <div className="stat"><span className="dot dotOpen" />open {statusCounts.open}</div>
          <div className="stat"><span className="dot dotResolved" />resolved {statusCounts.resolved}</div>
          <div className="stat"><span className="dot dotIgnored" />ignored {statusCounts.ignored}</div>
          <div className="kbdHint" aria-hidden><kbd>/</kbd> search</div>
        </div>
      </div>

      {/* Cards (mobile) */}
      <div className="cards">
        {state === "loading" && <div className="cardsSkeleton" />}
        {empty && (
          <div className="empty">
            <div className="emptyIcon" aria-hidden>🔎</div>
            <div><strong>No detections</strong><p className="muted">Try adjusting filters.</p></div>
          </div>
        )}

        {state === "ready" && pageRows.map((r) => (
          <article key={`issue-card-${r.id}`} className={`card sevAccent -${r.severity}`}>
            <header className="cardHead">
              <div>
                <div className="cardId">#{r.id} • media #{r.media_id}</div>
                <div className="cardDate">{new Date(r.created_at).toLocaleString()}</div>
              </div>
              <div className="badge">{r.class_name}</div>
            </header>

            <div className="cardBody">
              <div className="cardRow"><span className="metaLabel">Confidence</span><span className="metaValue">{r.confidence != null ? r.confidence.toFixed(2) : "—"}</span></div>
              <div className="cardRow">
                <span className="metaLabel">Severity</span>
                <span className="metaValue">
                  {isAdmin ? (
                    <span className="pillControl">
                      <span className={`sev-pill -${r.severity}`}>{r.severity}</span>
                      <select className="select selectInline" value={r.severity} onChange={(e) => changeSeverity(r, e.target.value as IssueSeverity)} title="Change severity" aria-label="Change severity">
                        <option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option>
                      </select>
                    </span>
                  ) : (
                    <span className={`sev-pill -${r.severity}`}>{r.severity}</span>
                  )}
                </span>
              </div>
              <div className="cardRow">
                <span className="metaLabel">Status</span>
                <span className="metaValue">
                  <span className="pillControl">
                    <span className={`status-pill -${r.status}`}>{r.status}</span>
                    <select
                      className="select selectInline"
                      value={r.status}
                      onChange={(e) => changeStatus(r, e.target.value as IssueStatus)}
                      title={r.assigned_to === me ? "Change status" : "Only assignee can resolve"}
                      aria-label="Change status"
                    >
                      <option value="open">Open</option>
                      <option value="resolved" disabled={r.assigned_to !== me}>Resolved</option>
                      {isAdmin && <option value="ignored">Ignored</option>}
                    </select>
                  </span>
                </span>
              </div>
              <div className="cardRow"><span className="metaLabel">Address</span><span className="metaValue wrap">{r.address ?? "—"}</span></div>

              {(r.annotated_image_url || r.annotated_video_url) && (
                <div className="cardRow">
                  <span className="metaLabel">Media</span>
                  <span className="metaValue">
                    {r.annotated_image_url && <img className="mediaThumb" src={absUrl(r.annotated_image_url) ?? undefined} alt="" loading="lazy" />}
                    {r.annotated_video_url && (
                      <div className="videoThumbWrapper">
                        {r.track_thumbnail_url ? (
                          <img 
                            className="mediaThumb" 
                            src={absUrl(r.track_thumbnail_url) ?? undefined} 
                            alt="Track thumbnail" 
                            loading="lazy" 
                          />
                        ) : (
                          <video
                            className="mediaThumb"
                            src={`${import.meta.env.VITE_API_BASE}${r.annotated_video_url}`}
                            poster={`${import.meta.env.VITE_API_BASE}${r.annotated_image_url || `/static/${r.media_id}.jpg`}`}
                            muted
                            preload="metadata"
                          />
                        )}
                        <div className="videoOverlay">
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="white" opacity="0.9">
                            <circle cx="12" cy="12" r="10" fill="rgba(0,0,0,0.6)" />
                            <path d="M10 8l6 4-6 4V8z" />
                          </svg>
                        </div>
                      </div>
                    )}
                  </span>
                </div>
              )}
            </div>

            <footer className="cardActions">
              {isAdmin && (
                <Button
                  className="actionBtn"
                  variant="secondary"
                  size="sm"
                  onClick={() => openAssign(r)}
                  disabled={!r.verified_at}
                  title={r.verified_at ? "Assign" : "Verify first to assign"}
                >
                  {r.assigned_to ? `Assigned @${r.assigned_to}` : "Assign"}
                </Button>
              )}
              {isAdmin && (
                <Button className="actionBtn" variant="secondary" size="sm" onClick={() => openVerify(r)}>
                  {r.verified_at ? "Verified" : "Verify"}
                </Button>
              )}
              <Button
              className="actionBtn"
              variant="secondary"
              size="sm"
              onClick={() => { location.href = `/list?media=${r.media_id}`; }}
            >
              See in list
            </Button>
            <Button
              className="actionBtn"
              variant="secondary"
              size="sm"
              onClick={() => { location.href = `/map?media=${r.media_id}`; }}
            >
              See in map
            </Button>

            </footer>
          </article>
        ))}
      </div>

      {/* Table (desktop) */}
      <div className="tableWrap">
        {state === "loading" && <div className="tableSkeleton" />}

        {state === "ready" && pageRows.length > 0 && (
          <>
            <table className="table issuesTable">
              <colgroup>
                <col className="c-id" />
                <col className="c-created" />
                <col className="c-class" />
                <col className="c-sev" />
                <col className="c-status" />
                <col className="c-source" />
                <col className="c-media" />
                <col className="c-address" />
                <col className="c-assignee" />
                <col className="c-actions" />
              </colgroup>

              <thead>
                <tr>
                  <th className="th col-id">ID</th>
                  <th className="th col-created">Created</th>
                  <th className="th col-class">Class</th>
                  <th className="th col-sev">Severity</th>
                  <th className="th col-status">Status</th>
                  <th className="th col-source">Source</th>
                  <th className="th col-media">Media</th>
                  <th className="th col-address left">Address</th>
                  <th className="th col-assignee">Assignee</th>
                  <th className="th col-actions">Actions</th>
                </tr>
              </thead>

              <tbody className="tbody">
                {pageRows.map((r) => (
                  <tr key={r.id} className="tr" data-sev={r.severity} data-status={r.status}>
                    <td className="td col-id">#{r.id}</td>
                    <td className="td col-created">{new Date(r.created_at).toLocaleString()}</td>
                    <td className="td col-class"><span className="chip">{r.class_name}</span></td>

                    <td className="td col-sev">
                      {isAdmin ? (
                        <div className="pillControl">
                          <span className={`sev-pill -${r.severity}`}>{r.severity}</span>
                          <select
                            className="select selectInline"
                            value={r.severity}
                            onChange={(e) => changeSeverity(r, e.target.value as IssueSeverity)}
                            title="Change severity"
                            aria-label="Change severity"
                          >
                            <option value="low">Low</option>
                            <option value="medium">Medium</option>
                            <option value="high">High</option>
                          </select>
                        </div>
                      ) : (
                        <span className={`sev-pill -${r.severity}`}>{r.severity}</span>
                      )}
                    </td>

                    <td className="td col-status">
                      <div className="pillControl">
                        <span className={`status-pill -${r.status}`}>{r.status}</span>
                        <select
                          className="select selectInline"
                          value={r.status}
                          onChange={(e) => changeStatus(r, e.target.value as IssueStatus)}
                          title={r.assigned_to === me ? "Change status" : "Only assignee can resolve"}
                          aria-label="Change status"
                        >
                          <option value="open">Open</option>
                          <option value="resolved" disabled={r.assigned_to !== me}>Resolved</option>
                          {isAdmin && <option value="ignored">Ignored</option>}
                        </select>
                      </div>
                    </td>

                    <td className="td col-source">{r.source}</td>

                    <td className="td col-media">
                      <div className="mediaCell">
                        {r.annotated_image_url ? (
                          <a
                            href={absUrl(r.annotated_image_url) ?? undefined}
                            target="_blank"
                            rel="noreferrer"
                            className="thumbLink"
                            title={`Open image #${r.media_id}`}
                          >
                            <img className="mediaThumb" src={absUrl(r.annotated_image_url) ?? undefined} alt="" loading="lazy" />
                          </a>
                        ) : r.annotated_video_url ? (
                          <a
                            href={`${import.meta.env.VITE_API_BASE}${r.annotated_video_url}`}
                            target="_blank"
                            rel="noreferrer"
                            className="thumbLink"
                            title={`Open video #${r.media_id}`}
                          >
                            <div className="videoThumbWrapper">
                              {r.track_thumbnail_url ? (
                                <img 
                                  className="mediaThumb" 
                                  src={absUrl(r.track_thumbnail_url) ?? undefined} 
                                  alt="Track thumbnail" 
                                  loading="lazy" 
                                />
                              ) : (
                                <video
                                  className="mediaThumb"
                                  src={`${import.meta.env.VITE_API_BASE}${r.annotated_video_url}`}
                                  poster={`${import.meta.env.VITE_API_BASE}${r.annotated_image_url || `/static/${r.media_id}.jpg`}`}
                                  muted
                                  preload="metadata"
                                />
                              )}
                              <div className="videoOverlay">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="white" opacity="0.9">
                                  <circle cx="12" cy="12" r="10" fill="rgba(0,0,0,0.6)" />
                                  <path d="M10 8l6 4-6 4V8z" />
                                </svg>
                              </div>
                            </div>
                          </a>
                        ) : <span className="muted">—</span>}
                        <span className="mediaId">#{r.media_id}</span>
                      </div>
                    </td>

                    <td className="td col-address left">{r.address ?? "—"}</td>

                    <td className="td col-assignee">
                      <div className="assigneeCell">
                        {r.assigned_to ? <span className="pill">@{r.assigned_to}</span> : <span className="muted">—</span>}
                      </div>
                    </td>

                    <td className="td col-actions">
                      <div className="row-actions">
                        {isAdmin && (
                          <Button
                            className="actionBtn"
                            variant="secondary"
                            size="sm"
                            onClick={() => openVerify(r)}
                          >
                            {r.verified_at ? "Verified" : "Verify"}
                          </Button>
                        )}
                        {isAdmin && (
                          <Button
                            className="actionBtn"
                            variant="secondary"
                            size="sm"
                            onClick={() => openAssign(r)}
                            disabled={!r.verified_at}
                            title={r.verified_at ? "Assign" : "Verify first to assign"}
                          >
                            {r.assigned_to ? "Change" : "Assign"}
                          </Button>
                        )}
                        <Button
                        className="actionBtn"
                        variant="secondary"
                        size="sm"
                        onClick={() => { location.href = `/list?media=${r.media_id}`; }}
                      >
                        See in list
                      </Button>
                      <Button
                        className="actionBtn"
                        variant="secondary"
                        size="sm"
                        onClick={() => { location.href = `/map?media=${r.media_id}`; }}
                      >
                        See in map
                      </Button>
                        {!isAdmin && <span className="muted">—</span>}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pager */}
            <div className="pager">
              <div className="pagerLeft">
                <span className="muted">
                  Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} of {total}
                </span>
              </div>
              <div className="pagerMid">
                <Button
                  variant="secondary"
                  size="sm"
                  iconOnly
                  aria-label="Previous page"
                  title="Previous page"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={!hasPrev}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 18l-6-6 6-6" />
                  </svg>
                </Button>
                <span className="pageNum">
                  {page} / {totalPages}
                </span>
                <Button
                  variant="secondary"
                  size="sm"
                  iconOnly
                  aria-label="Next page"
                  title="Next page"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={!hasNext}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </Button>
              </div>
              <div className="pagerRight">
                <select
                  className="select"
                  value={pageSize}
                  onChange={(e) => setPageSize(Number(e.target.value))}
                  title="Rows per page"
                  aria-label="Rows per page"
                >
                  {[10, 25, 50, 100, 200].map((n) => (
                    <option key={n} value={n}>{n} / page</option>
                  ))}
                </select>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="errorBanner" role="alert">
          <strong>Couldn’t load detections.</strong><span className="muted"> {error}</span>
        </div>
      )}

      {/* Dialogs */}
      {assignTarget && (
        <AssignDialog
          target={assignTarget}
          onClose={closeAssign}
          onAssign={async (u) => { await applyAssign(assignTarget, u); closeAssign(); }}
          onUnassign={async () => { await applyAssign(assignTarget, null); closeAssign(); }}
        />
      )}

      {verifyTarget && (
        <VerifyDialog
          target={verifyTarget}
          onClose={closeVerify}
          onVerify={async () => { await applyVerify(verifyTarget); closeVerify(); }}
        />
      )}
    </div>
  );
}

function AssignDialog({
  target,
  onClose,
  onAssign,
  onUnassign,
}: {
  target: Issue;
  onClose: () => void;
  onAssign: (username: string) => Promise<void>;
  onUnassign: () => Promise<void>;
}) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<UserLite[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const dq = useDebounced(q, 220);

  useEffect(() => {
    let cancel = false;
    setErr(null);
    if (!dq) { setResults([]); return; }
    searchUsers(dq).then((r) => { if (!cancel) setResults(r); }).catch(() => setErr("Search failed"));
    return () => { cancel = true; };
  }, [dq]);

  return (
    <>
      <div className="assignOverlay" onClick={onClose} />
      <div className="assignDialog" role="dialog" aria-modal="true" aria-labelledby="assignTitle">
        <div className="assignHead">
          <h3 id="assignTitle">Assign issue #{target.id}</h3>
          <button className="iconX" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="assignBody">
          <Input placeholder="Search username…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
          {err && <div className="assignErr">{err}</div>}
          <ul className="assignList">
            {results.map((u) => (
              <li key={u.username}>
                <button
                  className="assignRow"
                  disabled={busy}
                  onClick={async () => { setBusy(true); await onAssign(u.username); setBusy(false); }}
                  title={`Assign to @${u.username}`}
                >
                  <span className="uName">@{u.username}</span>
                  <span className="uRole">{u.role}</span>
                </button>
              </li>
            ))}
            {dq && results.length === 0 && !err && (<li className="assignEmpty">No matches</li>)}
          </ul>
        </div>
        <div className="assignFoot">
          <Button className="actionBtn" variant="secondary" size="sm" onClick={onUnassign} disabled={busy}>Unassign</Button>
          <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>
        </div>
      </div>
    </>
  );
}

function VerifyDialog({
  target,
  onClose,
  onVerify,
}: {
  target: Issue;
  onClose: () => void;
  onVerify: () => Promise<void>;
}) {
  return (
    <>
      <div className="assignOverlay" onClick={onClose} />
      <div className="confirmDialog" role="dialog" aria-modal="true" aria-labelledby="verifyTitle">
        <div className="assignHead">
          <h3 id="verifyTitle">Verify issue #{target.id}</h3>
          <button className="iconX" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="confirmBody">
          <p className="confirmText">Are you sure you want to mark this issue as verified?</p>
          <div className="confirmMeta">
            <span className="chip">{target.class_name}</span>
            {target.address && <span className="muted">{target.address}</span>}
          </div>
        </div>
        <div className="assignFoot">
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button className="actionBtn" size="sm" onClick={onVerify}>Verify</Button>
        </div>
      </div>
    </>
  );
}
