import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getProblems } from "../api/problems";
import type { Problem } from "../api/problems";
import Input from "../components/Input";
import Button from "../components/Button";
import IssueModal from "../components/IssueModal";
import "../styles/list-page.css";

type MediaType = "all" | "image" | "video";
type SortKey = "date" | "id";
type SortDir = "desc" | "asc";

export default function ListPage() {
  /* ---------- URL → initial filters ---------- */
  const initialType = (() => {
    const sp = new URLSearchParams(location.search);
    const t = sp.get("type");
    return t === "image" || t === "video" ? t : "all";
  })() as MediaType;

  const initialQuery = new URLSearchParams(location.search).get("q") ?? "";

  /* ---------- State ---------- */
  const [allItems, setAllItems] = useState<Problem[]>([]);
  const [type, setType] = useState<MediaType>(initialType);

  const handleCloseModal = useCallback(() => setModalProblem(null), []);

  // live input vs applied query (debounced)
  const [klassInput, setKlassInput] = useState(initialQuery);
  const [klass, setKlass] = useState(initialQuery);

  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [modalProblem, setModalProblem] = useState<Problem | null>(null);

  const searchRef = useRef<HTMLInputElement | null>(null);

  /* ---------- Fetch on media type only ---------- */
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getProblems({ media_type: type === "all" ? undefined : type })
      .then((items) => {
        if (cancelled) return;
        setAllItems(items);
      })
      .catch(() => {
        if (cancelled) return;
        setError("Failed to load items.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [type]);

  /* ---------- Debounce query ---------- */
  useEffect(() => {
    const t = setTimeout(() => setKlass(klassInput.trim()), 240);
    return () => clearTimeout(t);
  }, [klassInput]);

  /* ---------- Sync URL ---------- */
  useEffect(() => {
    const sp = new URLSearchParams(location.search);
    if (klass) sp.set("q", klass);
    else sp.delete("q");
    if (type !== "all") sp.set("type", type);
    else sp.delete("type");
    const newUrl = `${location.pathname}?${sp.toString()}`;
    history.replaceState(null, "", newUrl);
  }, [klass, type]);

  /* ---------- Client-side filter + sort ---------- */
  const filtered = useMemo(() => {
    const base = klass
      ? allItems.filter((p) =>
          (p.predicted_classes ?? []).some((c) =>
            c.toLowerCase().includes(klass.toLowerCase())
          )
        )
      : allItems.slice();

    base.sort((a, b) => {
      if (sortKey === "date") {
        const da = new Date(a.created_at).getTime();
        const db = new Date(b.created_at).getTime();
        return sortDir === "asc" ? da - db : db - da;
      } else {
        const ia = a.media_id;
        const ib = b.media_id;
        return sortDir === "asc" ? ia - ib : ib - ia;
      }
    });

    return base;
  }, [allItems, klass, sortKey, sortDir]);

  /* ---------- Pagination ---------- */
  useEffect(() => {
    setPage(1); // reset page when filters/sort change
  }, [klass, type, sortKey, sortDir, pageSize]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageItems = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [filtered, page, pageSize]);

  /* ---------- Stats ---------- */
  const stats = useMemo(() => {
    const images = filtered.filter((p) => p.media_type === "image").length;
    const videos = filtered.filter((p) => p.media_type === "video").length;
    return { total: filtered.length, images, videos };
  }, [filtered]);

  /* ---------- Shortcuts ---------- */

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName.toLowerCase();
      const inField = tag === "input" || tag === "textarea" || (el as any)?.isContentEditable;
      if (e.key === "/" && !e.shiftKey && !inField) {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (e.key === "ArrowRight" && !inField && page < totalPages) setPage((p) => p + 1);
      if (e.key === "ArrowLeft" && !inField && page > 1) setPage((p) => p - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [page, totalPages]);

  /* ---------- Actions ---------- */
  const clearFilters = () => {
    setKlassInput("");
    setKlass("");
    setType("all");
  };

  const exportCsv = () => {
    const safe = (v: unknown) =>
      String(v).replace(/"/g, '""').replace(/\r?\n/g, " ");

    const rows = [
      [
        "media_id",
        "created_at",
        "user_username",
        "media_type",
        "predicted_classes",
        "address",
        "description",
        "solution",
      ],
      ...filtered.map((p) => [
        p.media_id,
        new Date(p.created_at).toISOString(),
        p.user_username ?? "",
        p.media_type ?? "",
        (p.predicted_classes ?? []).join("|"),
        p.address ?? "",
        (p.descriptions ?? [])[0] ?? "",
        (p.solutions ?? [])[0] ?? "",
      ]),
    ];

    const csv = rows.map((r) => r.map((cell) => `"${safe(cell)}"`).join(","));
    const blob = new Blob([csv.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    a.download = `urban-ai-problems-${stamp}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  /* ---------- Render ---------- */
  return (
    <div className="page">
      {/* Toolbar */}
      <div className="toolbar">
        <div className="left">
          <Input
            ref={searchRef}
            placeholder="Filter by class…"
            value={klassInput}
            onChange={(e) => setKlassInput(e.target.value)}
            stopGlobalKeys
            prefixIcon={
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="11" cy="11" r="7" />
                <path d="M21 21l-4.3-4.3" />
              </svg>
            }
            clearable
          />

          <div className="selectWrap">
            <select
              className="select"
              value={type}
              onChange={(e) => setType(e.target.value as MediaType)}
              title="Media type"
              aria-label="Media type"
            >
              <option value="all">All types</option>
              <option value="image">Images</option>
              <option value="video">Videos</option>
            </select>
          </div>

          {(klass || type !== "all") && (
            <div className="activeFilters">
              {klass && <span className="pill">class: “{klass}”</span>}
              {type !== "all" && <span className="pill">{type}</span>}
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
        </div>

        <div className="right">
          <div className="stat">
            <span className="dot dotUploads" />
            {stats.total} items
          </div>
          <div className="stat">
            <span className="dot dotImages" />
            {stats.images} images
          </div>
          <div className="stat">
            <span className="dot dotVideos" />
            {stats.videos} videos
          </div>

          <div className="kbdHint" aria-hidden>
            <kbd>/</kbd> search · <kbd>←</kbd>/<kbd>→</kbd> pages
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={exportCsv}
            title="Export CSV"
            aria-label="Export CSV"
            leftIcon={
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M8 17l4 4 4-4" />
                <path d="M12 12v9" />
                <path d="M20 12a8 8 0 1 0-16 0" />
              </svg>
            }
          >
            Export
          </Button>
        </div>
      </div>

      {/* Card mode (mobile) */}
      <div className="cards">
        {loading && <div className="cardsSkeleton" />}
        {!loading && filtered.length === 0 && (
          <div className="empty">
            <div className="emptyIcon" aria-hidden>
              🔎
            </div>
            <div>
              <strong>No results</strong>
              <p className="muted">
                Try clearing filters or adjusting your search.
              </p>
            </div>
          </div>
        )}

        {!loading &&
          pageItems.map((it) => (
            <article key={`card-${it.media_id}`} className="card">
              <header className="cardHead">
                <div>
                  <div className="cardId">#{it.media_id}</div>
                  <div className="cardDate">
                    {new Date(it.created_at).toLocaleString()}
                  </div>
                </div>
                <div className="badge">
                  {it.media_type === "video" ? "🎞️ Video" : "📷 Image"}
                </div>
              </header>

              <div className="cardBody">
                <div className="cardRow">
                  <span className="metaLabel">User</span>
                  <span className="metaValue">
                    {it.user_username ?? "—"}
                  </span>
                </div>

                <div className="cardRow">
                  <span className="metaLabel">Classes</span>
                  <div className="chips">
                    {(it.predicted_classes ?? [])
                      .slice(0, 4)
                      .map((c, i) => (
                        <span key={`${c}-${i}`} className="chip">
                          {c}
                        </span>
                      ))}
                    {(it.predicted_classes?.length ?? 0) > 4 && (
                      <span className="chip chipMuted">
                        +{(it.predicted_classes!.length - 4).toString()}
                      </span>
                    )}
                  </div>
                </div>

                <div className="cardRow">
                  <span className="metaLabel">Description</span>
                  <span className="metaValue wrap">
                    {it.descriptions?.[0] ?? "—"}
                  </span>
                </div>

                <div className="cardRow">
                  <span className="metaLabel">Solution</span>
                  <span className="metaValue wrap">
                    {it.solutions?.[0] ?? "—"}
                  </span>
                </div>

                {it.annotated_image_url && (
                  <img
                    className="thumb"
                    src={`${import.meta.env.VITE_API_BASE}/static/${it.media_id}.jpg`}
                    alt=""
                    loading="lazy"
                  />
                )}
              </div>

              <footer className="cardActions">
                <Button
                  variant="secondary"
                  size="sm"
                  className="view-btn"
                  onClick={() => setModalProblem(it)}
                >
                  View details
                </Button>
              </footer>
            </article>
          ))}
      </div>

      {/* Table mode (desktop) */}
      <div className="tableWrap">
        {loading && <div className="tableSkeleton" />}

        {!loading && filtered.length > 0 && (
          <table className="table">
            <thead>
              <tr>
                <Th
                  label="ID"
                  active={sortKey === "id"}
                  dir={sortDir}
                  onClick={() => {
                    setSortKey("id");
                    setSortDir((d) =>
                      sortKey === "id" ? (d === "asc" ? "desc" : "asc") : "desc"
                    );
                  }}
                />
                <Th
                  label="Date"
                  active={sortKey === "date"}
                  dir={sortDir}
                  onClick={() => {
                    setSortKey("date");
                    setSortDir((d) =>
                      sortKey === "date" ? (d === "asc" ? "desc" : "asc") : "desc"
                    );
                  }}
                />
                <th className="th">User</th>
                <th className="th">Classes</th>
                <th className="th left">Description</th>
                <th className="th left">Solution</th>
                <th className="th">Thumb</th>
                <th className="th">Actions</th>
              </tr>
            </thead>
            <tbody className="tbody">
              {pageItems.map((it) => (
                <tr key={it.media_id} className="tr">
                  <td className="td">#{it.media_id}</td>
                  <td className="td">
                    {new Date(it.created_at).toLocaleString()}
                  </td>
                  <td className="td">{it.user_username ?? "—"}</td>
                  <td className="td">
                    <div className="chipsInline">
                      {(it.predicted_classes ?? [])
                        .slice(0, 4)
                        .map((c, i) => (
                          <span key={`${c}-${i}`} className="chip">
                            {c}
                          </span>
                        ))}
                      {(it.predicted_classes?.length ?? 0) > 4 && (
                        <span
                          className="chip chipMuted"
                          title={it.predicted_classes?.join(", ")}
                        >
                          +{(it.predicted_classes!.length - 4).toString()}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="td left">{it.descriptions?.[0] ?? "—"}</td>
                  <td className="td left">{it.solutions?.[0] ?? "—"}</td>
                  <td className="td">
                    {it.annotated_image_url && (
                      <img
                        src={`${import.meta.env.VITE_API_BASE}/static/${it.media_id}.jpg`}
                        className="thumbTable"
                        alt=""
                        loading="lazy"
                      />
                    )}
                    {it.annotated_video_url && "🎞️"}
                  </td>
                  <td className="td">
                    <Button
                      variant="secondary"
                      size="sm"
                      className="view-btn"
                      onClick={() => setModalProblem(it)}
                    >
                      View
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {!loading && filtered.length > 0 && (
          <div className="pager">
            <div className="pagerLeft">
              <span className="muted">
                Showing {(page - 1) * pageSize + 1}–
                {Math.min(page * pageSize, filtered.length)} of {filtered.length}
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
                disabled={page === 1}
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
                disabled={page === totalPages}
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
                {[10, 25, 50, 100].map((n) => (
                  <option key={n} value={n}>
                    {n} / page
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>
      {/* Error state */}
      {error && (
        <div className="errorBanner" role="alert">
          <strong>Couldn’t load data.</strong>
          <span className="muted"> Check your connection and try again.</span>
        </div>
      )}
      {modalProblem && (
        <IssueModal problem={modalProblem} onClose={handleCloseModal} />
      )}
    </div>
  );
}

/* ---------- Small clickable TH component ---------- */
function Th({
  label,
  active,
  dir,
  onClick,
}: {
  label: string;
  active?: boolean;
  dir?: SortDir;
  onClick?: () => void;
}) {
  return (
    <th
      className={`th thSort ${active ? "thActive" : ""}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onClick?.()}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
      title={`Sort by ${label}`}
    >
      <span className="thInner">
        {label}
        <span
          className={`sortIcon ${
            active ? (dir === "asc" ? "sortAsc" : "sortDesc") : ""
          }`}
          aria-hidden
        >
          ▲
        </span>
      </span>
    </th>
  );
}
