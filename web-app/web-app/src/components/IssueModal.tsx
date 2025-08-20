import { useEffect, useMemo, useRef, useState } from 'react';
import type { Problem } from '../api/problems';
import Button from './Button';
import '../styles/issue-modal.css';

interface Props {
  problem: Problem | null;
  onClose(): void;
}

function toAbsolute(apiBase: string | undefined, path?: string | null): string | null {
  if (!path) return null;
  try {
    return apiBase ? new URL(path, apiBase).toString() : path;
  } catch {
    return path;
  }
}

export default function IssueModal({ problem, onClose }: Props) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  const [toast, setToast] = useState<string | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);

  useEffect(() => {
    if (!problem) return;
    const html = document.documentElement;
    const prevHtmlOverflow = html.style.overflow;
    const prevBodyOverflow = document.body.style.overflow;
    html.classList.add('has-modal');
    html.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
    return () => {
      html.style.overflow = prevHtmlOverflow;
      document.body.style.overflow = prevBodyOverflow || '';
      html.classList.remove('has-modal');
    };
  }, [problem]);

  const created = useMemo<Date | null>(() => {
    if (!problem?.created_at) return null;
    return new Date(problem.created_at);
  }, [problem]);

  const { hasCoords, lat, lng, coordsStr } = useMemo(() => {
    if (!problem || typeof problem.latitude !== 'number' || typeof problem.longitude !== 'number') {
      return { hasCoords: false, lat: null as number | null, lng: null as number | null, coordsStr: '' };
    }
    const lat = problem.latitude;
    const lng = problem.longitude;
    return { hasCoords: true, lat, lng, coordsStr: `${lat.toFixed(6)}, ${lng.toFixed(6)}` };
  }, [problem]);

  const gmapsUrl = useMemo<string | null>(() => {
    if (!problem) return null;
    if (hasCoords && lat != null && lng != null) return `https://www.google.com/maps?q=${lat},${lng}`;
    if (problem.address) return `https://www.google.com/maps?q=${encodeURIComponent(problem.address)}`;
    return null;
  }, [problem, hasCoords, lat, lng]);

  useEffect(() => {
    if (!problem) return;
    const keyHandler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onCloseRef.current(); return; }
      if (e.key.toLowerCase() === 'c' && (e.metaKey || e.ctrlKey)) return;
      if (!e.metaKey && !e.ctrlKey && !e.altKey) {
        const k = e.key.toLowerCase();
        if (k === 'a') copyAddress();
        if (k === 'g' && gmapsUrl) window.open(gmapsUrl, '_blank');
        if (k === 'k') copyCoords();
      }
    };
    window.addEventListener('keydown', keyHandler);
    closeBtnRef.current?.focus();
    return () => window.removeEventListener('keydown', keyHandler);
  }, [problem, gmapsUrl]);

  function showToast(msg: string) {
    setToast(msg);
    window.setTimeout(() => setToast(null), 1300);
  }

  async function copyAddress() {
    if (!problem?.address) return;
    try { await navigator.clipboard?.writeText(problem.address); showToast('Address copied'); } catch { /* empty */ }
  }

  async function copyCoords() {
    if (!coordsStr) return;
    try { await navigator.clipboard?.writeText(coordsStr); showToast('Coordinates copied'); } catch { /* empty */ }
  }

  async function share() {
    const url = gmapsUrl ?? (imagePreview || location.href);
    const text = problem
      ? `Issue #${problem.media_id}${problem.address ? ` • ${problem.address}` : ''}`
      : 'Issue';
    type NavWithShare = Navigator & { share?: (data: ShareData) => Promise<void> };
    const nav = navigator as NavWithShare;
    if (typeof nav.share === 'function') {
      try { await nav.share({ title: text, text, url }); } catch { /* empty */ }
    } else {
      try { await navigator.clipboard?.writeText(url!); showToast('Link copied'); } catch { /* empty */ }
    }
  }

  const stop: React.MouseEventHandler<HTMLDivElement> = (e) => e.stopPropagation();

  const apiBase = import.meta.env.VITE_API_BASE as string | undefined;
  const imagePreview = toAbsolute(apiBase, problem?.annotated_image_url);
  const videoPreview = toAbsolute(apiBase, problem?.annotated_video_url);

  if (!problem) return null;

  const classes: string[] = problem.predicted_classes ?? [];
  const hasClasses = classes.length > 0;
  const primary = (classes[0] as string | undefined) ?? 'Issue';
  const hue = Array.from(primary).reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) % 360, 0);

  return (
    <div className="modal-overlay" onClick={onClose} role="presentation">
      <div
        className="modal-card"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="issue-title"
        onClick={stop}
      >
        {toast && (
          <div className="modal-toast" role="status" aria-live="polite">
            {toast}
          </div>
        )}

        {/* Header */}
        <div className="modal-head">
          <div
            className="modal-badge"
            aria-hidden
            style={{
              background: `linear-gradient(135deg, hsl(${hue} 85% 55%), hsl(${(hue + 28) % 360} 80% 45%))`,
            }}
          >
            <IconPin />
          </div>

          <div className="modal-titles">
            <h2 id="issue-title">Issue #{problem.media_id}</h2>
            <p className="muted">
              {created ? created.toLocaleString() : '—'}
              {problem.user_username ? <> • <span className="username">@{problem.user_username}</span></> : null}
            </p>
          </div>

          <button
            ref={closeBtnRef}
            className="icon-close"
            aria-label="Close dialog"
            onClick={onClose}
            title="Close (Esc)"
          >
            <IconX />
          </button>
        </div>

        {/* Media preview */}
        {(imagePreview || videoPreview) && (
          <div className="media-box">
            {!imgLoaded && <div className="media-skeleton shimmer" aria-hidden />}
            {imagePreview && (
              <img
                src={imagePreview}
                alt=""
                onLoad={() => setImgLoaded(true)}
                className="media-el"
                loading="lazy"
              />
            )}
            {videoPreview && (
              <video
                className="media-el"
                controls
                playsInline
                onLoadedData={() => setImgLoaded(true)}
              >
                <source src={videoPreview} type="video/mp4" />
              </video>
            )}

            <div className="media-toolbar">
              {imagePreview ? (
                <a className="tool-btn" href={imagePreview} target="_blank" rel="noreferrer" title="Open media in new tab">
                  <IconExternal />
                </a>
              ) : videoPreview ? (
                <a className="tool-btn" href={videoPreview} target="_blank" rel="noreferrer" title="Open media in new tab">
                  <IconExternal />
                </a>
              ) : null}

              {imagePreview && (
                <a className="tool-btn" href={imagePreview} download title="Download image">
                  <IconDownload />
                </a>
              )}
            </div>
          </div>
        )}

        {/* Details */}
        <div className="modal-body">
          <div className="chips">
            <span className="chip-label">Classes</span>
            {hasClasses ? (
              classes.slice(0, 8).map((c, i) => (
                <span key={`${c}-${i}`} className="chip">{c}</span>
              ))
            ) : (
              <span className="chip chip--muted">No predictions</span>
            )}
          </div>

          <div className="meta-grid">
            <div className="meta-item">
              <span className="meta-label">Address</span>
              <span className="meta-value wrap">{problem.address ?? '—'}</span>
              <div className="meta-actions">
                <button className="mini-btn" onClick={copyAddress} disabled={!problem.address} title="Copy address (A)">
                  <IconCopy />
                </button>
                {gmapsUrl && (
                  <a className="mini-btn" href={gmapsUrl} target="_blank" rel="noreferrer" title="Open in Google Maps (G)">
                    <IconMap />
                  </a>
                )}
              </div>
            </div>

            <div className="meta-item">
              <span className="meta-label">Coordinates</span>
              <span className="meta-value">{coordsStr || '—'}</span>
              <div className="meta-actions">
                <button className="mini-btn" onClick={copyCoords} disabled={!coordsStr} title="Copy coordinates (K)">
                  <IconCopy />
                </button>
              </div>
            </div>

            <div className="meta-item">
              <span className="meta-label">ID</span>
              <span className="meta-value">#{problem.media_id}</span>
            </div>

            <div className="meta-item">
              <span className="meta-label">Created</span>
              <span className="meta-value">{created ? created.toLocaleString() : '—'}</span>
            </div>
          </div>

          <div className="kbd-hints">
            <span className="kbd" aria-hidden>A</span> copy address
            <span className="sep">•</span>
            <span className="kbd" aria-hidden>K</span> copy coords
            <span className="sep">•</span>
            <span className="kbd" aria-hidden>G</span> open maps
            <span className="sep">•</span>
            <span className="kbd" aria-hidden>Esc</span> close
          </div>
        </div>

        {/* Actions */}
        <div className="modal-actions">
          <Button
            variant="secondary"
            size="sm"
            onClick={copyAddress}
            disabled={!problem.address}
            leftIcon={<IconCopy />}
          >
            Copy address
          </Button>

          <Button
            variant="secondary"
            size="sm"
            onClick={copyCoords}
            disabled={!coordsStr}
            leftIcon={<IconTarget />}
          >
            Copy coords
          </Button>

          <div className="spacer" />

          <Button variant="secondary" size="sm" onClick={share} leftIcon={<IconShare />}>
            Share
          </Button>

          <a
            href={gmapsUrl ?? undefined}
            target="_blank"
            rel="noreferrer"
            className="btn btn-sm btn-primary"
            aria-disabled={!gmapsUrl}
            onClick={(e) => { if (!gmapsUrl) e.preventDefault(); }}
          >
            <span className="btn__inner">
              <span className="btn__icon"><IconMap /></span>
              <span className="btn__label">Open in Maps</span>
            </span>
          </a>
        </div>
      </div>
    </div>
  );
}

function IconX() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  );
}
function IconPin() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 22s8-7.16 8-13.2A8 8 0 1 0 4 8.8C4 14.84 12 22 12 22Z" />
      <circle cx="12" cy="9.5" r="2.5" fill="white" />
    </svg>
  );
}
function IconMap() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 18l-6 3V6l6-3 6 3 6-3v15l-6 3-6-3z" />
      <path d="M9 3v15" />
      <path d="M15 6v15" />
    </svg>
  );
}
function IconCopy() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}
function IconExternal() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M7 17L17 7" />
      <path d="M7 7h10v10" />
    </svg>
  );
}
function IconDownload() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 3v12" />
      <path d="M7 10l5 5 5-5" />
      <path d="M5 21h14" />
    </svg>
  );
}
function IconShare() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <path d="M8.59 13.51l6.83 3.98M15.41 6.51L8.59 10.5" />
    </svg>
  );
}
function IconTarget() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}
