import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  listSessions,
  getSessionHistory,
  deleteSession,
  getRagChunk,
  updateSessionTitle,
    sendChatStream
} from '../api/chat';
import type React from 'react';
import type { ChatMessage, SessionSummary, RagChunk } from '../api/chat';
import Button from '../components/Button';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import '../styles/chat.css';

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<number | undefined>(() => {
    const s = localStorage.getItem('chatSessionId');
    return s ? Number(s) : undefined;
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);

  const logRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    const newHeight = Math.min(Math.max(textarea.scrollHeight, 44), 200);
    textarea.style.height = `${newHeight}px`;
  }, [text]);

  const [preview, setPreview] = useState<{ msgIdx: number; id: number; data?: RagChunk } | null>(null);
  const chunkCache = useRef(new Map<number, RagChunk>());

  const [confirmDel, setConfirmDel] = useState<SessionSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [editingTitle, setEditingTitle] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const streamingRef = useRef(false);
  useEffect(() => { listSessions().then(setSessions); }, []);
  useEffect(() => {
   if (streamingRef.current) return;
   if (selectedSession != null) {
     getSessionHistory(selectedSession).then(setMessages);
  } else {
     setMessages([]);
  }
 }, [selectedSession]);

  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    const behavior: ScrollBehavior = streamingRef.current ? 'auto' : (nearBottom ? 'smooth' : 'auto');
    requestAnimationFrame(() => el.scrollTo({ top: el.scrollHeight, behavior }));
}, [messages, busy]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target as HTMLElement | null)?.isContentEditable) return;
      const mac = navigator.platform.toLowerCase().includes('mac');
      if ((mac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  async function startNew() {
    setSelectedSession(undefined);
    setMessages([]);
    setPreview(null);
    localStorage.removeItem('chatSessionId');
    const updated = await listSessions();
    setSessions(updated);
  }

  useEffect(() => { setPreview(null); }, [selectedSession]);

  useEffect(() => {
    if (!preview) return;
    const m = messages[preview.msgIdx];
    if (!m || m.role !== 'assistant') { setPreview(null); return; }
    const ids = Array.from(m.content.matchAll(/\[#(\d+)\]/g)).map(g => g[1]);
    if (!ids.includes(String(preview.id))) setPreview(null);
  }, [messages, preview]);

  const openDelete = (s: SessionSummary, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setConfirmDel(s);
  };
  const closeDelete = () => setConfirmDel(null);
  const doDelete = async () => {
    if (!confirmDel) return;
    setDeleting(true);
    try {
      await deleteSession(confirmDel.id);
      const updated = await listSessions();
      setSessions(updated);
      if (selectedSession === confirmDel.id) await startNew();
    } finally {
      setDeleting(false);
      closeDelete();
    }
  };

  async function handleSend(e: FormEvent) {
  e.preventDefault();
  if (!text.trim() || busy) return;

  const content = text;
  setText('');
  setBusy(true);

  const baseLen = messages.length;
  const userMsg: ChatMessage = { role: 'user', content };
  const assistantIdx = baseLen + 1;

  setMessages(m => [...m, userMsg, { role: 'assistant', content: '', streaming: true }]);
  streamingRef.current = true;

  try {
    let sid = selectedSession;

    const result = await sendChatStream(
      { message: content, session_id: sid },
      (delta) => {
        setMessages(m => {
          const copy = m.slice();
          const cur = copy[assistantIdx];
          if (cur) copy[assistantIdx] = { ...cur, content: (cur.content ?? '') + delta };
          return copy;
        });
      },
      (sessionId) => {
        if (!sid) {
          sid = sessionId;
          setSelectedSession(sessionId);
          localStorage.setItem('chatSessionId', String(sessionId));
        }
      }
    );

    if (!sid) {
      sid = result.session_id;
      setSelectedSession(sid);
      localStorage.setItem('chatSessionId', String(sid));
    }

    listSessions().then(setSessions);
  } catch {
    setMessages(m => {
      const copy = m.slice();
      const cur = copy[assistantIdx];
      if (cur) copy[assistantIdx] = { ...cur, content: (cur.content ?? '') + '\n\n_(stream error)_' };
      return copy;
    });
  } finally {
    // Mark that specific assistant bubble as no longer streaming
    setMessages(m => {
      const copy = m.slice();
      if (copy[assistantIdx]) copy[assistantIdx] = { ...copy[assistantIdx], streaming: false };
      return copy;
    });
    streamingRef.current = false;
    setBusy(false);
  }
}

  function copyToClipboard(txt: string) {
    navigator.clipboard?.writeText(txt).catch(() => {});
  }

  async function prefetchChunk(id: number) {
    if (chunkCache.current.has(id)) return;
    try { chunkCache.current.set(id, await getRagChunk(id)); } catch { /* empty */ }
  }
  async function openPreview(msgIdx: number, id: number) {
    let data = chunkCache.current.get(id);
    if (!data) {
      try { data = await getRagChunk(id); chunkCache.current.set(id, data); } catch { /* empty */ }
    }
    setPreview({ msgIdx, id, data });
  }

  const markdownComponents: Components = useMemo(
    () => ({
      a({ href, children, ...props }) {
        // Check if it's a CSV download link
        if (href && href.includes('/download-csv/')) {
          return (
              <a
                  {...props}
                  href={href}
                  className="csv-download-btn"
                  download
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Download full CSV (opens in a new tab)"
                  title="Download full CSV"
              >
                <IconDownload/>
                <span>{children}</span>
              </a>
          );
        }
        // Regular link
        return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>;
      },
      ul(props) {
        return <ul className="md-ul" {...props} />;
      },
      ol(props) {
        return <ol className="md-ol" {...props} />;
      },
      p(props) {
        return <p className="md-p"  {...props} />; },
      h4(props) { return <h4 className="md-h4" {...props} />; },
      code({ inline, className, children, ...props }: { inline?: boolean; className?: string; children?: React.ReactNode }) {
        if (inline) return <code className="md-code-inline" {...props}>{children}</code>;
        return (
          <pre className="md-pre">
            <code className={className} {...props}>{children}</code>
          </pre>
        );
      },
    }),
    []
  );

  return (
    <div className="chat">
      {/* SIDEBAR */}
      <aside className="chat__sidebar card">
        <div className="chat__sidehead">
          <div className="title-stack">
            <div className="eyebrow">Assistant</div>
            <h2 className="grad-title">Conversations</h2>
          </div>
          <Button variant="secondary" size="sm" onClick={startNew} leftIcon={<IconPlus />}>
            New
          </Button>
        </div>

        <div className="conv-list" role="list">
          {sessions.length === 0 && (
            <div className="empty-hint">
              <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden>
                <path d="M4 4h16v12H7l-3 3V4z" fill="none" stroke="currentColor" strokeWidth="2" />
              </svg>
              No conversations yet
            </div>
          )}

          {sessions.map((s) => {
            const isActive = s.id === selectedSession;
            const last = new Date(s.last_message_at);
            return (
               <div
                key={s.id}
                className={`conv-item${isActive ? ' active' : ''}`}
                role="button"
                tabIndex={0}
                onClick={() => setSelectedSession(s.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') setSelectedSession(s.id);
                }}
                title={`Open #${s.id}`}
              >
                <div className="conv-item__main">
                  {editingTitle === s.id ? (
                    <input
                      type="text"
                      className="title-edit-input"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onBlur={async () => {
                        if (editTitle.trim() && editTitle !== s.title) {
                          await updateSessionTitle(s.id, editTitle);
                          const updated = await listSessions();
                          setSessions(updated);
                        }
                        setEditingTitle(null);
                      }}
                      onKeyDown={async (e) => {
                        if (e.key === 'Enter') {
                          if (editTitle.trim() && editTitle !== s.title) {
                            await updateSessionTitle(s.id, editTitle);
                            const updated = await listSessions();
                            setSessions(updated);
                          }
                          setEditingTitle(null);
                        } else if (e.key === 'Escape') {
                          setEditingTitle(null);
                        }
                        e.stopPropagation();
                      }}
                      onClick={(e) => e.stopPropagation()}
                      autoFocus
                    />
                  ) : (
                    <strong 
                      className="conv-title"
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        setEditingTitle(s.id);
                        setEditTitle(s.title || `Conversation #${s.id}`);
                      }}
                      title="Double-click to edit"
                    >
                      {s.title || `Conversation #${s.id}`}
                    </strong>
                  )}
                  <small className="muted">
                    {last.toLocaleString(undefined, { hour12: false })}
                  </small>
                </div>

                <Button
                  variant="ghost"
                  size="sm"
                  iconOnly
                  className="conv-item__del"
                  title="Delete conversation"
                  aria-label="Delete conversation"
                  onClick={(e) => {openDelete(s, e)}}
                >
                  <IconTrash />
                </Button>

                <span className="active-pill" aria-hidden />
              </div>
            );
          })}
        </div>
      </aside>

      {/* PANEL */}
      <section className="chat__panel card card--glow">
        {/* sticky subheader */}
        <header className="chatbar">
          <div className="chatbar__title">
            <span className="badge">Urban-AI</span>
            <h1>Urban Maintenance Assistant</h1>
          </div>
          <div className="chatbar__actions">
            <span className="chip chip--soft">
              Model <code className="mono">gpt-4.1</code>
            </span>
            <Button variant="ghost" size="sm" onClick={startNew} leftIcon={<IconSparkles />}>
              New chat
            </Button>
          </div>
        </header>

        {/* log */}
        <div className="chat-log" ref={logRef}>
          {messages.length === 0 && !busy && (
            <div className="intro">
              <div className="intro__icon" aria-hidden><IconChat /></div>
              <h3>How can I help?</h3>
              <p className="muted">
                Ask about an issue, paste a report, or reference a source with{' '}
                <code className="md-code-inline">[#12]</code>. I’ll cite and preview chunks inline.
              </p>
              <div className="kbd-hints">
                <span className="kbd" aria-hidden>Shift</span> + <span className="kbd">Enter</span> newline
                <span className="sep">•</span>
                <span className="kbd">⌘/Ctrl</span> + <span className="kbd">K</span> focus input
              </div>
            </div>
          )}

          {messages.map((m, i) => {
  const isUser = m.role === 'user';
  const isStreaming = !isUser && !!m.streaming;
  const isEmptyAssistant = !isUser && (!m.content || !m.content.trim());
  const citationIds = !isUser
    ? Array.from(new Set(Array.from(m.content.matchAll(/\[#(\d+)\]/g)).map((g) => g[1])))
    : [];

   if (isStreaming && isEmptyAssistant) return null;

  return (
    <div key={i} className={`msg-row ${isUser ? 'is-user' : 'is-assistant'}`}>
      <div className={`msg ${isUser ? 'msg--user' : 'msg--assistant'} ${isStreaming ? 'is-streaming' : ''}`}>
        {isUser ? (
          <div className="msg__text" style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
        ) : (
          <>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeSanitize]}
              components={markdownComponents}
            >
              {m.content}
            </ReactMarkdown>

            {citationIds.length > 0 && (
              <div className="citations">
                {citationIds.map((id) => (
                  <button
                    key={id}
                    className="rag-chip"
                    title={`Open chunk #${id}`}
                    onMouseEnter={() => prefetchChunk(Number(id))}
                    onClick={() => openPreview(i, Number(id))}
                  >
                    #{id}
                  </button>
                ))}
              </div>
            )}

            {preview && preview.msgIdx === i && preview.data && (
              <div className="preview">
                <div className="preview__head">
                  <strong>Chunk #{preview.id}</strong>
                  <Button
                    variant="ghost"
                    size="sm"
                    iconOnly
                    onClick={() => setPreview(null)}
                    aria-label="Close preview"
                  >
                    <IconX />
                  </Button>
                </div>
                {preview.data.image_url && (
                  <div className="preview__media">
                    <img src={preview.data.image_url} alt={`Media for chunk #${preview.id}`} />
                  </div>
                )}
                {preview.data.media_id && (
                  <div className="preview__meta">
                    Media ID: {preview.data.media_id}
                  </div>
                )}
                <div className="preview__text">
                  {preview.data.chunk}
                </div>
              </div>
            )}

            {/* show copy only when there’s content AND not streaming */}
            {!!m.content.trim() && !m.streaming && (
              <Button
                className="copy-btn"
                variant="ghost"
                size="sm"
                iconOnly
                title="Copy answer"
                aria-label="Copy answer"
                onClick={() => copyToClipboard(m.content)}
              >
                <IconCopy />
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  );
})}


          {busy && (() => {
  const last = messages[messages.length - 1];
  const showDots = !last || last.role !== 'assistant' || !last.content.trim();
  return showDots ? (
    <div className="msg-row is-assistant">
      <div className="msg msg--assistant thinking">
        <span className="dot" /><span className="dot" /><span className="dot" />
      </div>
    </div>
  ) : null;
})()}

        </div>

        {/* composer */}
        <form className="composer" onSubmit={handleSend}>
          <div className="composer-input-wrap">
            <span className="composer-icon"><IconPrompt /></span>
            <textarea
              ref={inputRef}
              className={`composer-textarea ${busy ? 'no-caret' : ''}`}
              placeholder="Ask about issues…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
                // Send on Enter (without Shift)
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!busy && text.trim()) {
                    handleSend(e as unknown as FormEvent<HTMLFormElement>);
                  }
                }
                if (e.key === 'Escape') {
                  setPreview(null);
                }
                e.stopPropagation();
              }}
              rows={1}
              style={{
                minHeight: '44px',
                maxHeight: '200px',
                resize: 'none',
                overflow: 'auto'
              }}
            />
          </div>
          <Button
            variant="primary"
            size="md"
            type="submit"
            loading={busy}
            disabled={busy || !text.trim()}
            leftIcon={<IconSend />}
          >
            Send
          </Button>
        </form>

        <div className="composer-hint muted">
          <span className="kbd">Shift</span> + <span className="kbd">Enter</span> to add a new line
        </div>
      </section>

{confirmDel && (
  <div className="modalRoot" role="presentation">
    <div className="modalOverlay" onClick={closeDelete} />
    <div className="confirmDialog" role="dialog" aria-modal="true" aria-labelledby="delTitle">
      <div className="confirmHead">
        <h3 id="delTitle">Delete conversation #{confirmDel.id}?</h3>
        <button className="iconX" onClick={closeDelete} aria-label="Close"><IconX /></button>
      </div>
      <div className="confirmBody">
        <p className="confirmText">This will permanently remove all messages in this conversation.</p>
      </div>
      <div className="confirmFoot">
        <Button variant="ghost" size="sm" onClick={closeDelete} autoFocus>Cancel</Button>
        <Button className="actionBtn" size="sm" variant="secondary" onClick={doDelete} disabled={deleting}>
          {deleting ? 'Deleting…' : 'Delete'}
        </Button>
      </div>
    </div>
  </div>
)}
    </div>
  );
}

/* ---------------- Icons ---------------- */
function IconPlus() { return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>); }
function IconTrash(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>); }
function IconSparkles(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 3l1.5 3L14 7l-3 1-1.5 3L8 8 5 7l3-1 1-3zM19 11l1 2 2 1-2 1-1 2-1-2-2-1 2-1 1-2z"/></svg>); }
function IconChat(){ return (<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V6a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>); }
function IconCopy(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>); }
function IconX(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>); }
function IconPrompt(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M7 8l-4 4 4 4"/><path d="M11 12h6"/></svg>); }
function IconSend(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>); }
function IconDownload(){ return (<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>); }
