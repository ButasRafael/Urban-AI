import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  sendChat,
  listSessions,
  getSessionHistory,
  deleteSession,
  getRagChunk,
} from '../api/chat';
import type { ChatMessage, SessionSummary, RagChunk } from '../api/chat';
import Button from '../components/Button';
import Input from '../components/Input';
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
  const inputRef = useRef<HTMLInputElement | null>(null);

  // RAG preview state & cache
  const [preview, setPreview] = useState<{ msgIdx: number; id: number; data?: RagChunk } | null>(null);
  const chunkCache = useRef(new Map<number, RagChunk>());
  
  /* ---------- data ---------- */
  useEffect(() => { listSessions().then(setSessions); }, []);
  useEffect(() => {
    if (selectedSession != null) getSessionHistory(selectedSession).then(setMessages);
    else setMessages([]);
  }, [selectedSession]);

  /* ---------- UX niceties ---------- */
  // autoscroll when appending near bottom
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    requestAnimationFrame(() => el.scrollTo({ top: el.scrollHeight, behavior: nearBottom ? 'smooth' : 'auto' }));
  }, [messages, busy]);

  // quick focus (⌘/Ctrl + K)
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
    localStorage.removeItem('chatSessionId');
    const updated = await listSessions();
    setSessions(updated);
  }

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    if (!text.trim() || busy) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    setMessages((m) => [...m, userMsg]);
    setText('');
    setBusy(true);
    try {
      const resp = await sendChat({ message: userMsg.content, session_id: selectedSession });
      if (!selectedSession) {
        setSelectedSession(resp.session_id);
        localStorage.setItem('chatSessionId', String(resp.session_id));
      }
      setMessages((m) => [...m, { role: 'assistant', content: resp.answer }]);
      listSessions().then(setSessions);
    } finally {
      setBusy(false);
    }
  }

  function copyToClipboard(txt: string) {
    navigator.clipboard?.writeText(txt).catch(() => {});
  }

  async function prefetchChunk(id: number) {
    if (chunkCache.current.has(id)) return;
    try { chunkCache.current.set(id, await getRagChunk(id)); } catch {}
  }
  async function openPreview(msgIdx: number, id: number) {
    let data = chunkCache.current.get(id);
    if (!data) {
      try { data = await getRagChunk(id); chunkCache.current.set(id, data); } catch {}
    }
    setPreview({ msgIdx, id, data });
  }

  const markdownComponents: Components = useMemo(
    () => ({
      a({ node, ...props }) {
        return <a {...props} target="_blank" rel="noopener noreferrer" />;
      },
      ul(props) { return <ul className="md-ul" {...props} />; },
      ol(props) { return <ol className="md-ol" {...props} />; },
      p(props)  { return <p className="md-p"  {...props} />; },
      h4(props) { return <h4 className="md-h4" {...props} />; },
      code({ inline, className, children, ...props }: any) {
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
                  <strong>#{s.id}</strong>
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
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!confirm('Delete this conversation?')) return;
                    await deleteSession(s.id);
                    const updated = await listSessions();
                    setSessions(updated);
                    if (selectedSession === s.id) startNew();
                  }}
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
            const citationIds = !isUser
              ? Array.from(new Set(Array.from(m.content.matchAll(/\[#(\d+)\]/g)).map((g) => g[1])))
              : [];
            return (
              <div key={i} className={`msg-row ${isUser ? 'is-user' : 'is-assistant'}`}>
                <div className={`msg ${isUser ? 'msg--user' : 'msg--assistant'}`}>
                  {isUser ? (
                    <div className="msg__text">{m.content}</div>
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

                      {preview && preview.msgIdx === i && (
                        <div className="preview">
                          <div className="preview__head">
                            <strong>Chunk #{preview.id}</strong>
                            <Button variant="ghost" size="sm" iconOnly aria-label="Close preview" onClick={() => setPreview(null)}>
                              <IconX />
                            </Button>
                          </div>

                          {preview.data ? (
                            <>
                              {preview.data.image_url ? (
                                <div className="preview__media">
                                  <img src={preview.data.image_url} alt={`media for chunk #${preview.id}`} loading="lazy" />
                                </div>
                              ) : (
                                <div className="preview__meta">media_id: {preview.data.media_id}</div>
                              )}
                              <div className="preview__text">{preview.data.chunk}</div>
                            </>
                          ) : (
                            <div className="sk-line shimmer" style={{ height: 18, borderRadius: 6 }} />
                          )}
                        </div>
                      )}

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
                    </>
                  )}
                </div>
              </div>
            );
          })}

          {busy && (
            <div className="msg-row is-assistant">
              <div className="msg msg--assistant thinking">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            </div>
          )}
        </div>

        {/* composer */}
        <form className="composer" onSubmit={handleSend}>
          <Input
            ref={inputRef}
            placeholder="Ask about issues…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            stopGlobalKeys
            fullWidth
            prefixIcon={<IconPrompt />}
            onKeyDown={(e: any) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!busy && text.trim()) handleSend(e as any);
              }
              if (e.key === 'Escape') setPreview(null);
            }}
          />
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
