import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  sendChat,
  listSessions,
  getSessionHistory,
  deleteSession
} from "../api/chat";
import type { ChatMessage, SessionSummary } from "../api/chat";
import Button from "../components/Button";
import Input from "../components/Input";
import { spacing, colors } from "../theme";

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSession, setSelectedSession] = useState<number | undefined>(() => {
    const s = localStorage.getItem("chatSessionId");
    return s ? Number(s) : undefined;
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  // load sessions once
  useEffect(() => {
    listSessions().then(setSessions);
  }, []);

  // when user picks a session, load its history
  useEffect(() => {
    if (selectedSession != null) {
      getSessionHistory(selectedSession).then((msgs) => {
        setMessages(msgs);
      });
    }
  }, [selectedSession]);

  // auto-scroll on new messages
  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function startNew() {
    setSelectedSession(undefined);
    setMessages([]);
    localStorage.removeItem("chatSessionId");
    const updated = await listSessions();
    setSessions(updated);
  }

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    setText("");
    setBusy(true);

    try {
      const resp = await sendChat({
        message: userMsg.content,
        session_id: selectedSession,
      });

      // if new, set and persist session_id
      if (!selectedSession) {
        setSelectedSession(resp.session_id);
        localStorage.setItem("chatSessionId", String(resp.session_id));
      }

      setMessages((m) => [
        ...m,
        { role: "assistant", content: resp.answer },
      ]);

      // refresh sidebar
      listSessions().then(setSessions);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Sidebar */}
      <aside
        style={{
          width: 240,
          borderRight: `1px solid ${colors.muted}`,
          padding: spacing.m,
        }}
      >
        <Button
          variant="secondary"
          onClick={startNew}
          style={{ width: "100%", marginBottom: spacing.s }}
        >
          + New Conversation
        </Button>
        {sessions.map((s) => (
          <div
            key={s.id}
            onClick={() => setSelectedSession(s.id)}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: spacing.s,
              marginBottom: spacing.xs,
              borderRadius: 4,
              background:
                s.id === selectedSession ? colors.primary : "#f9f9f9",
              color: s.id === selectedSession ? "#fff" : "#000",
              cursor: "pointer",
            }}
          >
            <div>
              <strong>#{s.id}</strong>
              <br />
              <small>
                Last:{" "}
                {new Date(s.last_message_at).toLocaleString(undefined, {
                  hour12: false,
                })}
              </small>
            </div>

            {/* 🗑 delete button */}
            <button
              onClick={async (e) => {
                e.stopPropagation();
                if (!confirm("Delete this conversation?")) return;
                await deleteSession(s.id);
                const updated = await listSessions();
                setSessions(updated);
                if (selectedSession === s.id) {
                  startNew();
                }
              }}
              style={{
                background: "transparent",
                border: "none",
                color: s.id === selectedSession ? "#fff" : "#888",
                cursor: "pointer",
                fontSize: 14,
              }}
              title="Delete conversation"
            >
              🗑
            </button>
          </div>
        ))}
      </aside>

      {/* Chat panel */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          maxWidth: 760,
          margin: "0 auto",
          padding: spacing.m,
        }}
      >
        {/* chat log */}
        <div
          ref={listRef}
          style={{
            flex: 1,
            overflowY: "auto",
            padding: spacing.m,
            border: `1px solid ${colors.muted}`,
            borderRadius: 8,
          }}
        >
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                marginBottom: spacing.m,
                textAlign: m.role === "user" ? "right" : "left",
              }}
            >
              <div
                style={{
                  display: "inline-block",
                  padding: spacing.s,
                  borderRadius: 8,
                  background:
                    m.role === "user" ? colors.primary : colors.secondary,
                  color: "#fff",
                }}
              >
                {m.content}
              </div>
            </div>
          ))}
          {busy && <p style={{ opacity: 0.6 }}>…thinking</p>}
        </div>

        {/* composer */}
        <form
          onSubmit={handleSend}
          style={{
            display: "flex",
            gap: spacing.s,
            marginTop: spacing.s,
          }}
        >
          <Input
            placeholder="Ask about issues…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            style={{ flex: 1 }}
          />
          <Button
            variant="primary"
            disabled={busy || !text.trim()}
            type="submit"
          >
            Send
          </Button>
        </form>
      </div>
    </div>
  );
}
