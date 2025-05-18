import {useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {sendChat} from "../api/chat";
import type { ChatMessage } from "../api/chat";
import Button from "../components/Button";
import Input from "../components/Input";
import { spacing, colors } from "../theme";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState("");
  const [sessionId, setSessionId] = useState<number | undefined>(() => {
    const s = localStorage.getItem("chatSessionId");
    return s ? Number(s) : undefined;
  });
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  // auto‑scroll
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;

    const msg: ChatMessage = { role: "user", content: text };
    setMessages((m) => [...m, msg]);
    setText("");
    setBusy(true);
    try {
      const resp = await sendChat({ message: msg.content, session_id: sessionId });
      setSessionId(resp.session_id);
      localStorage.setItem("chatSessionId", String(resp.session_id));
      setMessages((m) => [...m, { role: "assistant", content: resp.answer }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", display: "flex", flexDirection: "column", height: "calc(100vh - 120px)" }}>
      {/* chat log */}
      <div ref={listRef} style={{ flex: 1, overflowY: "auto", padding: spacing.m, border: `1px solid ${colors.muted}`, borderRadius: 8 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: spacing.m, textAlign: m.role === "user" ? "right" : "left" }}>
            <div style={{ display: "inline-block", padding: spacing.s, borderRadius: 8, background: m.role === "user" ? colors.primary : colors.secondary, color: "#fff" }}>
              {m.content}
            </div>
          </div>
        ))}
        {busy && <p style={{ opacity: 0.6 }}>…thinking</p>}
      </div>

      {/* composer */}
      <form onSubmit={handleSend} style={{ display: "flex", gap: spacing.s, marginTop: spacing.s }}>
        <Input placeholder="Ask about issues…" value={text} onChange={(e) => setText(e.target.value)} style={{ flex: 1 }} />
        <Button variant="primary" disabled={busy || !text.trim()} type="submit">
          Send
        </Button>
      </form>
    </div>
  );
}
