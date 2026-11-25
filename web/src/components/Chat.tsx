import { useEffect, useRef, useState } from "react";
import { useChat } from "../store";
import { TurnView } from "./TurnView";

const EXAMPLES = [
  "How do I make a model immutable after it is created?",
  "Two things: how do I drop None values in model_dump? And is the validate_call memory leak a known issue?",
  "Bug report: using @validate_call on a method keeps growing memory with every call.",
];

export function Chat() {
  const { turns, busy, send, newThread, threadId } = useChat();
  const [text, setText] = useState("");
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const submit = (value: string) => {
    const v = value.trim();
    if (!v || busy) return;
    setText("");
    void send(v);
  };

  return (
    <section className="chat">
      <div className="thread-bar">
        <span className="muted" title="Conversation memory is kept per thread on the server">
          thread {threadId.slice(0, 8)} · {turns.length} turn{turns.length === 1 ? "" : "s"}
        </span>
        <button className="ghost" onClick={newThread} disabled={busy}>
          New conversation
        </button>
      </div>

      {turns.length === 0 && (
        <div className="examples">
          <p className="muted">Ask a Pydantic usage question or paste an issue report. Try:</p>
          {EXAMPLES.map((e) => (
            <button key={e} className="example" onClick={() => submit(e)}>
              {e}
            </button>
          ))}
        </div>
      )}

      {turns.map((t) => (
        <TurnView key={t.id} turn={t} />
      ))}
      <div ref={end} />

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
        }}
      >
        <textarea
          value={text}
          placeholder={turns.length ? "Ask a follow-up…" : "Ask a question or paste an issue report…"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(text);
            }
          }}
          rows={2}
        />
        <button type="submit" disabled={busy || !text.trim()}>
          {busy ? "Working…" : "Send"}
        </button>
      </form>
    </section>
  );
}
