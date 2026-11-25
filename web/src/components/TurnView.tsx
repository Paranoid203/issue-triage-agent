import type { Decision, ToolCall, Turn } from "../types";

const NODE_LABEL: Record<string, string> = {
  planner: "planning subtasks",
  executor: "model is choosing the next step",
  tools: "running tools",
  nudge: "reminding the model to submit",
};

const TYPE_LABEL: Record<Decision["type"], string> = {
  question: "💬 question",
  bug: "🐞 bug",
  feature_request: "✨ feature request",
  documentation: "📄 documentation",
};

function summarize(result: unknown): string[] {
  if (Array.isArray(result))
    return result.map((r) => {
      const o = r as Record<string, unknown>;
      return o.number ? `#${o.number} ${o.title ?? ""}` : String(o.chunk_id ?? o.path ?? JSON.stringify(o));
    });
  if (result && typeof result === "object") {
    const o = result as Record<string, unknown>;
    return [o.number ? `#${o.number} ${o.title ?? ""} · ${o.state ?? ""} · opened ${o.created_at ?? "?"}` : JSON.stringify(o)];
  }
  return result === undefined ? [] : [String(result)];
}

function ToolCard({ call }: { call: ToolCall }) {
  const icon = { running: "⏳", ok: "✓", error: "✕" }[call.status];
  const tone = { running: "warn", ok: "good", error: "bad" }[call.status];
  const args = Object.entries(call.args)
    .filter(([k]) => call.name !== "submit_triage" || k === "subtask")
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");
  const lines = call.status === "ok" && call.name !== "submit_triage" ? summarize(call.result) : [];
  return (
    <div className="tool-card">
      <div className="tool-head">
        <span className={`dot ${tone}`} aria-hidden />
        <code>
          {call.name}({args})
        </code>
        <span className="tag" title={call.source === "mcp" ? "served by the MCP issue-tracker server" : "in-process tool"}>
          {call.source === "mcp" ? "MCP" : "local"}
        </span>
        <span className={`status ${tone}`}>{icon}</span>
      </div>
      {lines.length > 0 && (
        <ul className="tool-result">
          {lines.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      )}
      {call.error && <p className="tool-error">✕ {call.error}</p>}
    </div>
  );
}

function DecisionCard({ d }: { d: Decision }) {
  return (
    <div className="decision">
      <div className="decision-head">
        <span className="pill">subtask {d.subtask}</span>
        <span className="pill">{TYPE_LABEL[d.type] ?? d.type}</span>
        {d.duplicate_of != null && (
          <a className="pill info" href={`https://github.com/pydantic/pydantic/issues/${d.duplicate_of}`} target="_blank" rel="noreferrer">
            ⧉ duplicate of #{d.duplicate_of}
          </a>
        )}
      </div>
      <p>{d.answer}</p>
      {d.references.length > 0 && <p className="muted small">refs: {d.references.join(", ")}</p>}
    </div>
  );
}

export function TurnView({ turn }: { turn: Turn }) {
  const submitted = new Set(turn.submissions.map((s) => s.subtask));
  return (
    <article className="turn">
      <div className="bubble user">{turn.user}</div>
      <div className="bubble agent">
        {turn.plan.length > 0 && (
          <ol className="plan">
            {turn.plan.map((p, i) => (
              <li key={i} className={submitted.has(i + 1) ? "done" : ""}>
                <span aria-hidden>{submitted.has(i + 1) ? "✓" : "○"}</span> {p}
              </li>
            ))}
          </ol>
        )}
        {turn.tools.map((c) => (
          <ToolCard key={c.id} call={c} />
        ))}
        {turn.submissions.map((d, i) => (
          <DecisionCard key={i} d={d} />
        ))}
        {turn.text && turn.submissions.length === 0 && <p>{turn.text}</p>}
        {turn.status === "streaming" && (
          <p className="progress">
            <span className="spinner" aria-hidden /> {NODE_LABEL[turn.node ?? "planner"] ?? turn.node}
          </p>
        )}
        {turn.summary && (
          <p className="muted small">
            {turn.summary.stop_reason === "closed" ? "✓ every subtask answered" : `▲ stopped: ${turn.summary.stop_reason}`} ·{" "}
            {turn.summary.llm_calls} model calls · {turn.summary.input_tokens.toLocaleString()} in / {turn.summary.output_tokens} out
            tokens · {turn.summary.seconds}s
          </p>
        )}
        {turn.error && <p className="tool-error">✕ {turn.error}</p>}
      </div>
    </article>
  );
}
