import { useEffect, useMemo, useState } from "react";
import type { MetricsResponse, VariantMetrics } from "../types";

// Higher is better for every metric charted here; the false-duplicate rate (lower is better) is table-only.
const CHART: { key: keyof VariantMetrics; label: string; help: string }[] = [
  { key: "task_success_rate", label: "Task execution accuracy", help: "Cases whose every goal in every turn was met" },
  { key: "flow_closure_rate", label: "Flow closure", help: "Cases where every planned subtask got a submitted decision" },
  { key: "answer_accuracy", label: "Answer accuracy", help: "Goals met" },
  { key: "decomposition_accuracy", label: "Decomposition", help: "First turns split into as many subtasks as they have goals" },
  { key: "retrieval_hit_rate", label: "Retrieval hit", help: "Goals whose gold doc page or original issue was in a top-5 search result" },
  { key: "valid_call_rate", label: "Valid tool calls", help: "Tool calls whose arguments passed the tool schema" },
];

// Variants are parallel categories: fixed hues, none of them red/green/amber (those mean good/bad/warning).
const HUES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const GOAL_LABEL: Record<string, string> = {
  docs_answer: "Docs question",
  duplicate: "Duplicate issue",
  new_issue: "New issue",
  fact: "Follow-up fact",
};

const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);

function Bars({ names, metrics }: { names: string[]; metrics: Record<string, VariantMetrics> }) {
  const row = 18;
  const gap = 4;
  const group = names.length * (row + gap) + 18;
  const labelW = 190;
  const width = 720;
  const plotW = width - labelW - 60;
  const height = CHART.length * group + 24;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Metrics per variant" className="bars">
      <title>Metrics per variant, 0 to 100 percent</title>
      {[0, 0.25, 0.5, 0.75, 1].map((x) => (
        <g key={x}>
          <line x1={labelW + x * plotW} x2={labelW + x * plotW} y1={0} y2={height - 20} className="grid" />
          <text x={labelW + x * plotW} y={height - 6} textAnchor="middle" className="axis">
            {x * 100}%
          </text>
        </g>
      ))}
      {CHART.map((m, gi) => (
        <g key={m.key} transform={`translate(0, ${gi * group})`}>
          <text x={0} y={14} className="glabel">
            <title>{m.help}</title>
            {m.label}
          </text>
          {names.map((n, vi) => {
            const v = metrics[n][m.key] as number | null;
            const y = 4 + vi * (row + gap);
            return (
              <g key={n}>
                <rect x={labelW} y={y} width={plotW} height={row} className="track" rx={2} />
                {v != null && <rect x={labelW} y={y} width={Math.max(1, v * plotW)} height={row} fill={HUES[vi % HUES.length]} rx={2} />}
                <text x={labelW + plotW + 6} y={y + 13} className="value">
                  {pct(v)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

export function Dashboard() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const names = useMemo(() => (data ? Object.keys(data.metrics) : []), [data]);
  if (error) return <p className="tool-error">✕ could not load metrics: {error}</p>;
  if (!data) return <p className="muted">Loading the latest eval run…</p>;

  const best = names.reduce((a, b) => ((data.metrics[b].task_success_rate ?? 0) > (data.metrics[a].task_success_rate ?? 0) ? b : a));
  const runDate = data.run.replace(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5 UTC");
  const kinds = Object.keys(data.metrics[best].by_goal_kind);

  return (
    <section className="dashboard">
      <p className="muted">Read-only view of eval run {data.run} · started {runDate}</p>
      <div className="hero">
        <span className="hero-number">{pct(data.metrics[best].task_success_rate)}</span>
        <span>
          best task execution accuracy · <strong>{best}</strong>
        </span>
      </div>
      <div className="stats">
        <div>
          <b>{data.config.cases}</b> cases · {data.config.goals} goals
        </div>
        <div>
          <b>{names.length}</b> variants
        </div>
        <div>
          <b>${data.config.total_cost_usd.toFixed(2)}</b> total model cost
        </div>
        <div>
          <b>{data.metrics[best].repeats}</b> run{data.metrics[best].repeats === 1 ? "" : "s"} per case
        </div>
      </div>

      <h2>Metrics by variant</h2>
      <div className="legend">
        {names.map((n, i) => (
          <span key={n}>
            <i style={{ background: HUES[i % HUES.length] }} /> {n}
          </span>
        ))}
      </div>
      <Bars names={names} metrics={data.metrics} />

      <h2>Table</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              {CHART.map((m) => (
                <th key={m.key} title={m.help}>
                  {m.label}
                </th>
              ))}
              <th title="New-issue goals wrongly called a duplicate (lower is better)">False duplicate</th>
              <th title="Average tool calls per case">Calls / case</th>
              <th title="Model cost of this variant">Cost</th>
            </tr>
          </thead>
          <tbody>
            {names.map((n, i) => {
              const v = data.metrics[n];
              return (
                <tr key={n}>
                  <td>
                    <i className="swatch" style={{ background: HUES[i % HUES.length] }} /> {n}
                  </td>
                  {CHART.map((m) => (
                    <td key={m.key} className="num">
                      {pct(v[m.key] as number | null)}
                    </td>
                  ))}
                  <td className="num">{pct(v.false_duplicate_rate)}</td>
                  <td className="num">{v.avg_tool_calls_per_case}</td>
                  <td className="num">${v.cost_usd.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h2>Accuracy by goal type</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              {kinds.map((k) => (
                <th key={k} title="accuracy / retrieval hit">
                  {GOAL_LABEL[k] ?? k} (n={data.metrics[best].by_goal_kind[k].n})
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((n, i) => (
              <tr key={n}>
                <td>
                  <i className="swatch" style={{ background: HUES[i % HUES.length] }} /> {n}
                </td>
                {kinds.map((k) => {
                  const g = data.metrics[n].by_goal_kind[k];
                  return (
                    <td key={k} className="num">
                      {pct(g.accuracy)}
                      {g.retrieval_hit != null && <span className="muted"> / hit {pct(g.retrieval_hit)}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
