"""Grouped bar chart of the headline metrics, written as a hand-built SVG (no plotting dependency).

Re-render an existing run:  python -m evals.plot results/<timestamp>
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

METRICS = [
    ("task_success_rate", "Task success"),
    ("flow_closure_rate", "Flow closure"),
    ("answer_accuracy", "Answer accuracy"),
    ("tool_selection_accuracy", "Tool selection"),
    ("valid_call_rate", "Valid tool calls"),
    ("retrieval_hit_rate", "Retrieval hit@5"),
]
COLORS = ["#2563eb", "#d97706", "#6b7280", "#059669"]  # one fixed colour per variant, in VARIANTS order
W, H = 960, 400
LEFT, RIGHT, TOP, BOTTOM = 56, 16, 24, 92  # plot-area margins
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def plot(metrics: dict, path: Path) -> None:
    """Bars per metric, one colour per variant. Titles and captions live in the README."""
    names = list(metrics)
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    group_w = pw / len(METRICS)
    bar_w = group_w * 0.8 / len(names)

    def y(v: float) -> float:
        return TOP + ph * (1 - v / 100)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'role="img" aria-label="Eval results: four metrics for each agent variant, in percent">',
        "<title>Eval results by variant (percent)</title>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    for tick in range(0, 101, 20):
        ty = y(tick)
        out.append(f'<line x1="{LEFT}" y1="{ty:.1f}" x2="{W - RIGHT}" y2="{ty:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{LEFT - 8}" y="{ty + 4:.1f}" text-anchor="end" font-size="11" fill="#374151" {FONT}>{tick}%</text>')
    for j, (key, label) in enumerate(METRICS):
        gx = LEFT + j * group_w + group_w * 0.1
        for i, name in enumerate(names):
            value = metrics[name].get(key)
            x = gx + i * bar_w
            if value is None:
                out.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y(0) - 4:.1f}" text-anchor="middle" font-size="11" '
                    f'fill="#374151" {FONT}>n/a</text>'
                )
                continue
            v = value * 100
            out.append(
                f'<rect data-series="{escape(name)}" x="{x + 1:.1f}" y="{y(v):.1f}" width="{bar_w - 2:.1f}" '
                f'height="{y(0) - y(v):.1f}" fill="{COLORS[i % len(COLORS)]}"/>'
            )
            out.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y(v) - 4:.1f}" text-anchor="middle" font-size="11" '
                f'fill="#111827" {FONT}>{v:.0f}</text>'
            )
        out.append(
            f'<text x="{LEFT + (j + 0.5) * group_w:.1f}" y="{TOP + ph + 20}" text-anchor="middle" '
            f'font-size="12" fill="#111827" {FONT}>{label}</text>'
        )
    out.append(f'<line x1="{LEFT}" y1="{y(0):.1f}" x2="{W - RIGHT}" y2="{y(0):.1f}" stroke="#374151"/>')
    lx = LEFT
    for i, name in enumerate(names):
        color = COLORS[i % len(COLORS)]
        out.append(f'<rect data-series="{escape(name)}" x="{lx}" y="{H - 34}" width="12" height="12" fill="{color}"/>')
        out.append(f'<text x="{lx + 18}" y="{H - 24}" font-size="12" fill="#111827" {FONT}>{escape(name)}</text>')
        lx += 40 + 7 * len(name)
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    plot(json.loads((run_dir / "metrics.json").read_text()), run_dir / "results_chart.svg")
