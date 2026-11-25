"""Bucket every unmet goal of a run into a failure mode, straight from the traces.

    python -m evals.failure_report results/<timestamp>     # rescores the run, writes failure_modes.md

Failure modes, checked in this order (the first that applies wins):

  infra            the provider failed (rate limit, overload) even after retries
  not_closed       the agent never submitted a decision for this turn (step budget or text reply)
  decomposition    the planner produced a different number of subtasks than the turn has goals
  earlier_turn     follow-up goal whose first turn already failed (the error carried over)
  retrieval_miss   the gold doc page / original issue never appeared in the top 5 of any search
  judgment         the gold was retrieved, but the decision was wrong (missed or wrong duplicate,
                   answer without the needed API)
  classification   new issue: wrong type, or claimed a duplicate that is not one
  other            anything else (e.g. a follow-up fact answered wrongly)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evals.metrics import rescore

CASES = Path(__file__).resolve().parent / "cases.jsonl"

MODES = [
    "infra",
    "not_closed",
    "decomposition",
    "earlier_turn",
    "retrieval_miss",
    "judgment",
    "classification",
    "other",
]


def classify(row: dict, turn_idx: int, goal_idx: int) -> str:
    trace, score = row["trace"], row["score"]
    if trace.get("infra_error"):
        return "infra"
    t_score = score["turns"][turn_idx]
    kind = t_score["goal_kinds"][goal_idx]
    if not t_score["closed"]:
        return "not_closed"
    if turn_idx == 0 and row["score"]["category"] != "follow_up" and t_score["plan_size"] != t_score["n_goals"]:
        return "decomposition"
    if turn_idx > 0 and not all(score["turns"][0]["goals_met"]):
        return "earlier_turn"
    hit = t_score["retrieval_hits"][goal_idx]
    if hit is False:
        return "retrieval_miss"
    if hit is True:
        return "judgment"
    if kind == "new_issue":
        return "classification"
    return "other"


def failures(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        turns = row["score"]["turns"]
        if not turns:  # nothing ran at all
            out.append({"variant": row["variant"], "case_id": row["case_id"], "kind": "?", "mode": "infra"})
            continue
        for ti, t in enumerate(turns):
            for gi, met in enumerate(t["goals_met"]):
                if not met:
                    out.append(
                        {
                            "variant": row["variant"],
                            "case_id": row["case_id"],
                            "repeat": row["repeat"],
                            "turn": ti + 1,
                            "kind": t["goal_kinds"][gi],
                            "mode": classify(row, ti, gi),
                        }
                    )
    return out


def load_rows(run_dir: Path) -> list[dict]:
    """The scored rows of a run: latest row per (variant, case, repeat), within the planned repeats, rescored."""
    config = json.loads((run_dir / "config.json").read_text())
    reps = config.get("repeats_per_variant") or {v["name"]: config["repeats"] for v in config["variants"]}
    latest = {}
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        r = json.loads(line)
        latest[(r["variant"], r["case_id"], r["repeat"])] = r
    rows = [r for r in latest.values() if r["variant"] in reps and r["repeat"] < reps[r["variant"]]]
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    return rescore(rows, cases)


def report(rows: list[dict], run_name: str) -> str:
    fails = failures(rows)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    goals_per_variant = Counter()
    for r in rows:
        goals_per_variant[r["variant"]] += sum(len(t["goals_met"]) for t in r["score"]["turns"])

    lines = [f"Run `{run_name}`: unmet goals by failure mode (counts over all repeats).", ""]
    lines.append("| 变体 | 目标数 | 未达成 | " + " | ".join(MODES) + " |")
    lines.append("|---|---|---|" + "---|" * len(MODES))
    for v in variants:
        c = Counter(f["mode"] for f in fails if f["variant"] == v)
        lines.append(f"| {v} | {goals_per_variant[v]} | {sum(c.values())} | " + " | ".join(str(c[m]) for m in MODES) + " |")

    lines += ["", "| 变体 | 目标类型 | 未达成 | 主要失败模式 |", "|---|---|---|---|"]
    for v in variants:
        by_kind = defaultdict(Counter)
        for f in fails:
            if f["variant"] == v:
                by_kind[f["kind"]][f["mode"]] += 1
        for kind, c in sorted(by_kind.items()):
            top = ", ".join(f"{m} {n}" for m, n in c.most_common(3))
            lines.append(f"| {v} | {kind} | {sum(c.values())} | {top} |")

    runs_per_case = Counter((r["variant"], r["case_id"]) for r in rows)
    failed_runs = Counter((r["variant"], r["case_id"]) for r in rows if not r["score"]["task_success"])
    always = defaultdict(list)
    for (v, cid), n in failed_runs.items():
        if n == runs_per_case[(v, cid)]:
            always[cid].append(v)
    hard = sorted((cid for cid, vs in always.items() if len(vs) == len(variants)))
    lines += ["", f"Cases failed by every variant in every repeat ({len(hard)}): " + ", ".join(f"`{c}`" for c in hard)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    text = report(load_rows(run_dir), run_dir.name)
    (run_dir / "failure_modes.md").write_text(text)
    print(text)
