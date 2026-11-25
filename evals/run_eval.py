"""Run the full eval: every variant x every case x N repeats, then score, tabulate and plot.

    python -m evals.run_eval                  # all variants, all cases
    python -m evals.run_eval --limit 2        # smoke test: 2 cases per category, 1 repeat

Outputs go to results/<UTC timestamp>/; results/LATEST points at the newest full run.
Smoke runs (--limit) go to results/<UTC timestamp>_smoke/ and are not tracked by git.
A spend cap (--budget-usd) stops launching new cases once the paid models have used it up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.failure_report import report as failure_report
from evals.metrics import aggregate, rescore, score_case
from evals.plot import plot
from triage_agent.agent import Session
from triage_agent.config import MODELS, ROOT, load_dotenv
from triage_agent.graph import PLANNER_PROMPTS
from triage_agent.tools import local_tools, mcp_tools

CASES = ROOT / "evals" / "cases.jsonl"
RESULTS = ROOT / "results"
log = logging.getLogger("eval")


@dataclass(frozen=True)
class Variant:
    name: str
    model: str
    use_docs: bool
    description: str
    planner: str = "v1"


VARIANTS = [
    Variant("glm-4.7-flashx+rag", "glm-4.7-flashx", True, "Zhipu GLM-4.7-FlashX, docs search + MCP issue tools"),
    Variant("gemini-3.5-flash-lite+rag", "gemini-3.5-flash-lite", True, "Google Gemini 3.5 Flash-Lite, docs + MCP issue tools"),
    Variant("gemini-3.5-flash-lite-no-rag", "gemini-3.5-flash-lite", False, "Google Gemini 3.5 Flash-Lite, MCP issue tools only"),
]
# Targeted follow-up experiments, run only with --variants (not part of the default full run).
EXTRA_VARIANTS = [
    Variant(
        "glm-4.7-flashx+rag+planner-v2",
        "glm-4.7-flashx",
        True,
        "GLM-4.7-FlashX, docs + MCP issue tools, planner prompt v2",
        planner="v2",
    ),
]
INFRA_ATTEMPTS = 3  # a case whose turn hit a provider/infra error is rerun from scratch up to this often


class Budget:
    """Running spend across paid models; once over the cap no new case starts."""

    def __init__(self, cap_usd: float) -> None:
        self.cap, self.spent, self.skipped = cap_usd, 0.0, 0  # skipped: paid case runs not started

    def add(self, model: str, turns: list[dict]) -> None:
        self.spent += cost_of(model, turns)

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.cap


def cost_of(model: str, turns: list[dict]) -> float:
    spec = MODELS[model]
    return sum(t["input_tokens"] * spec.usd_per_m_input + t["output_tokens"] * spec.usd_per_m_output for t in turns) / 1e6


INFRA_MARKERS = ("RateLimit", "Connection", "Timeout", "InternalServer", "ServiceUnavailable", "429", "500", "502", "503")


def is_infra_error(trace: dict) -> bool:
    """Provider-side failures (rate limit, overload, connection) are not agent behaviour; retry those.

    A 400 means our request was wrong, so it is not retried and stays visible as a failure.
    """
    errors = [trace.get("error", "")] + [t["stop_reason"] for t in trace["turns"] if t["stop_reason"].startswith("error")]
    return any(marker in e for e in errors if e for marker in INFRA_MARKERS)


def load_cases(limit: int | None) -> list[dict]:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if limit:  # stratified: first `limit` cases of each category
        by_cat: dict[str, list[dict]] = {}
        for c in cases:
            by_cat.setdefault(c["category"], []).append(c)
        cases = [c for group in by_cat.values() for c in group[:limit]]
    return cases


async def run_case(variant: Variant, case: dict) -> dict:
    """One conversation: a fresh MCP server (with the case's tracker cutoff) and a fresh session."""
    async with mcp_tools(case.get("tracker_before")) as issue_tools:
        tools = local_tools(variant.use_docs) + issue_tools
        session = Session(MODELS[variant.model], tools, planner_prompt=PLANNER_PROMPTS[variant.planner])
        turns = []
        for turn in case["turns"]:
            trace = await session.send(turn["user"])
            turns.append(trace.to_dict())
            if trace.stop_reason.startswith("error"):
                break
    return {"tools_offered": [t.name for t in tools], "turns": turns}


class TraceLog:
    """traces.jsonl, appended and flushed one row at a time so a killed run loses at most the cases in flight."""

    def __init__(self, path: Path) -> None:
        self.path = path
        lines = path.read_text().splitlines() if path.exists() else []
        latest = {self.key(r): r for r in map(json.loads, lines)}  # a rerun replaces an earlier row
        self.rows = list(latest.values())
        # Rows that ended in a provider error are rerun on resume.
        self.done = {k for k, r in latest.items() if not r["trace"].get("infra_error")}

    @staticmethod
    def key(row: dict) -> tuple:
        return row["variant"], row["case_id"], row["repeat"]

    def append(self, row: dict) -> None:
        self.rows = [r for r in self.rows if self.key(r) != self.key(row)] + [row]
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run_variant(variant: Variant, cases: list[dict], repeats: int, concurrency: int, budget: Budget, tlog: TraceLog):
    sem = asyncio.Semaphore(concurrency)
    todo = [(c, r) for r in range(repeats) for c in cases if (variant.name, c["id"], r) not in tlog.done]
    total, done = len(todo), 0

    async def one(case: dict, rep: int) -> None:
        nonlocal done
        async with sem:
            if budget.exhausted and MODELS[variant.model].usd_per_m_input > 0:  # free models keep running
                budget.skipped += 1
                return None
            for attempt in range(1, INFRA_ATTEMPTS + 1):
                try:
                    trace = await run_case(variant, case)
                except Exception as exc:  # infrastructure failure (e.g. MCP server)
                    log.exception("case %s failed", case["id"])
                    trace = {"tools_offered": [], "turns": [], "error": repr(exc)}
                budget.add(variant.model, trace["turns"])
                trace["attempts"] = attempt
                trace["infra_error"] = is_infra_error(trace)
                if not trace["infra_error"]:
                    break
                log.warning("[%s] %s attempt %d hit a provider error, retrying", variant.name, case["id"], attempt)
                await asyncio.sleep(20 * attempt)
        done += 1
        s = score_case(case, trace)
        log.info(
            "[%s] %d/%d %s rep%d task=%s closed=%s goals=%s spent=$%.3f",
            variant.name,
            done,
            total,
            case["id"],
            rep,
            s["task_success"],
            s["flow_closed"],
            [m for t in s["turns"] for m in t["goals_met"]],
            budget.spent,
        )
        tlog.append({"variant": variant.name, "repeat": rep, "case_id": case["id"], "trace": trace, "score": s})

    await asyncio.gather(*(one(c, r) for c, r in todo))


def repeats_for(variant: Variant, args: argparse.Namespace) -> int:
    paid = MODELS[variant.model].usd_per_m_input > 0
    return args.paid_repeats if paid and args.paid_repeats else args.repeats


def summarize(variants: list[Variant], rows: list[dict], reps: dict[str, int]) -> dict:
    out = {}
    for v in variants:
        repeats = reps[v.name]
        vr = [r for r in rows if r["variant"] == v.name]
        if not vr:
            continue
        turns = [t for r in vr for t in r["trace"]["turns"]]
        m = aggregate([r["score"] for r in vr], [r["trace"] for r in vr], cost_of(v.model, turns))
        per_rep = [aggregate([r["score"] for r in vr if r["repeat"] == k], [], 0)["task_success_rate"] for k in range(repeats)]
        m["task_success_per_repeat"] = per_rep
        m["task_success_stdev"] = round(statistics.stdev([p or 0 for p in per_rep]), 4) if repeats > 1 else 0.0
        m["infra_error_runs"] = sum(r["trace"].get("infra_error", False) for r in vr)
        m["repeats"] = repeats
        m["model"], m["provider"], m["description"] = v.model, MODELS[v.model].provider.name, v.description
        out[v.name] = m
    return out


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def markdown_table(metrics: dict) -> str:
    main = (
        "| 变体 | 重复次数 | 任务执行准确率 | 流程闭环率 | 最终答案正确率 | 工具选择正确率 | 工具调用合法率 | 检索命中率 "
        "| 任务拆解正确率 | 误判重复率 | 每条用例平均耗时 (s) | 花费 (USD) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    main += "\n".join(
        f"| {name} | {m['repeats']} | {pct(m['task_success_rate'])}"
        + (f" ± {pct(m['task_success_stdev'])}" if m["repeats"] > 1 else "")
        + f" | {pct(m['flow_closure_rate'])} "
        f"| {pct(m['answer_accuracy'])} | {pct(m['tool_selection_accuracy'])} | {pct(m['valid_call_rate'])} "
        f"| {pct(m['retrieval_hit_rate'])} | {pct(m['decomposition_accuracy'])} | {pct(m['false_duplicate_rate'])} "
        f"| {m['avg_seconds_per_case']} | {m['cost_usd']:.4f} |"
        for name, m in metrics.items()
    )
    cat = (
        "\n\n| 变体 | 单轮任务 执行准确率 / 闭环率 | 一句多任务 执行准确率 / 闭环率 | 多轮追问 执行准确率 / 闭环率 |\n"
        "|---|---|---|---|\n"
    )
    cat += "\n".join(
        f"| {name} | "
        + " | ".join(
            f"{pct(m['by_category'][c]['task_success_rate'])} / {pct(m['by_category'][c]['flow_closure_rate'])}"
            for c in ("single", "multi_task", "follow_up")
        )
        + " |"
        for name, m in metrics.items()
    )
    goal = (
        "\n\n| 变体 | 文档问答 正确率 / 命中率 | 重复 issue 识别 正确率 / 命中率 | 新 issue 分类 正确率 | 追问事实 正确率 |\n"
        "|---|---|---|---|---|\n"
    )
    goal += "\n".join(
        f"| {name} | {pct(g['docs_answer']['accuracy'])} / {pct(g['docs_answer']['retrieval_hit'])} "
        f"| {pct(g['duplicate']['accuracy'])} / {pct(g['duplicate']['retrieval_hit'])} "
        f"| {pct(g['new_issue']['accuracy'])} | {pct(g['fact']['accuracy'])} |"
        for name, g in ((n, m["by_goal_kind"]) for n, m in metrics.items())
    )
    return main + cat + goal + "\n"


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--variants", nargs="*", default=[v.name for v in VARIANTS])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="cases per category (smoke test)")
    parser.add_argument("--cases", help="regex on case ids: a targeted experiment, written to <timestamp>_<--tag>/")
    parser.add_argument("--tag", default="exp", help="suffix of the output directory of a --cases experiment")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--budget-usd", type=float, default=0.4, help="stop starting new cases past this spend")
    parser.add_argument("--paid-repeats", type=int, help="fewer repeats for paid models (budget); default --repeats")
    parser.add_argument("--resume", type=Path, help="continue an interrupted run directory; finished cases are kept")
    args = parser.parse_args()
    if args.limit:
        args.repeats = 1

    if args.resume:
        out = args.resume.resolve()
        stamp = out.name.removesuffix("_smoke")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = "_smoke" if args.limit else (f"_{args.tag}" if args.cases else "")
        out = RESULTS / (stamp + suffix)
        out.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / f"run_{stamp}.log")],
    )
    for noisy in ("httpx", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    variants = [v for v in VARIANTS + EXTRA_VARIANTS if v.name in args.variants]
    cases = load_cases(args.limit)
    if args.cases:
        cases = [c for c in cases if re.fullmatch(args.cases, c["id"])]
    log.info("run %s: %d cases x %d repeats x %d variants", stamp, len(cases), args.repeats, len(variants))

    t0 = time.monotonic()
    tlog = TraceLog(out / "traces.jsonl")
    budget = Budget(args.budget_usd)
    for v in variants:  # spend already recorded by an interrupted run counts against the cap
        budget.spent += cost_of(v.model, [t for r in tlog.rows if r["variant"] == v.name for t in r["trace"]["turns"]])
    if tlog.rows:
        log.info("resuming: %d case runs already done, $%.3f spent", len(tlog.rows), budget.spent)
    reps = {v.name: repeats_for(v, args) for v in variants}
    for v in variants:
        await run_variant(v, cases, reps[v.name], args.concurrency, budget, tlog)
    # Only repeats inside each variant's planned count are scored (a resumed run may hold extra ones).
    rows = [r for r in tlog.rows if r["variant"] in reps and r["repeat"] < reps[r["variant"]]]
    rows = rescore(rows, cases)
    if budget.skipped:
        log.warning("budget cap $%.2f reached: %d case runs were not started", budget.cap, budget.skipped)

    metrics = summarize(variants, rows, reps)
    config = {
        "timestamp": stamp,
        "cases": len(cases),
        "goals": sum(len(t["goals"]) for c in cases for t in c["turns"]),
        "repeats": args.repeats,
        "repeats_per_variant": reps,
        "variants": [v.__dict__ | {"provider": MODELS[v.model].provider.name} for v in variants],
        "limit": args.limit,
        "budget_usd": args.budget_usd,
        "skipped_case_runs": budget.skipped,
        "resumed": bool(args.resume),
        "wall_seconds": round(time.monotonic() - t0, 1),
        "total_cost_usd": round(sum(m["cost_usd"] for m in metrics.values()), 4),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    table = markdown_table(metrics)
    (out / "results_table.md").write_text(table)
    plot(metrics, out / "results_chart.svg")
    (out / "failure_modes.md").write_text(failure_report(rows, out.name))
    if not args.limit and not args.cases and not budget.skipped:  # only complete full runs feed the README
        (RESULTS / "LATEST").write_text(stamp + "\n")
    log.info(
        "\n%s\ntotal cost $%.4f, wall %.0fs, outputs in %s",
        table,
        config["total_cost_usd"],
        config["wall_seconds"],
        out.relative_to(ROOT),
    )


if __name__ == "__main__":
    asyncio.run(main())
