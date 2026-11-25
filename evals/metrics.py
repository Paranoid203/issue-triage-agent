"""Deterministic scoring of conversation traces against eval cases (no LLM judge).

Units: a case is a conversation of one or more turns; each turn carries one or more goals.

* answer accuracy        - share of goals met
* task success           - share of cases whose every goal is met             (任务执行准确率)
* flow closure           - share of cases whose every turn closed, i.e. every
                           planned subtask got a submitted decision, no errors (流程闭环率)
* tool selection         - share of turns that called every tool their goals need
* valid tool calls       - share of tool calls whose arguments passed the tool schema
* retrieval hit          - share of goals with a gold doc/issue that a search surfaced in its top 5
* decomposition accuracy - share of first turns whose plan has as many subtasks as the turn has goals
* false duplicate rate   - share of new-issue goals where the agent claimed an unacceptable duplicate
"""

from __future__ import annotations

TOP_K = 5
REQUIRED_TOOLS = {"docs_answer": {"search_docs"}, "duplicate": {"search_issues"}, "new_issue": {"search_issues"}}
GOAL_KINDS = ("docs_answer", "duplicate", "new_issue", "fact")
CATEGORIES = ("single", "multi_task", "follow_up")


def _mentions(text: str, groups: list[list[str]]) -> bool:
    text = text.lower()
    return all(any(alt.lower() in text for alt in group) for group in groups)


def issue_number(value: object) -> int | None:
    """duplicate_of as the submit tool validated it: Pydantic accepts "7687" for an int field."""
    if isinstance(value, str) and value.strip().lstrip("#").isdigit():
        return int(value.strip().lstrip("#"))
    return value if isinstance(value, int) else None


def goal_met(goal: dict, submissions: list[dict]) -> bool:
    """A goal is met when any decision submitted in its turn satisfies it."""
    kind = goal["kind"]
    for s in submissions:
        dup, answer = issue_number(s.get("duplicate_of")), s.get("answer") or ""
        if kind == "docs_answer" and s.get("type") == "question" and _mentions(answer, goal["must_mention"]):
            return True
        if kind == "duplicate" and dup in goal["acceptable"]:
            return True
        if kind == "new_issue" and s.get("type") == goal["type"] and (dup is None or dup in goal["acceptable"]):
            return True
        if kind == "fact" and _mentions(answer, goal["must_mention"]):
            return True
    return False


def _search_results(turn: dict, tool: str) -> list[dict]:
    out: list[dict] = []
    for step in turn["steps"]:
        if step["tool"] == tool and step["valid"] and isinstance(step["result"], list):
            out += [r for r in step["result"][:TOP_K] if isinstance(r, dict)]
    return out


def retrieval_hit(goal: dict, turn: dict, offered: list[str]) -> bool | None:
    """Did a search in this turn surface the gold doc page / original issue? None if not applicable."""
    if goal["kind"] == "docs_answer" and "search_docs" in offered:
        return any(r.get("path") in goal["doc_paths"] for r in _search_results(turn, "search_docs"))
    if goal["kind"] == "duplicate":
        return any(r.get("number") in goal["acceptable"] for r in _search_results(turn, "search_issues"))
    return None


def _false_duplicate(goal: dict, goals: list[dict], submissions: list[dict]) -> bool | None:
    if goal["kind"] != "new_issue" or any(g["kind"] == "duplicate" for g in goals):
        return None
    return any(issue_number(s.get("duplicate_of")) not in (None, *goal["acceptable"]) for s in submissions)


def score_case(case: dict, trace: dict) -> dict:
    """`trace` = {"tools_offered": [...], "turns": [turn trace, ...]} for one conversation."""
    offered = trace["tools_offered"]
    turns_out = []
    for spec, turn in zip(case["turns"], trace["turns"], strict=False):
        goals = spec["goals"]
        called = {s["tool"] for s in turn["steps"]}
        required = set().union(*(REQUIRED_TOOLS.get(g["kind"], set()) for g in goals)) & set(offered)
        turns_out.append(
            {
                "goal_kinds": [g["kind"] for g in goals],
                "goals_met": [goal_met(g, turn["submissions"]) for g in goals],
                "retrieval_hits": [retrieval_hit(g, turn, offered) for g in goals],
                "false_duplicate": [_false_duplicate(g, goals, turn["submissions"]) for g in goals],
                "closed": turn["stop_reason"] == "closed",
                "stop_reason": turn["stop_reason"],
                "tool_selection_correct": required <= called,
                "plan_size": len(turn["plan"]),
                "n_goals": len(goals),
                "calls": len(turn["steps"]),
                "valid_calls": sum(s["valid"] for s in turn["steps"]),
            }
        )
    complete = len(trace["turns"]) == len(case["turns"])
    return {
        "case_id": case["id"],
        "category": case["category"],
        "turns": turns_out,
        "task_success": complete and all(all(t["goals_met"]) for t in turns_out),
        "flow_closed": complete and all(t["closed"] for t in turns_out),
    }


def rescore(rows: list[dict], cases: list[dict]) -> list[dict]:
    """Recompute every row's score from its trace, so scoring fixes apply to stored runs too."""
    by_id = {c["id"]: c for c in cases}
    return [r | {"score": score_case(by_id[r["case_id"]], r["trace"])} for r in rows]


def rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def aggregate(scores: list[dict], traces: list[dict], cost_usd: float) -> dict:
    turns = [t for s in scores for t in s["turns"]]
    goals = [
        {"kind": k, "met": m, "hit": h, "false_dup": f}
        for t in turns
        for k, m, h, f in zip(t["goal_kinds"], t["goals_met"], t["retrieval_hits"], t["false_duplicate"], strict=True)
    ]
    first_turns = [s["turns"][0] for s in scores if s["category"] != "follow_up" and s["turns"]]
    calls = sum(t["calls"] for t in turns)
    turn_traces = [t for tr in traces for t in tr["turns"]]
    n = len(scores)
    return {
        "n_cases": n,
        "n_goals": len(goals),
        "answer_accuracy": rate([g["met"] for g in goals]),
        "task_success_rate": rate([s["task_success"] for s in scores]),
        "flow_closure_rate": rate([s["flow_closed"] for s in scores]),
        "tool_selection_accuracy": rate([t["tool_selection_correct"] for t in turns]),
        "valid_call_rate": round(sum(t["valid_calls"] for t in turns) / calls, 4) if calls else None,
        "retrieval_hit_rate": rate([g["hit"] for g in goals]),
        "decomposition_accuracy": rate([t["plan_size"] == t["n_goals"] for t in first_turns]),
        "false_duplicate_rate": rate([g["false_dup"] for g in goals]),
        "avg_tool_calls_per_case": round(calls / n, 2) if n else None,
        "avg_seconds_per_case": round(sum(t["seconds"] for t in turn_traces) / n, 2) if n else None,
        "input_tokens": sum(t["input_tokens"] for t in turn_traces),
        "output_tokens": sum(t["output_tokens"] for t in turn_traces),
        "cost_usd": round(cost_usd, 4),
        "by_goal_kind": {
            k: {
                "n": sum(g["kind"] == k for g in goals),
                "accuracy": rate([g["met"] for g in goals if g["kind"] == k]),
                "retrieval_hit": rate([g["hit"] for g in goals if g["kind"] == k]),
            }
            for k in GOAL_KINDS
        },
        "by_category": {
            c: {
                "n": sum(s["category"] == c for s in scores),
                "task_success_rate": rate([s["task_success"] for s in scores if s["category"] == c]),
                "flow_closure_rate": rate([s["flow_closed"] for s in scores if s["category"] == c]),
            }
            for c in CATEGORIES
        },
    }
