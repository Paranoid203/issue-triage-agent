"""Unit tests: pure functions, no model API and no subprocess."""

import json
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.metrics import aggregate, goal_met, retrieval_hit, score_case
from evals.run_eval import is_infra_error, load_cases
from triage_agent.graph import parse_plan, submitted_subtasks, transcript
from triage_agent.retrieval import BM25, tokenize
from triage_agent.tools import TriageDecision, docs_corpus

# ---------------------------------------------------------------- retrieval


def test_tokenize_splits_identifiers():
    assert {"model_dump", "model", "dump"} <= set(tokenize("How do I use model_dump?"))


def test_bm25_ranks_relevant_doc_first():
    index = BM25(["cats purr", "dogs bark loudly", "the dog barks at cats"])
    assert index.search("dogs bark")[0][0] == 1


def test_docs_search_finds_known_page():
    paths = [r["path"] for r in docs_corpus().search("alias generator camel case", 5)]
    assert "docs/concepts/alias.md" in paths


# ---------------------------------------------------------------- graph helpers


def test_parse_plan_reads_json_and_falls_back():
    assert parse_plan('Sure: ["a", "b"]', "orig") == ["a", "b"]
    assert parse_plan("no json here", "orig") == ["orig"]
    assert parse_plan("[]", "orig") == ["orig"]
    assert len(parse_plan(json.dumps([str(i) for i in range(9)]), "orig")) == 4


def test_submitted_subtasks_ignores_failed_calls():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "submit_triage", "args": {"subtask": 1}, "id": "a", "type": "tool_call"},
            {"name": "submit_triage", "args": {"subtask": 2}, "id": "b", "type": "tool_call"},
        ],
    )
    msgs = [ai, ToolMessage("ok", tool_call_id="a"), ToolMessage("bad", tool_call_id="b", status="error")]
    assert submitted_subtasks(msgs) == {1}


def test_transcript_keeps_user_text_and_submitted_answers_only():
    ai = AIMessage(
        content="", tool_calls=[{"name": "submit_triage", "args": {"answer": "use X"}, "id": "a", "type": "tool_call"}]
    )
    text = transcript([HumanMessage("q1"), ai, ToolMessage("noise", tool_call_id="a"), HumanMessage("q2")])
    assert text == "USER: q1\n\nASSISTANT: use X\n\nUSER: q2"


def test_triage_decision_schema():
    TriageDecision(type="bug", answer="x")
    for bad in ({"answer": "x"}, {"type": "crash", "answer": "x"}):
        try:
            TriageDecision(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid decision {bad}")


# ---------------------------------------------------------------- scoring


DOCS_GOAL = {"kind": "docs_answer", "must_mention": [["model_dump"], ["model_validate"]], "doc_paths": ["a.md"]}
DUP_GOAL = {"kind": "duplicate", "acceptable": [7], "gold_issue": 7}
NEW_GOAL = {"kind": "new_issue", "type": "bug", "acceptable": []}


def test_goal_rules():
    assert goal_met(DOCS_GOAL, [{"type": "question", "answer": "model_dump and model_validate"}])
    assert not goal_met(DOCS_GOAL, [{"type": "question", "answer": "model_dump only"}])
    assert goal_met(DUP_GOAL, [{"type": "bug", "duplicate_of": 7}])
    assert not goal_met(NEW_GOAL, [{"type": "bug", "duplicate_of": 3}])
    assert goal_met({"kind": "fact", "must_mention": [["2020-04-23", "April 23, 2020"]]}, [{"answer": "April 23, 2020"}])


def _turn(subs, steps=(), stop="closed", plan=("p",)):
    return {
        "submissions": list(subs),
        "steps": list(steps),
        "stop_reason": stop,
        "plan": list(plan),
        "seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 1,
    }


def test_score_case_multi_task_and_aggregate():
    case = {"id": "m", "category": "multi_task", "turns": [{"user": "u", "goals": [DOCS_GOAL, DUP_GOAL]}]}
    steps = [
        {"tool": "search_docs", "valid": True, "result": [{"path": "a.md"}]},
        {"tool": "search_issues", "valid": True, "result": [{"number": 7}]},
        {"tool": "search_issues", "valid": False, "result": None},
    ]
    subs = [{"type": "question", "answer": "model_dump, model_validate"}, {"type": "bug", "duplicate_of": 7}]
    trace = {"tools_offered": ["search_docs", "search_issues", "submit_triage"], "turns": [_turn(subs, steps, plan=("a", "b"))]}
    s = score_case(case, trace)
    assert s["task_success"] and s["flow_closed"]
    assert retrieval_hit(DOCS_GOAL, trace["turns"][0], trace["tools_offered"]) is True
    m = aggregate([s], [trace], 0.0)
    assert m["task_success_rate"] == 1.0 and m["decomposition_accuracy"] == 1.0
    assert m["valid_call_rate"] == round(2 / 3, 4)


def test_flow_not_closed_when_a_turn_is_missing_or_open():
    case = {"id": "f", "category": "follow_up", "turns": [{"user": "a", "goals": [NEW_GOAL]}, {"user": "b", "goals": []}]}
    trace = {"tools_offered": [], "turns": [_turn([{"type": "bug", "duplicate_of": None}])]}
    s = score_case(case, trace)
    assert not s["flow_closed"] and not s["task_success"]
    trace["turns"].append(_turn([], stop="budget"))
    assert not score_case(case, trace)["flow_closed"]


def test_infra_errors_are_retried_but_bad_requests_are_not():
    assert is_infra_error({"turns": [_turn([], stop="error: OpenAIRateLimitError: Error code: 429")]})
    assert not is_infra_error({"turns": [_turn([], stop="error: OpenAIInvalidRequestError: Error code: 400")]})
    assert not is_infra_error({"turns": [_turn([])]})


# ---------------------------------------------------------------- data and docs


def test_cases_are_well_formed_and_reproducible():
    cases = load_cases(None)
    assert 30 <= len(cases) <= 50
    assert len({c["id"] for c in cases}) == len(cases)
    issues = {json.loads(line)["number"] for line in Path("data/issues.jsonl").read_text().splitlines()}
    for c in cases:
        for t in c["turns"]:
            for g in t["goals"]:
                if g["kind"] == "duplicate":
                    assert g["gold_issue"] in issues and g["gold_issue"] < c["tracker_before"]
    before = Path("evals/cases.jsonl").read_text()
    subprocess.run([sys.executable, "scripts/build_cases.py"], check=True, capture_output=True)
    assert Path("evals/cases.jsonl").read_text() == before


def test_readme_numbers_match_latest_run():
    if Path("results/LATEST").exists():
        assert subprocess.run([sys.executable, "scripts/sync_readme.py", "--check"]).returncode == 0


def test_failure_modes():
    from evals.failure_report import classify

    def row(category, turns, infra=False):
        return {"trace": {"infra_error": infra}, "score": {"category": category, "turns": turns}}

    def t(kinds, met, hits, closed=True, plan=1):
        return {
            "goal_kinds": kinds,
            "goals_met": met,
            "retrieval_hits": hits,
            "closed": closed,
            "plan_size": plan,
            "n_goals": len(kinds),
        }

    assert classify(row("single", [t(["duplicate"], [False], [True])]), 0, 0) == "judgment"
    assert classify(row("single", [t(["duplicate"], [False], [False])]), 0, 0) == "retrieval_miss"
    assert classify(row("single", [t(["new_issue"], [False], [None])]), 0, 0) == "classification"
    assert classify(row("single", [t(["new_issue"], [False], [None], closed=False)]), 0, 0) == "not_closed"
    assert classify(row("multi_task", [t(["docs_answer", "duplicate"], [True, False], [True, True])]), 0, 1) == "decomposition"
    follow = [t(["duplicate"], [False], [True]), t(["fact"], [False], [None])]
    assert classify(row("follow_up", follow), 1, 0) == "earlier_turn"
    assert classify(row("single", [t(["duplicate"], [False], [True])], infra=True), 0, 0) == "infra"


def test_duplicate_number_sent_as_string_is_scored_like_the_tool_validates_it():
    from triage_agent.tools import normalize_decision

    raw = {"type": "bug", "duplicate_of": "7687", "answer": "dup"}
    assert normalize_decision(raw)["duplicate_of"] == 7687
    assert goal_met(DUP_GOAL | {"acceptable": [7687]}, [raw])
    assert not goal_met(NEW_GOAL, [raw])
