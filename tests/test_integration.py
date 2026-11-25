"""Integration tests: the real MCP server over stdio and the full LangGraph agent, driven by a
scripted model so the run is deterministic and free."""

import asyncio

from tests.fakes import ScriptedModel, call
from triage_agent.agent import Session
from triage_agent.tools import local_tools, mcp_tools


def run(coro):
    return asyncio.run(coro)


def test_mcp_server_tools_and_time_cutoff():
    async def go():
        async with mcp_tools(tracker_before=8000) as tools:
            by_name = {t.name: t for t in tools}
            assert set(by_name) == {"search_issues", "get_issue"}
            hits = await by_name["search_issues"].ainvoke({"query": "validate_call memory leak"})
            text = str(hits)
            assert "7687" in text and "8146" not in text  # 8146 was filed after the cutoff
            hidden = await by_name["get_issue"].ainvoke({"number": 8146})
            assert "not found" in str(hidden)

    run(go())


def test_two_subtasks_then_follow_up_with_memory():
    """Turn 1 is split into two subtasks and closes; turn 2 relies on turn 1 being in memory."""
    model = ScriptedModel(
        script=[
            '["how to drop None values", "is the validate_call leak known"]',  # planner, turn 1
            [call("search_docs", query="exclude none model_dump"), call("search_issues", query="validate_call leak")],
            [call("get_issue", number=7687)],
            [
                call("submit_triage", subtask=1, type="question", answer="use exclude_none=True"),
                call("submit_triage", subtask=2, type="bug", duplicate_of=7687, answer="duplicate of #7687"),
            ],
            '["when was issue 7687 opened"]',  # planner, turn 2
            [call("submit_triage", subtask=1, type="question", answer="2023-09-28")],
        ]
    )

    async def go():
        async with mcp_tools(tracker_before=8239) as issue_tools:
            session = Session(model, local_tools() + issue_tools)
            t1 = await session.send("Two things: drop None values? and is the validate_call leak known?")
            t2 = await session.send("When was the issue you linked opened?")
        return t1, t2

    t1, t2 = run(go())
    assert t1.plan == ["how to drop None values", "is the validate_call leak known"]
    assert t1.stop_reason == "closed" and len(t1.submissions) == 2
    assert [s.source for s in t1.steps[:3]] == ["local", "mcp", "mcp"]
    assert any(r["number"] == 7687 for r in t1.steps[1].result)  # real MCP search result
    assert t1.steps[2].result["created_at"] == "2023-09-28"
    assert t2.stop_reason == "closed" and t2.submissions[0]["answer"] == "2023-09-28"
    planner_input_turn2 = model.seen[4][-1].content
    assert "duplicate of #7687" in planner_input_turn2  # memory: turn 1 answer reaches turn 2
    assert t1.llm_calls == 4 and t1.input_tokens == 40


def test_invalid_arguments_are_reported_back_and_fixed():
    model = ScriptedModel(
        script=[
            '["question"]',
            [call("submit_triage", answer="missing the type field")],
            [call("submit_triage", type="question", answer="fixed")],
        ]
    )

    async def go():
        return await Session(model, local_tools()).send("How do I make a model frozen?")

    t = run(go())
    assert [s.valid for s in t.steps] == [False, True]
    assert "type" in t.steps[0].error
    assert t.stop_reason == "closed" and len(t.submissions) == 1
    assert t.submissions[0]["type"] == "question" and t.submissions[0]["answer"] == "fixed"


def test_text_reply_is_nudged_and_only_submit_is_offered():
    model = ScriptedModel(
        script=[
            '["question"]',
            "Use frozen=True.",  # plain text instead of a tool call
            [call("submit_triage", type="question", answer="Use frozen=True.")],
        ]
    )

    async def go():
        return await Session(model, local_tools()).send("How do I make a model frozen?")

    t = run(go())
    assert t.stop_reason == "closed"
    assert model.bound_tools[-1] == ["submit_triage"]  # after the nudge only submit is offered


def test_turn_without_submission_is_not_closed():
    model = ScriptedModel(script=['["question"]', "text", "still text", "text again"])

    async def go():
        return await Session(model, local_tools()).send("How do I make a model frozen?")

    t = run(go())
    assert t.stop_reason == "text_answer" and not t.submissions
