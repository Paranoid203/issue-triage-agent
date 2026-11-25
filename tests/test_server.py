"""HTTP service tests: FastAPI app with the scripted model, so no model API is called."""

import json

from fastapi.testclient import TestClient

from tests.fakes import ScriptedModel, call
from triage_agent.server import create_app
from triage_agent.tools import local_tools, mcp_tools


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def client_for(script, tools_factory=None) -> TestClient:
    async def local_only(stack):
        return local_tools()

    model = ScriptedModel(script=script)
    client = TestClient(create_app(model_factory=lambda: model, tools_factory=tools_factory or local_only))
    client.model = model
    return client


def test_health_search_and_metrics():
    with client_for([]) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        ready = c.get("/readyz")
        assert ready.status_code == 200 and "submit_triage" in ready.json()["tools"]
        hits = c.get("/api/docs/search", params={"q": "alias generator camel case"}).json()
        assert any(h["path"] == "docs/concepts/alias.md" for h in hits)
        m = c.get("/api/metrics").json()
        assert m["run"] and set(m["metrics"]) >= {"glm-4.7-flashx+rag"}


def test_sse_turn_streams_plan_tools_and_decisions_and_keeps_memory():
    script = [
        '["how to drop None values"]',
        [call("search_docs", query="exclude none model_dump")],
        [call("submit_triage", subtask=1, type="question", answer="use exclude_none=True")],
        '["what did you recommend"]',
        [call("submit_triage", subtask=1, type="question", answer="exclude_none=True, as before")],
    ]
    with client_for(script) as c:
        r = c.post("/api/threads/t1/messages", json={"text": "How do I drop None values?"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        names = [e for e, _ in events]
        assert names[:2] == ["node", "plan"] and names[-1] == "done"
        assert events[1][1]["subtasks"] == ["how to drop None values"]
        calls = [d for e, d in events if e == "tool_call"]
        assert [(d["name"], d["source"]) for d in calls] == [("search_docs", "local"), ("submit_triage", "local")]
        result = next(d for e, d in events if e == "tool_result" and d["name"] == "search_docs")
        assert result["ok"] and {"chunk_id", "path"} <= set(result["result"][0])
        assert [d["answer"] for e, d in events if e == "submission"] == ["use exclude_none=True"]
        assert events[-1][1]["stop_reason"] == "closed"

        r2 = c.post("/api/threads/t1/messages", json={"text": "What did you recommend?"})
        done = parse_sse(r2.text)[-1][1]
        assert done["stop_reason"] == "closed" and done["submissions"][0]["answer"].startswith("exclude_none")
        planner_input_turn2 = c.model.seen[3][-1].content  # memory: turn 1's answer reaches turn 2
        assert "use exclude_none=True" in planner_input_turn2


def test_invalid_arguments_show_up_as_failed_tool_results():
    script = [
        '["question"]',
        [call("submit_triage", answer="missing type")],
        [call("submit_triage", type="question", answer="fixed")],
    ]
    with client_for(script) as c:
        events = parse_sse(c.post("/api/threads/t2/messages", json={"text": "frozen models?"}).text)
    results = [d for e, d in events if e == "tool_result"]
    assert [d["ok"] for d in results] == [False, True] and "type" in results[0]["error"]
    assert [d["answer"] for e, d in events if e == "submission"] == ["fixed"]


def test_real_mcp_tool_results_are_streamed():
    async def with_mcp(stack):
        return local_tools() + await stack.enter_async_context(mcp_tools(tracker_before=8000))

    script = [
        '["is the validate_call leak known"]',
        [call("search_issues", query="validate_call memory leak")],
        [call("submit_triage", subtask=1, type="bug", duplicate_of=7687, answer="duplicate of #7687")],
    ]
    with client_for(script, with_mcp) as c:
        assert set(c.get("/readyz").json()["tools"]) >= {"search_issues", "get_issue"}
        events = parse_sse(c.post("/api/threads/t3/messages", json={"text": "validate_call leaks memory"}).text)
    result = next(d for e, d in events if e == "tool_result" and d["name"] == "search_issues")
    assert any(r["number"] == 7687 for r in result["result"])
    assert next(d for e, d in events if e == "tool_call")["source"] == "mcp"
    assert events[-1][1]["submissions"][0]["duplicate_of"] == 7687


def test_not_ready_without_a_model_key_but_still_alive():
    def no_key():
        raise RuntimeError("No API key for provider zhipu")

    with TestClient(create_app(model_factory=no_key)) as c:
        assert c.get("/healthz").status_code == 200
        r = c.get("/readyz")
        assert r.status_code == 503 and "No API key" in r.json()["reason"]
        assert c.post("/api/threads/x/messages", json={"text": "hi"}).status_code == 503


def test_empty_message_is_rejected():
    with client_for([]) as c:
        assert c.post("/api/threads/x/messages", json={"text": ""}).status_code == 422
