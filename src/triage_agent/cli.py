"""Demo CLI.

triage-agent                      # scripted two-turn demo: a two-part request, then a follow-up
triage-agent "How do I ..."       # one message
triage-agent --chat               # interactive multi-turn chat
"""

from __future__ import annotations

import argparse
import asyncio
import json

from triage_agent.agent import Session, TurnTrace
from triage_agent.config import DEFAULT_MODEL, MODELS, load_dotenv
from triage_agent.tools import local_tools, mcp_tools

DEMO_TURNS = [
    "Two things. First, how do I drop every None value when I dump a model to a dict? "
    "Second, a bug report from my team: model_dump(exclude_defaults=True) drops a field that "
    "has a custom field_serializer even though its value is not the default. Is that known?",
    "For the issue you looked at, when was it opened and is it still open?",
]


def print_turn(i: int, trace: TurnTrace) -> None:
    print(f"\n=== turn {i}: {trace.user[:100]}{'...' if len(trace.user) > 100 else ''}")
    print("plan:", json.dumps(trace.plan, ensure_ascii=False))
    for n, step in enumerate(trace.steps, 1):
        status = "ok" if step.valid else f"ERROR {step.error}"
        args = json.dumps(step.arguments, ensure_ascii=False)
        print(f"[{n}] {step.tool}({args[:160]}{'...' if len(args) > 160 else ''})  [{step.source}, {status}]")
        if isinstance(step.result, list):
            for r in step.result[:3]:
                if isinstance(r, dict):
                    print("      ->", r.get("chunk_id") or f"#{r.get('number')} {str(r.get('title', ''))[:70]}")
    for sub in trace.submissions:
        print("submitted:", json.dumps(sub, ensure_ascii=False)[:400])
    print(
        f"{trace.llm_calls} LLM calls, {trace.input_tokens} in / {trace.output_tokens} out tokens, "
        f"{trace.seconds}s, stop={trace.stop_reason}"
    )


async def run(turns: list[str] | None, model: str) -> None:
    async with mcp_tools() as issue_tools:
        tools = local_tools() + issue_tools
        session = Session(MODELS[model], tools)
        print(f"model: {model}\ntools: {', '.join(t.name for t in tools)}")
        if turns is not None:
            for i, text in enumerate(turns, 1):
                print_turn(i, await session.send(text))
            return
        i = 0
        while (text := input("\nyou> ").strip()) not in {"", "exit", "quit"}:
            i += 1
            print_turn(i, await session.send(text))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("message", nargs="?")
    parser.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    args = parser.parse_args()
    turns = None if args.chat else ([args.message] if args.message else DEMO_TURNS)
    asyncio.run(run(turns, args.model))


if __name__ == "__main__":
    main()
