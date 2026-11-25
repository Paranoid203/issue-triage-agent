"""A scripted chat model for offline integration tests: returns queued replies, records what it saw."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def call(name: str, **args: Any) -> dict:
    call.counter = getattr(call, "counter", 0) + 1
    return {"name": name, "args": args, "id": f"call_{call.counter}", "type": "tool_call"}


class ScriptedModel(BaseChatModel):
    """Each invocation pops the next scripted reply (a string, or a list of tool calls)."""

    script: list[Any]
    seen: list[list[BaseMessage]] = []
    bound_tools: list[list[str]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # the same instance serves every binding
        names = [t.name for t in tools]
        model = self

        class Bound:
            async def ainvoke(self, messages, *a, **k):
                model.bound_tools.append(names)
                return await model.ainvoke(messages)

        return Bound()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.script.pop(0)
        msg = AIMessage(content=reply) if isinstance(reply, str) else AIMessage(content="", tool_calls=reply)
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        return ChatResult(generations=[ChatGeneration(message=msg)])
