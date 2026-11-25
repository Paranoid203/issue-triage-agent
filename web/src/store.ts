import { create } from "zustand";
import { postSse } from "./sse";
import type { Decision, ToolCall, Turn, TurnSummary } from "./types";

interface ChatState {
  threadId: string;
  turns: Turn[];
  busy: boolean;
  send: (text: string) => Promise<void>;
  newThread: () => void;
}

const newId = () => crypto.randomUUID().replace(/-/g, "");

export const useChat = create<ChatState>((set, get) => {
  // Every SSE event updates the last turn in place.
  const patch = (fn: (t: Turn) => Turn) =>
    set((s) => ({ turns: s.turns.map((t, i) => (i === s.turns.length - 1 ? fn(t) : t)) }));

  const onEvent = (event: string, data: unknown) => {
    const d = data as Record<string, unknown>;
    switch (event) {
      case "node":
        return patch((t) => ({ ...t, node: d.name as string }));
      case "plan":
        return patch((t) => ({ ...t, plan: d.subtasks as string[] }));
      case "tool_call":
        return patch((t) => ({ ...t, tools: [...t.tools, { ...(d as unknown as ToolCall), status: "running" }] }));
      case "tool_result":
        return patch((t) => ({
          ...t,
          tools: t.tools.map((c) =>
            c.id === d.id ? { ...c, status: d.ok ? "ok" : "error", result: d.result, error: d.error as string } : c,
          ),
        }));
      case "submission":
        return patch((t) => ({ ...t, submissions: [...t.submissions, d as unknown as Decision] }));
      case "text":
        return patch((t) => ({ ...t, text: d.text as string }));
      case "done":
        return patch((t) => ({ ...t, status: "done", node: undefined, summary: d as unknown as TurnSummary }));
      case "error":
        return patch((t) => ({ ...t, status: "error", node: undefined, error: d.message as string }));
    }
  };

  return {
    threadId: newId(),
    turns: [],
    busy: false,
    newThread: () => set({ threadId: newId(), turns: [] }),
    send: async (text) => {
      const { threadId, turns } = get();
      const turn: Turn = { id: turns.length + 1, user: text, status: "streaming", plan: [], tools: [], submissions: [] };
      set({ turns: [...turns, turn], busy: true });
      try {
        await postSse(`/api/threads/${threadId}/messages`, { text }, onEvent);
        patch((t) => (t.status === "streaming" ? { ...t, status: "error", error: "stream ended without a result" } : t));
      } catch (err) {
        patch((t) => ({ ...t, status: "error", error: String(err) }));
      } finally {
        set({ busy: false });
      }
    },
  };
});
