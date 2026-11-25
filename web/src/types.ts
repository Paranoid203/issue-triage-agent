// Shapes of the SSE events and the metrics API (see src/triage_agent/server.py).

export type IssueType = "question" | "bug" | "feature_request" | "documentation";

export interface Decision {
  subtask: number;
  type: IssueType;
  duplicate_of: number | null;
  answer: string;
  references: string[];
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  source: "local" | "mcp";
  status: "running" | "ok" | "error";
  result?: unknown;
  error?: string;
}

export interface TurnSummary {
  stop_reason: string;
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  seconds: number;
}

export interface Turn {
  id: number;
  user: string;
  status: "streaming" | "done" | "error";
  node?: string;
  plan: string[];
  tools: ToolCall[];
  submissions: Decision[];
  text?: string;
  summary?: TurnSummary;
  error?: string;
}

export interface VariantMetrics {
  model: string;
  provider: string;
  description: string;
  n_cases: number;
  n_goals: number;
  task_success_rate: number | null;
  flow_closure_rate: number | null;
  answer_accuracy: number | null;
  tool_selection_accuracy: number | null;
  valid_call_rate: number | null;
  retrieval_hit_rate: number | null;
  decomposition_accuracy: number | null;
  false_duplicate_rate: number | null;
  avg_tool_calls_per_case: number | null;
  avg_seconds_per_case: number | null;
  cost_usd: number;
  repeats: number;
  by_goal_kind: Record<string, { n: number; accuracy: number | null; retrieval_hit: number | null }>;
}

export interface MetricsResponse {
  run: string;
  config: { cases: number; goals: number; total_cost_usd: number };
  metrics: Record<string, VariantMetrics>;
}
