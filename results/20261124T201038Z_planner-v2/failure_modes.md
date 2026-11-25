Run `20261124T201038Z_planner-v2`: unmet goals by failure mode (counts over all repeats).

| 变体 | 目标数 | 未达成 | infra | not_closed | decomposition | earlier_turn | retrieval_miss | judgment | classification | other |
|---|---|---|---|---|---|---|---|---|---|---|
| glm-4.7-flashx+rag+planner-v2 | 10 | 8 | 0 | 0 | 1 | 0 | 0 | 7 | 0 | 0 |

| 变体 | 目标类型 | 未达成 | 主要失败模式 |
|---|---|---|---|
| glm-4.7-flashx+rag+planner-v2 | duplicate | 8 | judgment 7, decomposition 1 |

Cases failed by every variant in every repeat (8): `dup-1151`, `dup-1388`, `dup-7655`, `dup-7747`, `dup-8232`, `dup-8581`, `dup-9348`, `dup-9766`
