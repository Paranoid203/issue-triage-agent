Run `20261124T184625Z`: unmet goals by failure mode (counts over all repeats).

| 变体 | 目标数 | 未达成 | infra | not_closed | decomposition | earlier_turn | retrieval_miss | judgment | classification | other |
|---|---|---|---|---|---|---|---|---|---|---|
| glm-4.7-flashx+rag | 50 | 21 | 0 | 0 | 8 | 1 | 1 | 8 | 3 | 0 |
| gemini-3.5-flash-lite+rag | 50 | 15 | 0 | 0 | 0 | 0 | 2 | 9 | 4 | 0 |
| gemini-3.5-flash-lite-no-rag | 50 | 14 | 0 | 0 | 0 | 0 | 1 | 11 | 2 | 0 |

| 变体 | 目标类型 | 未达成 | 主要失败模式 |
|---|---|---|---|
| glm-4.7-flashx+rag | docs_answer | 2 | judgment 2 |
| glm-4.7-flashx+rag | duplicate | 15 | decomposition 8, judgment 6, retrieval_miss 1 |
| glm-4.7-flashx+rag | fact | 1 | earlier_turn 1 |
| glm-4.7-flashx+rag | new_issue | 3 | classification 3 |
| gemini-3.5-flash-lite+rag | docs_answer | 1 | judgment 1 |
| gemini-3.5-flash-lite+rag | duplicate | 10 | judgment 8, retrieval_miss 2 |
| gemini-3.5-flash-lite+rag | new_issue | 4 | classification 4 |
| gemini-3.5-flash-lite-no-rag | duplicate | 12 | judgment 11, retrieval_miss 1 |
| gemini-3.5-flash-lite-no-rag | new_issue | 2 | classification 2 |

Cases failed by every variant in every repeat (13): `dup-1151`, `dup-1388`, `dup-7655`, `dup-8152`, `dup-8232`, `dup-8581`, `follow-dup-1447`, `follow-dup-1476`, `follow-dup-1539`, `multi-docs-12+new-13888`, `multi-docs-13+dup-1281`, `multi-new-12580+dup-1470`, `new-11561`
