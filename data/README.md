# Data sources

All data is public and was collected on 2026-11-24 with `scripts/build_data.py`.

| File | Content | Source and license |
|---|---|---|
| `doc_chunks.jsonl` | 793 chunks of the Pydantic documentation (`docs/**/*.md`), split on headings | [pydantic/pydantic](https://github.com/pydantic/pydantic) at commit `2b555b79`, MIT License, Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors |
| `issues.jsonl` | 799 public GitHub issues (number, title, state, labels, date, URL, first 1500 characters of the body) | public issues of pydantic/pydantic; each row links back to the original |
| `query_issues.jsonl` | 28 issues used as eval queries; kept out of `issues.jsonl` | same |

No private or company data is used.
