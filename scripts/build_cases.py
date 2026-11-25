"""Assemble the eval cases (evals/cases.jsonl) from the labelled item pool (evals/items.jsonl).

Every item is one gradable goal (a docs answer, a duplicate to find, or a new issue to classify).
Cases are conversations built from items:

* single     - one message, one goal (10 docs questions, 10 duplicates, 10 new issues)
* multi_task - one message that bundles two goals; the agent has to split it into subtasks
* follow_up  - two turns; the second turn only makes sense with the first turn in memory

The script is deterministic, so the case file can always be regenerated and reviewed.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = {json.loads(line)["id"]: json.loads(line) for line in (ROOT / "evals/items.jsonl").read_text().splitlines()}
ISSUES = {json.loads(line)["number"]: json.loads(line) for line in (ROOT / "data/issues.jsonl").read_text().splitlines()}
C = "docs/concepts/"

SINGLE = (
    [f"docs-{i:02d}" for i in range(1, 11)]
    + ["dup-9766", "dup-9348", "dup-8581", "dup-8239", "dup-8232", "dup-8152", "dup-7747", "dup-7655", "dup-1388"]
    + ["dup-1151"]
    + ["new-13647", "new-12347", "new-13074", "new-12273", "new-12492", "new-13173", "new-11551", "new-11622"]
    + ["new-11606", "new-11561"]
)
MULTI_TASK = [("docs-12", "new-13888"), ("docs-13", "dup-1281"), ("docs-14", "new-12775"), ("new-12580", "dup-1470")]
FOLLOW_UP_ISSUE = ["dup-1539", "dup-1476", "dup-1447"]
FOLLOW_UP_DOCS = {
    "docs-11": (
        "How do I give that attribute a default value that is created fresh for every instance, like an empty list?",
        [["default_factory"]],
        [C + "models.md"],
    ),
    "docs-15": (
        "And what replaced .json() and .parse_raw()?",
        [["model_dump_json"], ["model_validate_json"]],
        ["docs/migration.md", C + "serialization.md", C + "models.md", C + "json.md"],
    ),
    "docs-16": (
        "Can I do the same for a plain type like list[int] that is not a model?",
        [["TypeAdapter"]],
        [C + "json_schema.md", C + "type_adapter.md"],
    ),
}


def goal(item: dict) -> dict:
    g, cat = item["gold"], item["category"]
    if cat == "docs_qa":
        return {"kind": "docs_answer", "must_mention": g["must_mention"], "doc_paths": g["doc_paths"]}
    if cat == "duplicate":
        return {"kind": "duplicate", "acceptable": g["acceptable_duplicates"], "gold_issue": g["duplicate_of"]}
    return {"kind": "new_issue", "type": g["type"], "acceptable": g["acceptable_duplicates"]}


def message(item: dict) -> str:
    return item["query"]


def date_forms(iso: str) -> list[str]:
    d = date.fromisoformat(iso)
    month, short = d.strftime("%B"), d.strftime("%b")
    return [iso, f"{month} {d.day}, {d.year}", f"{d.day} {month} {d.year}", f"{short} {d.day}, {d.year}"]


def tracker_cutoff(*items: dict) -> int | None:
    cutoffs = [i["tracker_before"] for i in items if i.get("tracker_before")]
    return min(cutoffs) if cutoffs else None


def build() -> list[dict]:
    cases = []
    for item_id in SINGLE:
        item = ITEMS[item_id]
        cases.append(
            {
                "id": item_id,
                "category": "single",
                "tracker_before": tracker_cutoff(item),
                "turns": [{"user": message(item), "goals": [goal(item)]}],
            }
        )
    for a, b in MULTI_TASK:
        ia, ib = ITEMS[a], ITEMS[b]
        text = (
            f"I have two separate things.\n\n(1) {message(ia)}\n\n(2) Separately, a teammate filed this report:\n\n{message(ib)}"
        )
        cases.append(
            {
                "id": f"multi-{a}+{b}",
                "category": "multi_task",
                "tracker_before": tracker_cutoff(ia, ib),
                "turns": [{"user": text, "goals": [goal(ia), goal(ib)]}],
            }
        )
    for item_id in FOLLOW_UP_ISSUE:
        item = ITEMS[item_id]
        original = ISSUES[item["gold"]["duplicate_of"]]
        cases.append(
            {
                "id": f"follow-{item_id}",
                "category": "follow_up",
                "tracker_before": tracker_cutoff(item),
                "turns": [
                    {"user": message(item), "goals": [goal(item)]},
                    {
                        "user": "When was the existing issue you linked opened? Give the date.",
                        "goals": [{"kind": "fact", "must_mention": [date_forms(original["created_at"])]}],
                    },
                ],
            }
        )
    for item_id, (follow, must, paths) in FOLLOW_UP_DOCS.items():
        item = ITEMS[item_id]
        cases.append(
            {
                "id": f"follow-{item_id}",
                "category": "follow_up",
                "tracker_before": None,
                "turns": [
                    {"user": message(item), "goals": [goal(item)]},
                    {"user": follow, "goals": [{"kind": "docs_answer", "must_mention": must, "doc_paths": paths}]},
                ],
            }
        )
    return cases


if __name__ == "__main__":
    cases = build()
    with (ROOT / "evals/cases.jsonl").open("w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    goals = sum(len(t["goals"]) for c in cases for t in c["turns"])
    print(f"{len(cases)} cases, {goals} goals")
