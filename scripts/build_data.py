"""Build the public data snapshot used by the agent and the eval.

Sources (all public):
  * Pydantic documentation (MIT licensed) at a pinned commit -> data/doc_chunks.jsonl
  * Pydantic GitHub issues (public) -> data/issues.jsonl (the tracker the MCP server serves)
                                     -> data/query_issues.jsonl (issues used as eval queries,
                                        deliberately kept OUT of the tracker)

Requires the GitHub CLI (`gh`) to be authenticated. The snapshot is committed, so running
this script is only needed to refresh it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = "pydantic/pydantic"
DOCS_COMMIT = "2b555b79a603ab7eb27a23a9b1b9598897cdfef8"
BODY_LIMIT = 1500

# Issues that appear as eval queries. Their canonical originals must be in the tracker,
# the queries themselves must not.
QUERY_ISSUES = json.loads((ROOT / "scripts" / "query_issue_ids.json").read_text())


def gh_json(args: list[str]) -> object:
    out = subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def slim_issue(raw: dict) -> dict:
    body = re.sub(r"<!--.*?-->", "", raw.get("body") or "", flags=re.S).strip()
    return {
        "number": raw["number"],
        "title": raw["title"],
        "state": raw["state"],
        "labels": sorted(label["name"] for label in raw.get("labels", [])),
        "created_at": raw["created_at"][:10],
        "url": raw["html_url"],
        "body": body[:BODY_LIMIT],
    }


def fetch_issue(number: int) -> dict:
    return slim_issue(gh_json([f"repos/{REPO}/issues/{number}"]))


def search_issues(query: str, pages: int) -> list[dict]:
    items: list[dict] = []
    for page in range(1, pages + 1):
        res = gh_json(
            [
                "-X",
                "GET",
                "search/issues",
                "-f",
                f"q=repo:{REPO} is:issue {query}",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
                "-f",
                "sort=created",
            ]
        )
        items += [slim_issue(item) for item in res["items"]]
    return items


def build_issues() -> None:
    query_ids = {q["query"] for q in QUERY_ISSUES}
    canonical_ids = {q["original"] for q in QUERY_ISSUES if q.get("original")}
    tracker: dict[int, dict] = {}
    # Distractors: a realistic spread of recent and older labelled issues.
    for query, pages in [
        ('label:"bug V2"', 3),
        ('label:"feature request"', 2),
        ("label:question", 1),
        ("label:documentation", 1),
        ('label:"bug V1"', 1),
    ]:
        for issue in search_issues(query, pages):
            tracker[issue["number"]] = issue
    for number in canonical_ids:
        tracker[number] = fetch_issue(number)
    for number in query_ids:
        tracker.pop(number, None)
    rows = sorted(tracker.values(), key=lambda i: i["number"])
    write_jsonl(DATA / "issues.jsonl", rows)
    write_jsonl(DATA / "query_issues.jsonl", [fetch_issue(n) for n in sorted(query_ids)])
    print(f"tracker: {len(rows)} issues, queries: {len(query_ids)}")


def split_markdown(path: str, text: str, max_chars: int = 1800) -> list[dict]:
    """Split a markdown file on headings, then cap chunk size on paragraph boundaries."""
    chunks: list[dict] = []
    heading = path
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        while body:
            if len(body) <= max_chars:
                piece, body = body, ""
            else:
                cut = body.rfind("\n\n", 0, max_chars)
                cut = cut if cut > max_chars // 3 else max_chars
                piece, body = body[:cut], body[cut:].strip()
            if len(piece) > 80:
                chunks.append({"id": f"{path}#{len(chunks)}", "path": path, "heading": heading, "text": piece})

    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,4} ", line):
            flush()
            heading = line.lstrip("#").strip()
        buf.append(line)
    flush()
    return chunks


def build_docs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "-q", "--filter=blob:none", "--sparse", f"https://github.com/{REPO}.git", tmp], check=True
        )
        subprocess.run(["git", "-C", tmp, "checkout", "-q", DOCS_COMMIT], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "docs"], check=True)
        chunks: list[dict] = []
        for md in sorted(Path(tmp, "docs").rglob("*.md")):
            rel = md.relative_to(tmp).as_posix()
            chunks += split_markdown(rel, md.read_text(encoding="utf-8"))
    write_jsonl(DATA / "doc_chunks.jsonl", chunks)
    print(f"docs: {len(chunks)} chunks")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["docs", "issues"]
    if "docs" in targets:
        build_docs()
    if "issues" in targets:
        build_issues()
