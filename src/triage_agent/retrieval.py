"""A small, dependency-free BM25 index used by both the docs search tool and the issue tracker."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    (
        "a an and are as at be but by can do does for from how i if in into is it its me my no not "
        "of on or so that the their then there these this to use using was we what when which why "
        "will with you your"
    ).split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; identifiers like `model_dump` also yield their parts."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        tokens.append(tok)
        if "_" in tok:
            tokens += [part for part in tok.split("_") if part and part not in _STOP]
    return tokens


class BM25:
    """Okapi BM25 over a list of documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        terms = [t for t in tokenize(query) if t in self.idf]
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class Corpus:
    """JSONL records plus a BM25 index over a text view of each record."""

    def __init__(self, path: Path, text_of: Callable[[dict], str]) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.text_of = text_of
        self.index = BM25([text_of(r) for r in self.rows])

    def restrict(self, keep: Callable[[dict], bool]) -> None:
        """Keep only rows matching `keep` and rebuild the index (IDF reflects the subset)."""
        self.rows = [r for r in self.rows if keep(r)]
        self.index = BM25([self.text_of(r) for r in self.rows])

    def search(self, query: str, k: int = 5) -> list[dict]:
        return [self.rows[i] | {"score": round(s, 2)} for i, s in self.index.search(query, k)]
