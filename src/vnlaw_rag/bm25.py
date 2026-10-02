"""Okapi BM25 tự viết.

Corpus chỉ vài nghìn chunk nên viết thuần Python là đủ nhanh (<5ms/query),
khỏi kéo thêm rank_bm25. Tự viết còn để chỉnh được tokenizer (bigram, bỏ dấu).
"""

import math
from collections import Counter, defaultdict
from collections.abc import Callable

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Callable[[str], list[str]] = tokenize):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, docs: list[str]) -> "BM25":
        for i, doc in enumerate(docs):
            tf = Counter(self.tokenizer(doc))
            self.doc_len.append(sum(tf.values()))
            for term, c in tf.items():
                self.postings[term].append((i, c))
        n = len(docs)
        self.avgdl = sum(self.doc_len) / max(n, 1)
        for term, plist in self.postings.items():
            df = len(plist)
            # idf kiểu Lucene, luôn dương
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return self

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(self.tokenizer(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                out[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 50, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        sc = self.scores(query)
        items = [(i, s) for i, s in sc.items() if allowed is None or i in allowed]
        items.sort(key=lambda x: -x[1])
        return items[:k]
