"""Pipeline truy hồi: lọc hiệu lực -> BM25 + vector -> RRF -> rerank -> phạt lệch loại xe -> mở rộng tham chiếu.

Mỗi bước bật/tắt được qua `SearchConfig` để chạy ablation (xem eval/run_eval.py).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .bm25 import BM25
from .fusion import rrf
from .parse import Chunk, build_backrefs
from .text import has_diacritics, tokenize
from .validity import DocMeta, effective_doc_ids
from .vehicle import detect_vehicles, vehicle_penalty


@dataclass
class SearchConfig:
    use_bm25: bool = True
    use_dense: bool = True
    use_rerank: bool = True
    use_validity: bool = True
    vehicle_aware: bool = True
    expand_refs: bool = True
    restore_diacritics: bool = True  # chỉ có tác dụng khi truyền restorer (dự án vi-diacritics-transformer)
    n_candidates: int = 50
    n_rerank: int = 20
    max_expanded: int = 3


@dataclass
class Hit:
    chunk: dict
    score: float
    expanded_from: str | None = None
    debug: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.chunk["id"]

    @property
    def citation(self) -> str:
        return self.chunk["citation"]


def load_chunks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    def __init__(self, chunks: list[dict], sources: dict[str, DocMeta], dense=None, reranker=None,
                 restorer: Callable[[str], str] | None = None):
        self.chunks = chunks
        self.sources = sources
        self.dense = dense
        self.reranker = reranker
        self.restorer = restorer
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25().fit(texts)
        # index thứ hai trên text đã bỏ dấu, dùng khi người dùng gõ không dấu
        self.bm25_fold = BM25(tokenizer=lambda t: tokenize(t, fold=True)).fit(texts)
        self.id2idx = {c["id"]: i for i, c in enumerate(chunks)}
        self.backrefs = build_backrefs_from_dicts(chunks)
        self._titles = [c["text"].split("\n", 1)[0] for c in chunks]

    def _allowed(self, as_of: date | None, use_validity: bool) -> set[int] | None:
        if not use_validity:
            return None
        ok_docs = effective_doc_ids(self.sources, as_of)
        return {i for i, c in enumerate(self.chunks) if c["doc_id"] in ok_docs}

    def search(self, query: str, k: int = 5, as_of: date | None = None,
               cfg: SearchConfig | None = None) -> list[Hit]:
        cfg = cfg or SearchConfig()
        debug: dict = {"query": query}
        q = query
        if cfg.restore_diacritics and self.restorer and not has_diacritics(query):
            q = self.restorer(query)
            debug["restored"] = q

        allowed = self._allowed(as_of, cfg.use_validity)
        rankings: list[list[int]] = []
        if cfg.use_bm25:
            bm = self.bm25 if has_diacritics(q) else self.bm25_fold
            rankings.append([i for i, _ in bm.search(q, cfg.n_candidates, allowed)])
        if cfg.use_dense and self.dense is not None:
            mask = None
            if allowed is not None:
                import numpy as np

                mask = np.zeros(len(self.chunks), dtype=bool)
                mask[list(allowed)] = True
            rankings.append([i for i, _ in self.dense.search(q, cfg.n_candidates, mask)])
        if not rankings:
            raise ValueError("phải bật ít nhất BM25 hoặc dense")

        fused = rrf(rankings) if len(rankings) > 1 else [(i, 1.0 / (60 + r)) for r, i in enumerate(rankings[0], 1)]
        cand = fused[: cfg.n_rerank] if cfg.use_rerank and self.reranker else fused
        scored = list(cand)
        if cfg.use_rerank and self.reranker is not None:
            rr = self.reranker.rerank(q, [self.chunks[i]["text"] for i, _ in cand])
            scored = list(zip([i for i, _ in cand], rr, strict=True))

        if cfg.vehicle_aware:
            qv = detect_vehicles(q)
            debug["vehicles"] = sorted(qv)
            if qv:
                # điểm rerank có thể âm (logit) -> phạt bằng cách trừ thay vì nhân
                spread = (max(s for _, s in scored) - min(s for _, s in scored)) if scored else 0
                scored = [(i, s - (1 - vehicle_penalty(qv, self._titles[i])) * (spread + 1e-6))
                          for i, s in scored]

        scored.sort(key=lambda x: -x[1])
        hits = [Hit(self.chunks[i], s, debug=debug) for i, s in scored[:k]]

        if cfg.expand_refs:
            seen = {h.id for h in hits}
            extra = []
            for h in hits:
                for rid in self.backrefs.get(h.id, []):
                    if rid not in seen and rid in self.id2idx and len(extra) < cfg.max_expanded:
                        idx = self.id2idx[rid]
                        if allowed is None or idx in allowed:
                            extra.append(Hit(self.chunks[idx], 0.0, expanded_from=h.id, debug=debug))
                            seen.add(rid)
            hits += extra
        return hits


def build_backrefs_from_dicts(chunks: list[dict]) -> dict[str, list[str]]:
    objs = [Chunk(**{k: c[k] for k in Chunk.__dataclass_fields__}) for c in chunks]
    return build_backrefs(objs)
