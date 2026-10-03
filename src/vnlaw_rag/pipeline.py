"""Gói retriever + answer thành một object dùng chung cho API, CLI và dự án viet-copilot.

    from vnlaw_rag.pipeline import LawRAG
    rag = LawRAG.from_dir("data")          # tự dò dense/reranker nếu có
    rag.search("xe máy vượt đèn đỏ")       # -> list[Hit]
    rag.answer("xe máy vượt đèn đỏ")       # -> Answer (có trích dẫn)
    rag.lookup_sign("P.131a")              # -> điều khoản phạt liên quan tới biển báo
"""

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .answer import Answer, answer
from .llm import LLMClient
from .retriever import Hit, Retriever, SearchConfig, load_chunks
from .validity import load_sources


@dataclass
class SignInfo:
    code: str
    name: str
    hits: list[Hit]
    query: str
    speak_fine: bool = True  # False: mức phạt tuỳ tình huống (biển tốc độ: quá 5 hay 35 km/h) -> đừng đọc 1 con số


class LawRAG:
    def __init__(self, retriever: Retriever, sign_map: dict[str, dict] | None = None, llm: LLMClient | None = None):
        self.retriever = retriever
        self.sign_map = sign_map or {}
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    @classmethod
    def from_dir(cls, data_dir: str | Path = "data", models_dir: str | Path | None = None, restorer=None,
                 threads: int | None = None) -> "LawRAG":
        data_dir = Path(data_dir)
        models_dir = Path(models_dir or os.getenv("VNLAW_MODELS", "models"))
        chunks = load_chunks(data_dir / "processed" / "chunks.jsonl")
        sources = load_sources(data_dir / "sources.json")

        dense = reranker = None
        dense_dir = data_dir / "processed" / "dense"
        emb_dir = models_dir / "embedder"
        if (dense_dir / "vectors.npy").exists() and (emb_dir / "tokenizer.json").exists():
            from .dense import DenseIndex

            onnx_file = next((p.name for p in sorted(emb_dir.glob("*.onnx"))), "model.onnx")
            dense = DenseIndex.load(dense_dir, model_override=str(emb_dir), backend_override="onnx",
                                    onnx_file=onnx_file, threads=threads)
        rr_dir = models_dir / "reranker"
        if (rr_dir / "tokenizer.json").exists():
            from .rerank import Reranker

            onnx_file = next((p.name for p in sorted(rr_dir.glob("*.onnx"))), "model.onnx")
            reranker = Reranker(str(rr_dir), backend="onnx", onnx_file=onnx_file, threads=threads)

        sign_map = {}
        sm = data_dir / "sign_map.json"
        if sm.exists():
            sign_map = {s["code"]: s for s in json.loads(sm.read_text("utf-8"))}
        return cls(Retriever(chunks, sources, dense, reranker, restorer), sign_map)

    def search(self, query: str, k: int = 5, as_of: date | None = None, cfg: SearchConfig | None = None) -> list[Hit]:
        return self.retriever.search(query, k=k, as_of=as_of, cfg=cfg)

    def answer(self, query: str, k: int = 5, as_of: date | None = None) -> Answer:
        return answer(query, self.search(query, k=k, as_of=as_of), self.llm)

    def lookup_sign(self, code: str, vehicle: str | None = None, k: int = 3) -> SignInfo | None:
        """Biển báo -> điều khoản xử phạt. vehicle: 'ô tô' / 'xe máy' để lấy đúng Điều."""
        s = self.sign_map.get(code.upper())
        if s is None:
            return None
        q = s["query"] + (f" {vehicle}" if vehicle else "")
        return SignInfo(code=s["code"], name=s["name"], hits=self.search(q, k=k), query=q,
                        speak_fine=s.get("speak_fine", True))
