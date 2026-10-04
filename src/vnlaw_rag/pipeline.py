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
        """Biển báo -> điều khoản xử phạt. vehicle: 'ô tô' / 'xe máy' để lấy đúng Điều.

        Mã từ detector có thể kèm hậu tố sau dấu '*' (VNTS: "P.127*50", "P.124a*", "P.106a*Xe tải").
        Biển có trường "fine" thì lấy thẳng điểm luật đã soát tay theo loại xe; không tìm kiếm, vì
        cảnh báo phải đúng số tiền. null = biển không áp dụng cho loại xe đó -> chỉ báo tên biển.
        """
        base, _, suffix = code.strip().partition("*")
        s = self.sign_map.get(base.upper())
        if s is None:
            return None
        name = s["name"]
        if suffix.isdigit() and s.get("unit"):
            name = f"{name} {suffix} {s['unit']}"
        veh = _vehicle_key(vehicle)
        if "fine" in s and veh in s["fine"]:
            cid = s["fine"][veh]
            if cid is None:
                return SignInfo(code=s["code"], name=name, hits=[], query="", speak_fine=False)
            return SignInfo(code=s["code"], name=name, hits=self._pinned(cid), query=cid,
                            speak_fine=s.get("speak_fine", True))
        if not s.get("query"):  # biển cảnh báo/chỉ dẫn: không có hành vi bị phạt riêng
            return SignInfo(code=s["code"], name=name, hits=[], query="", speak_fine=False)
        q = s["query"] + (f" {vehicle}" if vehicle else "")
        return SignInfo(code=s["code"], name=name, hits=self.search(q, k=k), query=q,
                        speak_fine=s.get("speak_fine", True))

    def _pinned(self, cid: str) -> list[Hit]:
        r = self.retriever
        hits = [Hit(r.chunks[r.id2idx[cid]], 1.0)]
        # kéo theo khoản trừ điểm / tước bằng trỏ về điểm này, giống expand_refs khi tìm kiếm
        hits += [Hit(r.chunks[r.id2idx[x]], 0.0, expanded_from=cid) for x in r.backrefs.get(cid, []) if x in r.id2idx]
        return hits


def _vehicle_key(vehicle: str | None) -> str | None:
    if not vehicle:
        return None
    v = vehicle.lower()
    if "máy" in v or "mô tô" in v:
        return "xe máy"
    return "ô tô" if "ô tô" in v else None
