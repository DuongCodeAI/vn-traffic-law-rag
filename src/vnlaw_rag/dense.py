"""Vector search bằng numpy.

Corpus chỉ ~2-4 nghìn chunk -> ma trận vài MB, nhân ma trận tìm chính xác mất ~1ms.
Dựng Qdrant/FAISS cho cỡ này là thừa, thêm một service phải chạy mà không nhanh hơn.
Khi corpus lên cỡ trăm nghìn chunk thì mới đáng đổi sang ANN.

Hai backend:
- "st"   : sentence-transformers (torch) - dùng trên Kaggle để thử nhiều model
- "onnx" : onnxruntime - dùng trên laptop / HF Space, không cần torch
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbedConfig:
    model_name: str  # tên trên HF hub (backend st) hoặc thư mục chứa model.onnx + tokenizer.json (backend onnx)
    backend: str = "st"
    onnx_file: str = "model.onnx"
    query_prefix: str = ""
    passage_prefix: str = ""
    segment: bool = False  # bkai bi-encoder / PhoBERT cần tách từ bằng pyvi trước
    max_seq_length: int = 512


# cấu hình các model đã thử, xem NOTES.md
PRESETS = {
    "e5-small": EmbedConfig("intfloat/multilingual-e5-small", query_prefix="query: ", passage_prefix="passage: "),
    "bkai": EmbedConfig("bkai-foundation-models/vietnamese-bi-encoder", segment=True, max_seq_length=256),
}


class Embedder:
    def __init__(self, cfg: EmbedConfig, device: str | None = None, threads: int | None = None):
        self.cfg = cfg
        if cfg.backend == "onnx":
            from .onnx_models import OnnxEncoder

            self._onnx = OnnxEncoder(cfg.model_name, cfg.onnx_file, cfg.max_seq_length, threads)
            self._st = None
        else:
            from sentence_transformers import SentenceTransformer  # import muộn, CI không cần torch

            self._st = SentenceTransformer(cfg.model_name, device=device)
            self._st.max_seq_length = cfg.max_seq_length
        self._seg = None
        if cfg.segment:
            from pyvi import ViTokenizer

            self._seg = ViTokenizer.tokenize

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        if self._seg:
            texts = [self._seg(t) for t in texts]
        return [prefix + t for t in texts]

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self._st is not None:
            return self._st.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                   show_progress_bar=len(texts) > 100).astype(np.float32)
        return self._onnx.encode(texts, batch_size=batch_size)

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(self._prep(texts, self.cfg.passage_prefix), batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(self._prep([text], self.cfg.query_prefix), 1)[0]


class DenseIndex:
    def __init__(self, vectors: np.ndarray, embedder: Embedder | None = None):
        self.vectors = vectors
        self.embedder = embedder

    def search(self, query: str, k: int = 50, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        q = self.embedder.encode_query(query)
        sims = self.vectors @ q
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def save(self, folder: str | Path, cfg: EmbedConfig) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "embed_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), "utf-8")

    @classmethod
    def load(cls, folder: str | Path, device: str | None = None, model_override: str | None = None,
             backend_override: str | None = None, onnx_file: str | None = None,
             threads: int | None = None) -> "DenseIndex":
        """model_override/backend_override: index build trên Kaggle bằng torch,
        chạy trên laptop bằng bản ONNX của CÙNG model."""
        folder = Path(folder)
        cfg = EmbedConfig(**json.loads((folder / "embed_config.json").read_text("utf-8")))
        if model_override:
            cfg.model_name = model_override
        if backend_override:
            cfg.backend = backend_override
        if onnx_file:
            cfg.onnx_file = onnx_file
        return cls(np.load(folder / "vectors.npy"), Embedder(cfg, device=device, threads=threads))
