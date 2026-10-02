"""Vector search bằng numpy.

Corpus chỉ ~2-4 nghìn chunk -> ma trận vài MB, nhân ma trận tìm chính xác mất ~1ms.
Dựng Qdrant/FAISS cho cỡ này là thừa, thêm một service phải chạy mà không nhanh hơn.
Khi corpus lên cỡ trăm nghìn chunk thì mới đáng đổi sang ANN.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EmbedConfig:
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    segment: bool = False  # một số model VN (vd. bkai bi-encoder) train trên text đã tách từ bằng pyvi
    max_seq_length: int = 512


class Embedder:
    def __init__(self, cfg: EmbedConfig, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # import muộn, CI không cần torch

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=device)
        self.model.max_seq_length = cfg.max_seq_length
        self._seg = None
        if cfg.segment:
            from pyvi import ViTokenizer

            self._seg = ViTokenizer.tokenize

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        if self._seg:
            texts = [self._seg(t) for t in texts]
        return [prefix + t for t in texts]

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(self._prep(texts, self.cfg.passage_prefix), batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(self._prep([text], self.cfg.query_prefix), normalize_embeddings=True)[0]


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
        (folder / "embed_config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), "utf-8")

    @classmethod
    def load(cls, folder: str | Path, device: str | None = None) -> "DenseIndex":
        folder = Path(folder)
        cfg = EmbedConfig(**json.loads((folder / "embed_config.json").read_text("utf-8")))
        return cls(np.load(folder / "vectors.npy"), Embedder(cfg, device=device))
