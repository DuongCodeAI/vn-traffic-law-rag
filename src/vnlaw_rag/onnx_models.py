"""Chạy embedding và reranker bằng onnxruntime + tokenizers, không cần torch.

Laptop của mình (Ryzen 5 5625U, không GPU) mà cài torch thì mất ~1GB chỉ để
encode câu hỏi. onnxruntime + tokenizers chưa tới 50MB.

Lưu ý: file int8 "avx512_vnni" trên HF không chạy được trên Zen 3,
phải dùng bản fp32 hoặc bản "quint8_avx2".
"""

from pathlib import Path

import numpy as np


def _session(model_path: str | Path, threads: int | None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def _tokenizer(tok_path: str | Path, max_length: int, pair: bool = False):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=max_length, strategy="only_second" if pair else "longest_first")
    tok.enable_padding()
    return tok


def _feed(session, enc) -> dict:
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    return feed


class OnnxEncoder:
    """Mean pooling + L2 normalize (đúng cách pooling của họ e5)."""

    def __init__(self, folder: str | Path, model_file: str = "model.onnx", max_length: int = 512,
                 threads: int | None = None):
        folder = Path(folder)
        self.session = _session(folder / model_file, threads)
        self.tok = _tokenizer(folder / "tokenizer.json", max_length)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            feed = _feed(self.session, enc)
            hidden = self.session.run(None, feed)[0]  # (b, seq, dim)
            mask = feed["attention_mask"][..., None].astype(np.float32)
            emb = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            out.append(emb / np.linalg.norm(emb, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


class OnnxCrossEncoder:
    def __init__(self, folder: str | Path, model_file: str = "model.onnx", max_length: int = 512,
                 threads: int | None = None):
        folder = Path(folder)
        self.session = _session(folder / model_file, threads)
        self.tok = _tokenizer(folder / "tokenizer.json", max_length, pair=True)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        scores = []
        for i in range(0, len(pairs), batch_size):
            enc = self.tok.encode_batch(pairs[i:i + batch_size])
            logits = self.session.run(None, _feed(self.session, enc))[0]
            scores.append(logits[:, 0] if logits.ndim == 2 else logits)
        return np.concatenate(scores)
