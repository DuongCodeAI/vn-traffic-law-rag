"""Chuẩn bị model ONNX để chạy trên laptop mà KHÔNG cần Kaggle:
- e5-small: tải onnx fp32 của intfloat rồi tự quantize dynamic -> uint8 (~120MB)
  (bản int8 sẵn trên HF là avx512_vnni, không chạy được trên CPU AMD Zen 3)
- reranker mmarco-mMiniLMv2: lấy luôn bản quint8_avx2 (~119MB)
Sau đó embed toàn bộ chunk -> data/processed/dense/.

python scripts/prepare_local_models.py
"""

import os
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

E5 = "intfloat/multilingual-e5-small"
RR = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def main():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from vnlaw_rag.dense import PRESETS, DenseIndex, Embedder
    from vnlaw_rag.retriever import load_chunks

    emb_dir, rr_dir = Path("models/embedder"), Path("models/reranker")
    emb_dir.mkdir(parents=True, exist_ok=True)
    rr_dir.mkdir(parents=True, exist_ok=True)

    if not (emb_dir / "model.int8.onnx").exists():
        fp32 = hf_hub_download(E5, "onnx/model.onnx")
        quantize_dynamic(fp32, str(emb_dir / "model.int8.onnx"), weight_type=QuantType.QUInt8)
        os.remove(os.path.realpath(fp32))  # bỏ bản fp32 470MB khỏi cache cho nhẹ ổ
    for f in ("tokenizer.json", "config.json"):
        shutil.copy(hf_hub_download(E5, f), emb_dir / f)
    if not (rr_dir / "model_quint8_avx2.onnx").exists():
        shutil.copy(hf_hub_download(RR, "onnx/model_quint8_avx2.onnx"), rr_dir / "model_quint8_avx2.onnx")
    shutil.copy(hf_hub_download(RR, "tokenizer.json"), rr_dir / "tokenizer.json")
    for p in sorted(Path("models").rglob("*.onnx")):
        print(p, f"{p.stat().st_size / 1e6:.0f} MB")

    from dataclasses import replace

    cfg = replace(PRESETS["e5-small"], model_name=str(emb_dir), backend="onnx", onnx_file="model.int8.onnx")
    chunks = load_chunks("data/processed/chunks.jsonl")
    t0 = time.time()
    vec = Embedder(cfg).encode_passages([c["text"] for c in chunks], batch_size=16)
    DenseIndex(vec).save("data/processed/dense", PRESETS["e5-small"])
    print(f"embed {len(chunks)} chunk: {time.time() - t0:.0f}s, shape {vec.shape}")


if __name__ == "__main__":
    main()
