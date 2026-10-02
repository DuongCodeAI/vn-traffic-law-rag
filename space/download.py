"""Chạy lúc Space khởi động: tải index + model ONNX từ HF Hub, metadata từ GitHub."""

import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "DuongCodeAI/vn-traffic-law-rag-index"
GH = "https://raw.githubusercontent.com/DuongCodeAI/vn-traffic-law-rag/main/data"

p = Path(snapshot_download(REPO, allow_patterns=["models/*", "processed/dense/*", "processed/chunks.jsonl"]))
shutil.copytree(p / "models", "models", dirs_exist_ok=True)
shutil.copytree(p / "processed", "data/processed", dirs_exist_ok=True)
for f in ("sources.json", "sign_map.json"):
    urllib.request.urlretrieve(f"{GH}/{f}", f"data/{f}")
print("ok:", [str(x) for x in Path("models").rglob("*.onnx")])
