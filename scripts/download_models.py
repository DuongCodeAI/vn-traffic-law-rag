"""Tải index + model ONNX đã build trên Kaggle từ HF Hub về máy (~250MB).

python scripts/download_models.py --repo DuongCodeAI/vn-traffic-law-rag-index
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="DuongCodeAI/vn-traffic-law-rag-index")
    args = ap.parse_args()
    src = Path(snapshot_download(args.repo, allow_patterns=["models/*", "processed/dense/*",
                                                            "processed/chunks.jsonl"]))
    shutil.copytree(src / "models", "models", dirs_exist_ok=True)
    shutil.copytree(src / "processed", "data/processed", dirs_exist_ok=True)
    print("xong: models/, data/processed/")


if __name__ == "__main__":
    main()
