"""Fail CI nếu recall@5 của BM25 tụt dưới ngưỡng (ngưỡng đặt hơi thấp hơn kết quả đo được lần đầu)."""

import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--min-recall5", type=float, required=True)
args = ap.parse_args()

r = json.load(open(args.path, encoding="utf-8"))["bm25"]["by_type"]["all"]
print(r)
if r["recall@5"] < args.min_recall5:
    sys.exit(f"recall@5 = {r['recall@5']:.3f} < {args.min_recall5}")
