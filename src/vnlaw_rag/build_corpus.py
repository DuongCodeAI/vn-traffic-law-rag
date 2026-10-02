"""data/raw/*.json (snapshot vbpl) -> data/processed/chunks.jsonl

python -m vnlaw_rag.build_corpus
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .parse import build_backrefs, build_chunks, parse_lines
from .validity import load_sources
from .vbpl import rows_to_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--sources", default="data/sources.json")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    args = ap.parse_args()

    sources = load_sources(args.sources)
    all_chunks = []
    for doc_id, meta in sources.items():
        raw = Path(args.raw) / f"{doc_id}.json"
        if not raw.exists():
            print(f"thiếu {raw}, bỏ qua")
            continue
        rows = json.loads(raw.read_text("utf-8"))["rows"]
        lines, flags = rows_to_lines(rows)
        arts = parse_lines(lines, flags)
        chunks = build_chunks(arts, doc_id, meta.number)
        st = Counter(c.status or "gốc" for c in chunks)
        print(f"{doc_id}: {len(arts)} điều, {len(chunks)} chunk, trạng thái {dict(st)}")
        all_chunks += chunks

    back = build_backrefs(all_chunks)
    print(f"tổng {len(all_chunks)} chunk, {sum(len(v) for v in back.values())} tham chiếu ngược")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
