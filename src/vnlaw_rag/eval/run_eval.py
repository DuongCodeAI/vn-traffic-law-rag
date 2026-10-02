"""Chạy ablation retrieval trên bộ câu hỏi.

python -m vnlaw_rag.eval.run_eval --questions data/eval/questions.jsonl \
    --dense-dir data/processed/dense --reranker <model> --out results/retrieval.json

Không có --dense-dir / --reranker thì các config cần chúng tự bị bỏ qua.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from ..retriever import Retriever, SearchConfig, load_chunks
from ..validity import load_sources
from .metrics import complete_at_k, mrr, recall_at_k

OFF = SearchConfig(use_dense=False, use_rerank=False, vehicle_aware=False, expand_refs=False,
                   use_validity=False, restore_diacritics=False)

CONFIGS: dict[str, SearchConfig] = {
    "bm25": OFF,
    "dense": replace(OFF, use_bm25=False, use_dense=True),
    "hybrid_rrf": replace(OFF, use_dense=True),
    "+rerank": replace(OFF, use_dense=True, use_rerank=True),
    "+validity": replace(OFF, use_dense=True, use_rerank=True, use_validity=True),
    "+vehicle": replace(OFF, use_dense=True, use_rerank=True, use_validity=True, vehicle_aware=True),
    "+refexp (full)": SearchConfig(restore_diacritics=False),
    "full +restore": SearchConfig(),
}


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(retriever: Retriever, questions: list[dict], cfg: SearchConfig, k: int = 5) -> dict:
    per_type: dict[str, list[dict]] = defaultdict(list)
    lat = []
    outdated = 0
    n_hits = 0
    for q in questions:
        if q.get("type") == "unanswerable":
            continue
        as_of = date.fromisoformat(q["as_of"]) if q.get("as_of") else None
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k, as_of=as_of, cfg=cfg)
        lat.append(time.perf_counter() - t0)
        main = [h.id for h in hits if h.expanded_from is None]
        all_ids = [h.id for h in hits]
        golds = q["gold"]
        row = {
            "recall@1": recall_at_k(main, golds, 1),
            f"recall@{k}": recall_at_k(main, golds, k),
            "mrr": mrr(main, golds),
            "complete": complete_at_k(all_ids, golds + q.get("gold_extra", [])),
        }
        per_type["all"].append(row)
        per_type[q.get("type", "other")].append(row)
        if as_of is None:
            # câu hỏi về hiện tại mà lại trả về văn bản đã hết hiệu lực
            ok = {d for d, m in retriever.sources.items() if m.is_effective(date.today())}
            outdated += sum(1 for h in hits if h.chunk["doc_id"] not in ok)
            n_hits += len(hits)

    def avg(rows: list[dict]) -> dict:
        return {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in rows[0]} if rows else {}

    lat.sort()
    return {
        "by_type": {t: {**avg(rows), "n": len(rows)} for t, rows in per_type.items()},
        "outdated_hit_rate": round(outdated / max(n_hits, 1), 4),
        "latency_p50_ms": round(1000 * lat[len(lat) // 2], 1) if lat else None,
        "latency_p95_ms": round(1000 * lat[int(len(lat) * 0.95) - 1], 1) if lat else None,
    }


def to_markdown(results: dict[str, dict], k: int) -> str:
    head = (f"| config | recall@1 | recall@{k} | MRR | complete | outdated hits | p50 ms |\n"
            "|---|---|---|---|---|---|---|\n")
    rows = []
    for name, r in results.items():
        a = r["by_type"]["all"]
        rows.append(f"| {name} | {a['recall@1']:.3f} | {a[f'recall@{k}']:.3f} | {a['mrr']:.3f} | "
                    f"{a['complete']:.3f} | {r['outdated_hit_rate']:.1%} | {r['latency_p50_ms']} |")
    return head + "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--sources", default="data/sources.json")
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    ap.add_argument("--dense-dir", default=None)
    ap.add_argument("--dense-backend", default=None, help="onnx để chạy không cần torch")
    ap.add_argument("--dense-model", default=None, help="thư mục ONNX khi --dense-backend onnx")
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--reranker-backend", default="st")
    ap.add_argument("--reranker-onnx-file", default="model.onnx")
    ap.add_argument("--restorer", default=None, help="đường dẫn model thêm dấu (dự án vi-diacritics-transformer)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--configs", default=None, help="danh sách config, cách nhau bởi dấu phẩy")
    ap.add_argument("--out", default="results/retrieval.json")
    args = ap.parse_args()

    dense = reranker = restorer = None
    if args.dense_dir:
        from ..dense import DenseIndex

        dense = DenseIndex.load(args.dense_dir, model_override=args.dense_model, backend_override=args.dense_backend)
    if args.reranker:
        from ..rerank import Reranker

        reranker = Reranker(args.reranker, backend=args.reranker_backend, onnx_file=args.reranker_onnx_file)
    if args.restorer:
        from vi_diacritics import Restorer  # type: ignore  # cài từ repo vi-diacritics-transformer

        restorer = Restorer.load(args.restorer).restore

    retriever = Retriever(load_chunks(args.chunks), load_sources(args.sources), dense, reranker, restorer)
    questions = load_questions(args.questions)
    names = args.configs.split(",") if args.configs else list(CONFIGS)

    results = {}
    for name in names:
        cfg = CONFIGS[name]
        if (cfg.use_dense and dense is None) or (cfg.use_rerank and reranker is None):
            print(f"bỏ qua {name}: thiếu dense/reranker")
            continue
        if name == "full +restore" and restorer is None:
            continue
        if not cfg.use_bm25 and not cfg.use_dense:
            continue
        results[name] = run(retriever, questions, cfg, args.k)
        results[name]["config"] = asdict(cfg)
        print(name, results[name]["by_type"]["all"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    md = to_markdown(results, args.k)
    Path(args.out).with_suffix(".md").write_text(md, "utf-8")
    print(md)


if __name__ == "__main__":
    main()
