"""Câu hỏi gõ không dấu: tìm thẳng, thêm dấu rồi tìm, hay tìm cả 2 bản rồi ghép?

    pip install -e ../vi-diacritics-transformer      # hoặc thêm src của repo đó vào PYTHONPATH
    python scripts/eval_restorer.py --restorer path/to/diacritics-model

Model thêm dấu: thư mục có tagger.int8.onnx + vocab.json (HF hgdkakhs/vi-diacritics-tagger).
"""
import argparse
import time

from vi_diacritics import Restorer
from vnlaw_rag.dense import DenseIndex
from vnlaw_rag.eval.metrics import complete_at_k, mrr, recall_at_k
from vnlaw_rag.eval.run_eval import CONFIGS, load_questions
from vnlaw_rag.rerank import Reranker
from vnlaw_rag.retriever import Retriever, load_chunks
from vnlaw_rag.validity import load_sources


def score(main, all_ids, q):
    return {"recall@1": recall_at_k(main, q["gold"], 1), "recall@5": recall_at_k(main, q["gold"], 5),
            "mrr": mrr(main, q["gold"]), "complete": complete_at_k(all_ids, q["gold"] + q.get("gold_extra", []))}


def search_one(ret, text, cfg):
    hits = ret.search(text, k=5, cfg=cfg)
    return [h.id for h in hits if h.expanded_from is None], [h.id for h in hits]


def search_both(ret, text, restored, cfg):
    # 2 lượt tìm, ghép danh sách chính bằng RRF; khoản kéo theo (trừ điểm...) đi theo khoản gốc của nó
    rrf, extra = {}, {}
    for t in (text, restored):
        hits = ret.search(t, k=10, cfg=cfg)
        for r, h in enumerate([h for h in hits if h.expanded_from is None], 1):
            rrf[h.id] = rrf.get(h.id, 0) + 1 / (60 + r)
        for h in hits:
            if h.expanded_from is not None:
                extra.setdefault(h.expanded_from, []).append(h.id)
    main = sorted(rrf, key=lambda i: -rrf[i])[:5]
    return main, main + [e for m in main for e in extra.get(m, [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorer", required=True)
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    args = ap.parse_args()

    dense = DenseIndex.load("data/processed/dense", model_override="models/embedder", backend_override="onnx")
    rer = Reranker("models/reranker", backend="onnx", onnx_file="model_quint8_avx2.onnx")
    ret = Retriever(load_chunks("data/processed/chunks.jsonl"), load_sources("data/sources.json"), dense, rer)
    restore = Restorer.load(args.restorer).restore
    cfg = CONFIGS["+refexp (full)"]

    qs = load_questions(args.questions)
    # mỗi câu không dấu "s001n" có câu gốc có dấu "s001"
    by_id = {q["id"]: q for q in qs}
    nod = [q for q in qs if q.get("type") == "no_diacritics"]

    runs = {
        "câu có dấu gốc (trần)": lambda q: search_one(ret, by_id[q["id"][:-1]]["question"], cfg),
        "không dấu, tìm thẳng": lambda q: search_one(ret, q["question"], cfg),
        "không dấu -> thêm dấu -> tìm": lambda q: search_one(ret, restore(q["question"]), cfg),
        "tìm cả 2 bản, RRF": lambda q: search_both(ret, q["question"], restore(q["question"]), cfg),
    }
    print("| đầu vào | recall@1 | recall@5 | MRR | complete | ms/câu |\n|---|---|---|---|---|---|")
    for name, fn in runs.items():
        rows, lat = [], []
        for q in nod:
            t0 = time.perf_counter()
            main_ids, all_ids = fn(q)
            lat.append(time.perf_counter() - t0)
            rows.append(score(main_ids, all_ids, q))
        avg = {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}
        p50 = 1000 * sorted(lat)[len(lat) // 2]
        print(f"| {name} | {avg['recall@1']:.3f} | {avg['recall@5']:.3f} | {avg['mrr']:.3f} | "
              f"{avg['complete']:.3f} | {p50:.0f} |")


if __name__ == "__main__":
    main()
