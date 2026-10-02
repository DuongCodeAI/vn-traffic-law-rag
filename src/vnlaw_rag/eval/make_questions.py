"""Sinh câu hỏi ứng viên cho bộ eval từ chunk, rồi DUYỆT TAY trước khi dùng.

Câu hỏi do LLM sinh thường "chép" lại chữ trong luật ("không chấp hành hiệu lệnh
của đèn tín hiệu") nên BM25 thắng dễ dàng -> eval bị lạc quan. Prompt bắt viết theo
kiểu người dân hỏi ("vượt đèn đỏ"), và mỗi câu còn có một bản gõ không dấu.

python -m vnlaw_rag.eval.make_questions --n 120 --out data/eval/candidates.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from ..llm import LLMClient
from ..retriever import build_backrefs_from_dicts, load_chunks
from ..text import remove_diacritics

SYSTEM = """Bạn tạo dữ liệu đánh giá cho hệ thống hỏi đáp luật giao thông.
Cho một đoạn luật, viết 2 câu hỏi mà NGƯỜI DÂN BÌNH THƯỜNG sẽ hỏi, đáp án nằm trong đoạn đó.
Yêu cầu:
- Dùng từ ngữ đời thường, KHÔNG chép nguyên cụm từ trong luật
  (vd. "vượt đèn đỏ" thay vì "không chấp hành hiệu lệnh của đèn tín hiệu").
- Nêu rõ loại xe nếu đoạn luật chỉ áp dụng cho một loại xe.
- Câu 1 hỏi mức phạt hoặc hậu quả; câu 2 hỏi theo tình huống cụ thể.
- answer_hint: tóm tắt đáp án đúng trong 1 câu (mức phạt, số điểm trừ nếu có).
Trả về JSON: {"questions": [{"question": "...", "answer_hint": "..."}, ...]}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--doc", default="nd168-2024", help="chỉ lấy chunk của văn bản này")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/eval/candidates.jsonl")
    args = ap.parse_args()

    all_chunks = load_chunks(args.chunks)
    back = build_backrefs_from_dicts(all_chunks)
    chunks = [c for c in all_chunks if c["doc_id"] == args.doc and len(c["text"]) > 120]
    random.Random(args.seed).shuffle(chunks)
    llm = LLMClient()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("a", encoding="utf-8") as f:
        for c in chunks:
            if n_written >= args.n:
                break
            try:
                resp = llm.complete(SYSTEM, f"ĐOẠN LUẬT ({c['citation']}):\n{c['text']}", temperature=0.7,
                                    json_mode=True)
                qs = json.loads(resp.text)["questions"]
            except Exception as e:  # noqa: BLE001 - bỏ qua chunk lỗi, chạy tiếp
                print("lỗi", c["id"], e)
                continue
            for q in qs[:2]:
                for variant, text in (("fine", q["question"]), ("no_diacritics", remove_diacritics(q["question"]))):
                    n_written += 1
                    rec = {"id": f"g{n_written:04d}", "question": text, "gold": [c["id"]], "type": variant,
                           "gold_extra": back.get(c["id"], []),
                           "answer_hint": q.get("answer_hint", ""), "source_chunk": c["id"], "reviewed": False}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(n_written, c["id"])


if __name__ == "__main__":
    main()
