"""Metric cho retrieval.

Gold là id chunk dạng "nd168-2024/d7/k10/a". Chunk có thể ở cấp khoản (khoản ngắn
không tách điểm) nên coi là khớp nếu một bên là tiền tố của bên kia.
"""


def id_match(hit_id: str, gold: str) -> bool:
    return hit_id == gold or hit_id.startswith(gold + "/") or gold.startswith(hit_id + "/")


def first_rank(hit_ids: list[str], golds: list[str]) -> int | None:
    for r, hid in enumerate(hit_ids, 1):
        if any(id_match(hid, g) for g in golds):
            return r
    return None


def recall_at_k(hit_ids: list[str], golds: list[str], k: int) -> float:
    r = first_rank(hit_ids[:k], golds)
    return 1.0 if r is not None else 0.0


def mrr(hit_ids: list[str], golds: list[str]) -> float:
    r = first_rank(hit_ids, golds)
    return 1.0 / r if r else 0.0


def complete_at_k(hit_ids: list[str], golds: list[str]) -> float:
    """Tất cả gold đều có mặt (vd. câu hỏi cần cả mức phạt lẫn số điểm bị trừ)."""
    return 1.0 if all(any(id_match(h, g) for h in hit_ids) for g in golds) else 0.0
