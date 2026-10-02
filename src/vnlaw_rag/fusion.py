"""Gộp nhiều danh sách kết quả bằng Reciprocal Rank Fusion.

Điểm BM25 (0..30) và cosine (0..1) khác thang đo nên không cộng thẳng được.
RRF chỉ dùng thứ hạng: score(d) = sum_i w_i / (k + rank_i(d)).
k=60 lấy theo paper gốc (Cormack et al., 2009); k lớn thì đuôi danh sách
vẫn còn đóng góp, k nhỏ thì top-1 của từng hệ áp đảo.
"""


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
