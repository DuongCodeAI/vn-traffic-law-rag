"""Test retriever chỉ với BM25 (không cần torch)."""

from datetime import date

from vnlaw_rag.parse import build_chunks, parse_lines
from vnlaw_rag.retriever import Retriever, SearchConfig
from vnlaw_rag.validity import DocMeta

NEW = """
Điều 6. Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông
1. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe thực hiện hành vi sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
b) Đỗ xe trên đường cao tốc;
2. Ngoài việc bị phạt tiền, người điều khiển xe thực hiện hành vi quy định tại điểm a khoản 1 Điều này còn bị trừ điểm giấy phép lái xe 4 điểm.
Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy tắc giao thông
1. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện hành vi sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
b) Đi vào đường cao tốc;
"""
OLD = """
Điều 6. Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy tắc giao thông
1. Phạt tiền từ 600.000 đồng đến 1.000.000 đồng đối với người điều khiển xe thực hiện hành vi sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
b) Chở người ngồi sau không đội mũ bảo hiểm;
"""

SOURCES = {
    "new": DocMeta("new", "NĐ mới", "", date(2025, 1, 1)),
    "old": DocMeta("old", "NĐ cũ", "", date(2020, 1, 1), effective_to=date(2025, 1, 1)),
}


def _retriever():
    chunks = []
    for doc_id, text in (("new", NEW), ("old", OLD)):
        chunks += [c.to_dict() for c in build_chunks(parse_lines(text.splitlines()), doc_id, doc_id, 120)]
    return Retriever(chunks, SOURCES)


CFG = SearchConfig(use_dense=False, use_rerank=False)


def test_vehicle_aware_picks_motorbike_article():
    r = _retriever()
    hits = r.search("xe máy không chấp hành đèn tín hiệu giao thông", k=1, as_of=date(2025, 6, 1), cfg=CFG)
    assert hits[0].id == "new/d7/k1/a"


def test_validity_excludes_old_doc():
    r = _retriever()
    hits = r.search("xe máy đèn tín hiệu", k=5, as_of=date(2025, 6, 1), cfg=CFG)
    assert all(h.chunk["doc_id"] == "new" for h in hits)
    old = r.search("xe máy đèn tín hiệu", k=1, as_of=date(2023, 6, 1), cfg=CFG)
    assert old[0].chunk["doc_id"] == "old"


def test_no_diacritics_query_uses_folded_index():
    r = _retriever()
    hits = r.search("xe may khong chap hanh den tin hieu", k=1, as_of=date(2025, 6, 1), cfg=CFG)
    assert hits[0].id == "new/d7/k1/a"


def test_reference_expansion_adds_point_deduction():
    r = _retriever()
    hits = r.search("ô tô vượt đèn tín hiệu giao thông", k=1, as_of=date(2025, 6, 1), cfg=CFG)
    assert hits[0].id == "new/d6/k1/a"
    assert any(h.expanded_from == "new/d6/k1/a" and h.id == "new/d6/k2" for h in hits)
