from datetime import date

from vnlaw_rag.bm25 import BM25
from vnlaw_rag.fusion import rrf
from vnlaw_rag.validity import DocMeta, effective_doc_ids

DOCS = [
    "Phạt tiền người điều khiển xe không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "Phạt tiền người điều khiển xe có nồng độ cồn trong máu hoặc hơi thở",
    "Dừng xe, đỗ xe trên đường cao tốc",
]


def test_bm25_finds_right_doc():
    bm = BM25().fit(DOCS)
    assert bm.search("vượt đèn đỏ đèn tín hiệu", k=1)[0][0] == 0
    assert bm.search("uống rượu nồng độ cồn", k=1)[0][0] == 1


def test_bm25_allowed_filter():
    bm = BM25().fit(DOCS)
    res = bm.search("phạt tiền", allowed={1})
    assert [i for i, _ in res] == [1]


def test_rrf_prefers_consensus():
    fused = rrf([[1, 2, 3], [2, 1, 3], [2, 3, 1]])
    assert fused[0][0] == 2


def test_validity():
    src = {
        "old": DocMeta("old", "NĐ 100", "", date(2020, 1, 1), effective_to=date(2025, 1, 1)),
        "new": DocMeta("new", "NĐ 168", "", date(2025, 1, 1)),
    }
    assert effective_doc_ids(src, date(2023, 6, 1)) == {"old"}
    assert effective_doc_ids(src, date(2025, 1, 1)) == {"new"}
