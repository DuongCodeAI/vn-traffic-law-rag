"""lookup_sign: mã kèm hậu tố từ detector, điểm luật ghim theo loại xe, biển không áp dụng cho loại xe."""

import json
from datetime import date
from pathlib import Path

from vnlaw_rag.parse import build_chunks, parse_lines
from vnlaw_rag.pipeline import LawRAG
from vnlaw_rag.retriever import Retriever
from vnlaw_rag.validity import DocMeta

LAW = """
Điều 6. Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông
5. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
i) Đi vào khu vực cấm, đường có biển báo hiệu có nội dung cấm đi vào đối với loại phương tiện đang điều khiển;
16. Ngoài việc bị phạt tiền, người điều khiển xe thực hiện hành vi vi phạm còn bị trừ điểm giấy phép lái xe như sau:
b) Thực hiện hành vi quy định tại điểm i khoản 5 Điều này bị trừ điểm giấy phép lái xe 02 điểm.
Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy tắc giao thông
6. Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
b) Đi vào khu vực cấm, đường có biển báo hiệu có nội dung cấm đi vào đối với loại phương tiện đang điều khiển;
"""

SIGNS = [
    {"code": "P.103A", "name": "Cấm xe ô tô", "fine": {"ô tô": "nd/d6/k5/i", "xe máy": None}},
    {"code": "P.104", "name": "Cấm xe mô tô và xe máy", "fine": {"ô tô": None, "xe máy": "nd/d7/k6/b"}},
    {"code": "P.127", "name": "Tốc độ tối đa cho phép", "query": "chạy quá tốc độ", "speak_fine": False, "unit": "km/h"},
    {"code": "W.225", "name": "Trẻ em"},
]


def _rag():
    chunks = [c.to_dict() for c in build_chunks(parse_lines(LAW.splitlines()), "nd", "nd", 400)]
    ret = Retriever(chunks, {"nd": DocMeta("nd", "NĐ", "", date(2025, 1, 1))})
    return LawRAG(ret, {s["code"]: s for s in SIGNS})


def test_pinned_clause_and_points():
    info = _rag().lookup_sign("P.103a", "ô tô")
    assert info.hits[0].id == "nd/d6/k5/i" and info.speak_fine
    # khoản trừ điểm trỏ về điểm i được kéo theo
    assert any(h.expanded_from == "nd/d6/k5/i" for h in info.hits)


def test_sign_not_for_this_vehicle():
    # biển cấm ô tô: xe máy đi vào không vi phạm -> chỉ báo tên biển, không đọc mức phạt
    info = _rag().lookup_sign("P.103A", "xe máy")
    assert info.hits == [] and not info.speak_fine
    assert _rag().lookup_sign("P.104", "ô tô").speak_fine is False
    assert _rag().lookup_sign("P.104", "xe mô tô").hits[0].id == "nd/d7/k6/b"


def test_detector_suffix():
    rag = _rag()
    info = rag.lookup_sign("P.127*50", "ô tô")
    assert info.name == "Tốc độ tối đa cho phép 50 km/h" and not info.speak_fine
    assert rag.lookup_sign("P.103a*", "ô tô").hits[0].id == "nd/d6/k5/i"


def test_warning_sign_announce_only():
    info = _rag().lookup_sign("W.225", "ô tô")
    assert info.name == "Trẻ em" and info.hits == [] and not info.speak_fine


def test_sign_map_pins_exist():
    # mọi điểm luật ghim trong data/sign_map.json phải có trong corpus
    root = Path(__file__).resolve().parents[1] / "data"
    chunks = root / "processed" / "chunks.jsonl"
    if not chunks.exists():
        return
    ids = {json.loads(line)["id"] for line in chunks.open(encoding="utf-8")}
    for s in json.loads((root / "sign_map.json").read_text("utf-8")):
        for cid in s.get("fine", {}).values():
            assert cid is None or cid in ids, (s["code"], cid)
