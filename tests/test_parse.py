from vnlaw_rag.parse import build_backrefs, build_chunks, extract_refs, parse_lines

# đoạn mẫu tự viết theo đúng format văn bản xử phạt (không phải trích nguyên văn)
SAMPLE = """
CHÍNH PHỦ
Căn cứ Luật Tổ chức Chính phủ;
Chương II
HÀNH VI VI PHẠM, HÌNH THỨC XỬ PHẠT
Mục 1. VI PHẠM QUY TẮC GIAO THÔNG
Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy tắc giao thông
1. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
a) Không chấp hành hiệu lệnh, chỉ dẫn của biển báo hiệu, vạch kẻ đường;
b) Không có báo hiệu xin vượt trước khi vượt;
2. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe thực hiện hành vi sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
b) Dừng xe, đỗ xe trên đường cao tốc;
3. Ngoài việc bị phạt tiền, người điều khiển xe thực hiện hành vi quy định tại điểm a khoản 2 Điều này còn bị trừ điểm giấy phép lái xe 4 điểm.
Điều 8. Điều khoản thi hành
Nghị định này có hiệu lực từ ngày 01 tháng 01 năm 2025.
"""


def _chunks():
    arts = parse_lines(SAMPLE.splitlines())
    return arts, build_chunks(arts, "nd-test", "Nghị định test", max_clause_chars=120)


def test_structure():
    arts, _ = _chunks()
    assert [a.label for a in arts] == ["7", "8"]
    a7 = arts[0]
    assert a7.chapter.startswith("Chương II")
    assert a7.section.startswith("Mục 1")
    assert [c.label for c in a7.clauses] == ["1", "2", "3"]
    assert [p.label for p in a7.clauses[1].points] == ["a", "b"]


def test_point_chunk_carries_fine_from_clause_lead():
    _, chunks = _chunks()
    c = next(c for c in chunks if c.id == "nd-test/d7/k2/a")
    assert "4.000.000" in c.text  # mức phạt nằm ở câu dẫn khoản, phải đi theo điểm
    assert "đèn tín hiệu" in c.text
    assert c.citation == "Điểm a, Khoản 2, Điều 7, Nghị định test"


def test_article_without_clauses():
    _, chunks = _chunks()
    c = next(c for c in chunks if c.id == "nd-test/d8")
    assert "hiệu lực" in c.text


def test_amendment_flags_go_to_chunk():
    lines = SAMPLE.strip().splitlines()
    flags = [None] * len(lines)
    i = next(i for i, ln in enumerate(lines) if ln.startswith("a) Không chấp hành hiệu lệnh của đèn"))
    flags[i] = "amended"
    chunks = build_chunks(parse_lines(lines, flags), "nd-test", "NĐ test", max_clause_chars=120)
    c = next(c for c in chunks if c.id == "nd-test/d7/k2/a")
    assert c.status == "amended"
    assert c.citation.endswith("(được sửa đổi, bổ sung)")
    assert next(c for c in chunks if c.id == "nd-test/d7/k2/b").status == ""


def test_refs_and_backrefs():
    assert extract_refs("hành vi quy định tại điểm a, điểm b khoản 2 Điều này", "x", "7") == [
        "x/d7/k2/a", "x/d7/k2/b"]
    assert extract_refs("theo khoản 3 Điều 6", "x", "7") == ["x/d6/k3"]
    _, chunks = _chunks()
    back = build_backrefs(chunks)
    assert back["nd-test/d7/k2/a"] == ["nd-test/d7/k3"]
