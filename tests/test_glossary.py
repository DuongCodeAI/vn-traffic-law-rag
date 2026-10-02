from vnlaw_rag.glossary import expand


def test_red_light_expanded():
    q, added = expand("xe máy vượt đèn đỏ phạt bao nhiêu")
    assert "đèn tín hiệu giao thông" in q
    assert len(added) == 1


def test_no_diacritics_also_matches():
    _, added = expand("uong ruou lai xe")
    assert added and "nồng độ cồn" in added[0]


def test_no_false_trigger():
    assert expand("biển báo cấm rẽ trái")[1] == []
