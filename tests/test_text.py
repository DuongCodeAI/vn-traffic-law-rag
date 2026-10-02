from vnlaw_rag.text import normalize, remove_diacritics, tokenize


def test_tone_placement_unified():
    assert normalize("hoà") == normalize("hòa")
    assert normalize("thuỷ") == normalize("thủy")
    assert normalize("khoẻ") == normalize("khỏe")


def test_qu_not_touched():
    assert normalize("quý") == "quý"


def test_closed_syllable_not_touched():
    assert normalize("hoàng") == "hoàng"


def test_money_dots_removed():
    assert normalize("phạt 4.000.000 đồng") == "phạt 4000000 đồng"


def test_remove_diacritics():
    assert remove_diacritics("Đèn đỏ") == "Den do"


def test_tokenize_bigrams_and_fold():
    toks = tokenize("Vượt đèn đỏ")
    assert "vượt_đèn" in toks
    assert "vuot_den" in tokenize("Vượt đèn đỏ", fold=True)
