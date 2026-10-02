"""Chuẩn hoá tiếng Việt cho việc tìm kiếm.

Văn bản luật lấy từ nhiều nguồn nên gặp đủ kiểu: Unicode dựng sẵn / tổ hợp,
"hoà" vs "hòa", "thuỷ" vs "thủy", số tiền "4.000.000" vs "4000000"...
Không chuẩn hoá thì BM25 coi chúng là các từ khác nhau.
"""

import re
import unicodedata
from itertools import pairwise

# dấu thanh ở dạng combining sau NFD: huyền, sắc, ngã, hỏi, nặng
_TONES = {"̀", "́", "̃", "̉", "̣"}

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUM_DOT_RE = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _split_tone(syl: str) -> tuple[str, str | None]:
    tone = None
    out = []
    for c in unicodedata.normalize("NFD", syl):
        if c in _TONES:
            tone = c
        else:
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out)), tone


def normalize_tone_placement(syl: str) -> str:
    """Đưa 'hoà/hoá/thuý/khoẻ' về một kiểu duy nhất: 'hòa/hóa/thúy/khỏe'.

    Chỉ xử lý âm tiết mở kết thúc bằng oa/oe/uy, vì đây là chỗ hai kiểu bỏ dấu
    khác nhau. 'qu' + y (quý, quỳ) giữ nguyên vì 'u' thuộc phụ âm đầu.
    """
    bare, tone = _split_tone(syl)
    if tone is None:
        return syl
    low = bare.lower()
    if not low.endswith(("oa", "oe", "uy")):
        return syl
    if low.endswith("uy") and low[:-2].endswith("q"):
        return syl
    pos = len(bare) - 2  # dấu đặt lên nguyên âm đầu của cặp
    decomposed = unicodedata.normalize("NFD", bare[pos]) + tone
    fixed = bare[:pos] + unicodedata.normalize("NFC", decomposed) + bare[pos + 1:]
    return fixed


def normalize(text: str) -> str:
    text = nfc(text)
    text = _NUM_DOT_RE.sub("", text)  # 4.000.000 -> 4000000
    text = _WORD_RE.sub(lambda m: normalize_tone_placement(m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


def remove_diacritics(text: str) -> str:
    text = nfc(text).replace("đ", "d").replace("Đ", "D")
    d = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def has_diacritics(text: str) -> bool:
    return remove_diacritics(text) != nfc(text)


def tokenize(text: str, bigrams: bool = True, fold: bool = False) -> list[str]:
    """Tách âm tiết + ghép bigram.

    Không dùng word segmentation (underthesea/pyvi) cho BM25: bigram âm tiết
    bắt được phần lớn từ ghép 2 âm tiết ('vượt_đèn', 'nồng_độ') mà không thêm
    dependency nặng. fold=True thì bỏ dấu luôn (dùng cho truy vấn gõ không dấu).
    """
    text = normalize(text).lower()
    if fold:
        text = remove_diacritics(text)
    syls = _WORD_RE.findall(text)
    if not bigrams:
        return syls
    return syls + [f"{a}_{b}" for a, b in pairwise(syls)]

