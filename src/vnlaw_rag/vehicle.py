"""Nhận biết loại phương tiện trong câu hỏi và trong tiêu đề Điều.

NĐ 168 chia Điều theo loại xe (ô tô / mô tô, xe gắn máy / xe đạp / người đi bộ...)
và nhiều khoản giữa các Điều gần như trùng chữ, chỉ khác mức phạt. Người dùng hỏi
"xe máy vượt đèn đỏ" mà retriever trả về Điều của ô tô là sai mức phạt hoàn toàn.

Phải so khớp theo ranh giới từ: "ô tô" là chuỗi con của "mô tô"!
"""

import re

from .text import nfc, remove_diacritics

VEHICLE_KEYWORDS: dict[str, list[str]] = {
    "oto": ["ô tô", "xe hơi", "xe con", "xe tải", "xe khách", "oto", "xe 4 bánh", "xe bốn bánh"],
    "moto": ["mô tô", "xe máy", "xe gắn máy", "xe moto", "xe máy điện", "xe 2 bánh", "xe hai bánh"],
    "bicycle": ["xe đạp", "xe thô sơ"],
    "pedestrian": ["người đi bộ"],
    "tractor": ["máy kéo", "xe máy chuyên dùng"],
}


def _pattern(kws: list[str]) -> re.Pattern:
    alts = sorted({k for kw in kws for k in (kw, remove_diacritics(kw))}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")


_PATTERNS = {v: _pattern(kws) for v, kws in VEHICLE_KEYWORDS.items()}


def detect_vehicles(text: str) -> set[str]:
    low = nfc(text).lower()
    # "xe máy chuyên dùng" là máy kéo/xe chuyên dùng, không phải xe máy
    low_moto = low.replace("xe máy chuyên dùng", " ").replace("xe may chuyen dung", " ")
    found = set()
    for vtype, pat in _PATTERNS.items():
        target = low_moto if vtype == "moto" else low
        if pat.search(target):
            found.add(vtype)
    return found


def vehicle_penalty(query_vehicles: set[str], article_title: str, penalty: float = 0.5) -> float:
    """Trả hệ số nhân điểm: 1.0 nếu khớp hoặc không xác định, `penalty` nếu lệch loại xe."""
    if not query_vehicles:
        return 1.0
    art = detect_vehicles(article_title)
    if not art or art & query_vehicles:
        return 1.0
    return penalty
