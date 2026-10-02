"""Nhận biết loại phương tiện trong câu hỏi và trong tiêu đề Điều.

NĐ 168 chia Điều theo loại xe (ô tô / mô tô, xe gắn máy / xe đạp / người đi bộ...)
và nhiều khoản giữa các Điều gần như trùng chữ, chỉ khác mức phạt. Người dùng hỏi
"xe máy vượt đèn đỏ" mà retriever trả về Điều của ô tô là sai mức phạt hoàn toàn.
"""

from .text import nfc, remove_diacritics

VEHICLE_KEYWORDS: dict[str, list[str]] = {
    "oto": ["ô tô", "xe hơi", "xe con", "xe tải", "xe khách", "oto", "xe 4 bánh", "xe bốn bánh"],
    "moto": ["mô tô", "xe máy", "xe gắn máy", "xe moto", "xe máy điện", "xe 2 bánh", "xe hai bánh"],
    "bicycle": ["xe đạp", "xe thô sơ"],
    "pedestrian": ["người đi bộ", "đi bộ"],
    "tractor": ["máy kéo", "xe máy chuyên dùng"],
}


def detect_vehicles(text: str) -> set[str]:
    low = nfc(text).lower()
    folded = remove_diacritics(low)
    found = set()
    for vtype, kws in VEHICLE_KEYWORDS.items():
        for kw in kws:
            if kw in low or remove_diacritics(kw) in folded:
                found.add(vtype)
                break
    # "xe máy chuyên dùng" chứa "xe máy" -> không tính là moto
    if "tractor" in found and "moto" in found and "xe máy chuyên dùng" in low and "mô tô" not in low:
        found.discard("moto")
    return found


def vehicle_penalty(query_vehicles: set[str], article_title: str, penalty: float = 0.5) -> float:
    """Trả hệ số nhân điểm: 1.0 nếu khớp hoặc không xác định, `penalty` nếu lệch loại xe."""
    if not query_vehicles:
        return 1.0
    art = detect_vehicles(article_title)
    if not art or art & query_vehicles:
        return 1.0
    return penalty
