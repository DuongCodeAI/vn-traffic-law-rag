"""Mở rộng câu hỏi: tiếng đời thường -> cách viết trong văn bản luật.

Thử thật trên NĐ 168: hỏi "xe máy vượt đèn đỏ" thì cả BM25, e5-small lẫn reranker
đều trả về "vượt xe", "vận chuyển hàng hoá vượt phía trước"... vì luật không hề
dùng chữ "vượt đèn đỏ" mà viết "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
Model nhỏ không đủ hiểu biết để nối hai cách nói này.

Thay vì gọi LLM viết lại câu hỏi (thêm ~1s và tốn quota), dùng một bảng nhỏ do mình
tự soạn khi đọc luật. Bảng chỉ THÊM cụm từ luật vào câu hỏi, không xoá gì.
"""

import re

from .text import nfc, remove_diacritics

# (các cách nói đời thường) -> cụm từ trong luật
GLOSSARY: list[tuple[list[str], str]] = [
    (["vượt đèn đỏ", "vượt đèn", "chạy đèn đỏ", "vượt đèn vàng"],
     "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"),
    (["uống rượu", "uống bia", "rượu bia", "say rượu", "say xỉn", "có cồn", "nhậu"],
     "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn"),
    (["không bằng lái", "không có bằng", "chưa có bằng", "không mang bằng", "quên bằng"],
     "không có giấy phép lái xe hoặc không mang theo giấy phép lái xe"),
    (["không đăng ký xe", "không cà vẹt", "quên cà vẹt", "không mang cà vẹt"],
     "không mang theo chứng nhận đăng ký xe"),
    (["không mũ", "không đội mũ", "không nón bảo hiểm", "không đội nón"],
     "không đội mũ bảo hiểm cho người đi mô tô, xe máy"),
    (["chở ba", "chở 3", "kẹp ba", "kẹp 3"],
     "chở theo từ 02 người trở lên trên xe"),
    (["lấn làn", "đi sai làn", "sai làn", "chạy sai làn"],
     "đi không đúng phần đường hoặc làn đường quy định"),
    (["quá tốc độ", "chạy nhanh quá", "vượt tốc độ"],
     "điều khiển xe chạy quá tốc độ quy định"),
    (["đi ngược chiều", "chạy ngược chiều"],
     "đi ngược chiều của đường một chiều, đi ngược chiều trên đường có biển cấm đi ngược chiều"),
    (["dùng điện thoại", "nghe điện thoại", "nhắn tin khi lái", "cầm điện thoại"],
     "dùng tay cầm và sử dụng điện thoại hoặc các thiết bị điện tử khác"),
    (["nẹt pô", "rú ga", "nẹt bô"], "rú ga liên tục"),
    (["đánh võng", "lạng lách"], "điều khiển xe lạng lách, đánh võng"),
    (["không gương", "thiếu gương", "không có kính chiếu hậu"], "không có gương chiếu hậu"),
    (["đỗ sai chỗ", "đậu xe sai", "đỗ xe sai", "đậu sai chỗ"], "dừng xe, đỗ xe không đúng nơi quy định"),
    (["cấm đỗ", "cấm đậu"], "đỗ xe nơi có biển “Cấm đỗ xe” hoặc biển “Cấm dừng xe và đỗ xe”"),
    (["không xi nhan", "không bật xi nhan", "không xinhan"], "chuyển hướng không có tín hiệu báo hướng rẽ"),
    (["quay đầu sai", "quay đầu chỗ cấm"], "quay đầu xe tại nơi có biển báo hiệu có nội dung cấm quay đầu"),
    (["bỏ chạy", "gây tai nạn rồi bỏ"], "gây tai nạn giao thông không dừng ngay phương tiện, bỏ trốn"),
    (["bằng lái", "giấy phép lái"], "giấy phép lái xe"),
]


def _compile(phrases: list[str]) -> re.Pattern:
    alts = sorted({p for ph in phrases for p in (ph, remove_diacritics(ph))}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")


_RULES = [(_compile(phr), legal) for phr, legal in GLOSSARY]


def expand(query: str) -> tuple[str, list[str]]:
    """Trả (câu hỏi đã mở rộng, các cụm luật đã thêm)."""
    low = nfc(query).lower()
    added = [legal for pat, legal in _RULES if pat.search(low)]
    if not added:
        return query, []
    return query + " " + " ; ".join(added), added
