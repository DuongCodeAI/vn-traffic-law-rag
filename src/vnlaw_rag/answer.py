"""Sinh câu trả lời có trích dẫn + kiểm tra số liệu.

LLM hay "bịa" mức phạt nhất (nhớ nhầm từ NĐ 100 cũ hoặc trộn khung phạt của
ô tô với xe máy). Nên sau khi sinh, mọi con số tiền / số điểm trong câu trả lời
phải xuất hiện trong đoạn luật được trích; không có thì gắn cờ cảnh báo.
"""

import json
import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .retriever import Hit
from .text import normalize

SYSTEM = """Bạn là trợ lý tra cứu luật giao thông Việt Nam.
Chỉ dùng các đoạn luật trong NGỮ CẢNH để trả lời. Không dùng kiến thức bên ngoài.
- Trả lời ngắn gọn, đúng trọng tâm, bằng tiếng Việt.
- Mỗi ý phải ghi nguồn dạng [số] theo số thứ tự đoạn trong ngữ cảnh.
- Nêu rõ mức phạt tiền và hình phạt bổ sung (trừ điểm, tước giấy phép...) nếu ngữ cảnh có.
- Nếu ngữ cảnh không đủ để trả lời, đặt "not_found": true và nói rõ là không tìm thấy.
Trả về JSON: {"answer": "...", "citations": [1, 2], "not_found": false}"""

MONEY_RE = re.compile(r"(\d[\d.,]*)\s*(đồng|triệu|nghìn|ngàn)", re.IGNORECASE)
POINTS_RE = re.compile(r"trừ\s+(?:điểm[^\d]{0,30})?(\d+)\s*điểm", re.IGNORECASE)


@dataclass
class Answer:
    answer: str
    citations: list[dict]
    not_found: bool = False
    warnings: list[str] = field(default_factory=list)
    provider: str = ""
    latency_s: float = 0.0


def build_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        tag = f" (liên quan tới {h.expanded_from})" if h.expanded_from else ""
        parts.append(f"[{i}] {h.citation}{tag}\n{h.chunk['text']}")
    return "\n\n".join(parts)


def _money_values(text: str) -> set[int]:
    """'4.000.000 đồng' -> 4000000, '4 triệu' -> 4000000."""
    out = set()
    for num, unit in MONEY_RE.findall(text):
        digits = num.replace(".", "").replace(",", "")
        if not digits.isdigit():
            continue
        v = int(digits)
        unit = unit.lower()
        if unit == "triệu":
            v *= 1_000_000
        elif unit in ("nghìn", "ngàn"):
            v *= 1_000
        out.add(v)
    return out


def check_numbers(answer: str, cited_texts: list[str]) -> list[str]:
    ctx = " ".join(cited_texts)
    warnings = []
    missing_money = _money_values(answer) - _money_values(ctx)
    if missing_money:
        warnings.append(f"Số tiền không có trong nguồn trích dẫn: {sorted(missing_money)}")
    ans_pts = set(POINTS_RE.findall(normalize(answer)))
    ctx_pts = set(POINTS_RE.findall(normalize(ctx)))
    if ans_pts - ctx_pts:
        warnings.append(f"Số điểm trừ không có trong nguồn: {sorted(ans_pts - ctx_pts)}")
    return warnings


def answer(query: str, hits: list[Hit], llm: LLMClient | None = None) -> Answer:
    if not hits:
        return Answer("Không tìm thấy điều khoản liên quan.", [], not_found=True)
    llm = llm or LLMClient()
    user = f"NGỮ CẢNH:\n{build_context(hits)}\n\nCÂU HỎI: {query}"
    try:
        resp = llm.complete(SYSTEM, user, json_mode=True)
    except LLMError as e:
        # LLM chết hết thì vẫn trả được top đoạn luật, người dùng tự đọc
        return Answer("Không gọi được LLM, dưới đây là các điều khoản liên quan nhất.",
                      [_cite(h) for h in hits[:3]], warnings=[str(e)])
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        data = {"answer": resp.text, "citations": [], "not_found": False}

    idxs = [i for i in data.get("citations", []) if isinstance(i, int) and 1 <= i <= len(hits)]
    cited = [hits[i - 1] for i in idxs]
    warnings = []
    if len(idxs) != len(data.get("citations", [])):
        warnings.append("LLM trích dẫn số thứ tự không tồn tại")
    if not data.get("not_found") and not cited:
        warnings.append("Câu trả lời không có trích dẫn")
    warnings += check_numbers(data.get("answer", ""), [h.chunk["text"] for h in cited])
    return Answer(
        answer=data.get("answer", ""),
        citations=[_cite(h) for h in cited],
        not_found=bool(data.get("not_found")),
        warnings=warnings,
        provider=f"{resp.provider}/{resp.model}",
        latency_s=resp.latency_s,
    )


def _cite(h: Hit) -> dict:
    return {"id": h.id, "citation": h.citation, "text": h.chunk["text"]}
