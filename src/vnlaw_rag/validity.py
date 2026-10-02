"""Lọc văn bản theo thời điểm hiệu lực.

Corpus cố tình chứa cả văn bản đã hết hiệu lực (NĐ 100/2019) để trả lời được câu
kiểu "năm 2023 vượt đèn đỏ phạt bao nhiêu". Mặc định as_of = hôm nay nên văn bản
cũ bị loại khỏi tập ứng viên ngay từ đầu, trước cả BM25/vector.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class DocMeta:
    id: str
    number: str  # "Nghị định 168/2024/NĐ-CP"
    title: str
    effective_from: date
    effective_to: date | None = None  # None = còn hiệu lực
    replaced_by: str | None = None
    url: str = ""

    def is_effective(self, as_of: date) -> bool:
        if as_of < self.effective_from:
            return False
        return self.effective_to is None or as_of < self.effective_to


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def load_sources(path: str | Path) -> dict[str, DocMeta]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in raw:
        out[r["id"]] = DocMeta(
            id=r["id"], number=r["number"], title=r["title"],
            effective_from=_d(r["effective_from"]), effective_to=_d(r.get("effective_to")),
            replaced_by=r.get("replaced_by"), url=r.get("url", ""),
        )
    return out


def effective_doc_ids(sources: dict[str, DocMeta], as_of: date | None = None) -> set[str]:
    as_of = as_of or date.today()
    return {d.id for d in sources.values() if d.is_effective(as_of)}
