"""Tách văn bản luật thành cây Chương > Mục > Điều > Khoản > Điểm rồi cắt chunk.

Ý chính: KHÔNG cắt theo số ký tự. Một điểm như
    "a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;"
đứng một mình thì vô nghĩa, mức phạt nằm ở câu dẫn của khoản:
    "1. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với ... sau đây:"
nên mỗi chunk cấp điểm = tiêu đề Điều + câu dẫn Khoản + nội dung Điểm.
"""

import re
from dataclasses import dataclass, field

from .text import nfc

CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC]+)\b\.?\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Mục\s+(\d+)\b\.?\s*(.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Điều\s+(\d+[a-z]?)\s*[.:]\s*(.*)$")
CLAUSE_RE = re.compile(r"^(\d+[a-z]?)\.\s+(.+)$")
POINT_RE = re.compile(r"^([a-zđ]\d?)\)\s+(.+)$")

# Một nhóm tham chiếu: "điểm a, điểm b khoản 2" hoặc "khoản 3". Điều của nhóm là chữ "Điều N / Điều này"
# đứng SAU nó gần nhất trong cùng câu, vì NĐ 168 hay viết:
#   "điểm h, điểm i khoản 3; điểm a, điểm b khoản 4; điểm c khoản 10 Điều này"
REF_GROUP_RE = re.compile(r"((?:điểm\s+[a-zđ]\d?\s*,?\s*)*)khoản\s+(\d+[a-z]?)\b", re.IGNORECASE)
REF_ARTICLE_RE = re.compile(r"Điều\s+(\d+[a-z]?|này)\b")
REF_POINT_RE = re.compile(r"điểm\s+([a-zđ]\d?)", re.IGNORECASE)
# câu có những cụm này mới là "hình phạt bổ sung" trỏ ngược về hành vi (khác với "trừ các hành vi ..." = ngoại lệ)
PENALTY_RE = re.compile(r"còn bị|bị trừ điểm|trừ điểm giấy phép|tước quyền sử dụng|tịch thu|buộc ", re.IGNORECASE)


@dataclass
class Point:
    label: str
    text: str
    status: str | None = None  # amended / added / repealed (nhãn của vbpl)


@dataclass
class Clause:
    label: str
    text: str  # câu dẫn
    points: list[Point] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)  # dòng tiếp theo không phải điểm
    status: str | None = None


@dataclass
class Article:
    label: str
    title: str
    chapter: str = ""
    section: str = ""
    intro: list[str] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    doc_id: str
    article: str
    clause: str
    point: str
    text: str  # text đưa vào index (đã ghép ngữ cảnh)
    citation: str
    refs: list[str] = field(default_factory=list)  # id các chunk mà chunk này nhắc tới
    status: str = ""  # "", amended, added, repealed

    def to_dict(self) -> dict:
        return self.__dict__.copy()


STATUS_NOTE = {"amended": " (được sửa đổi, bổ sung)", "added": " (được bổ sung)", "repealed": " (đã bị bãi bỏ)"}


def parse_lines(lines: list[str], flags: list[str | None] | None = None) -> list[Article]:
    """flags[i]: nhãn sửa đổi của dòng i (lấy từ vbpl), None nếu không có."""
    articles: list[Article] = []
    chapter = section = ""
    cur: Article | None = None
    pending_chapter_title = pending_section_title = False
    flags = flags or [None] * len(lines)

    for raw, flag in zip(lines, flags, strict=True):
        line = nfc(raw).strip()
        if not line:
            continue

        if m := CHAPTER_RE.match(line):
            chapter = f"Chương {m.group(1).upper()}"
            if m.group(2):
                chapter += f". {m.group(2)}"
                pending_chapter_title = False
            else:
                pending_chapter_title = True
            section = ""
            cur = None
            continue
        if pending_chapter_title and cur is None and line.isupper():
            chapter += f". {line.capitalize()}"
            pending_chapter_title = False
            continue
        if m := SECTION_RE.match(line):
            section = f"Mục {m.group(1)}. {m.group(2)}".strip(". ")
            pending_section_title = not m.group(2)
            cur = None
            continue
        if pending_section_title and cur is None and line.isupper():
            section += f". {line.capitalize()}"
            pending_section_title = False
            continue
        if m := ARTICLE_RE.match(line):
            cur = Article(label=m.group(1), title=m.group(2).strip(), chapter=chapter, section=section)
            articles.append(cur)
            pending_chapter_title = False
            continue
        if cur is None:
            continue  # phần đầu văn bản (căn cứ, quốc hiệu...) bỏ qua

        if m := CLAUSE_RE.match(line):
            cur.clauses.append(Clause(label=m.group(1), text=m.group(2).strip(), status=flag))
        elif (m := POINT_RE.match(line)) and cur.clauses:
            cur.clauses[-1].points.append(Point(label=m.group(1), text=m.group(2).strip(), status=flag))
        elif cur.clauses:
            c = cur.clauses[-1]
            if c.points:
                # dòng nối tiếp của điểm (gạch đầu dòng, xuống dòng giữa câu)
                c.points[-1].text += "\n" + line
            else:
                c.extra.append(line)
        else:
            cur.intro.append(line)
    return articles


def _cite(point: str, clause: str, article: str, doc_number: str) -> str:
    parts = []
    if point:
        parts.append(f"Điểm {point}")
    if clause:
        parts.append(f"Khoản {clause}")
    parts.append(f"Điều {article}")
    return ", ".join(parts) + f", {doc_number}"


def chunk_id(doc_id: str, article: str, clause: str = "", point: str = "") -> str:
    cid = f"{doc_id}/d{article}"
    if clause:
        cid += f"/k{clause}"
    if point:
        cid += f"/{point}"
    return cid


def extract_refs(text: str, doc_id: str, article: str) -> list[str]:
    refs = []
    for m in REF_GROUP_RE.finditer(text):
        points_str, clause = m.groups()
        tail = text[m.end():]
        end = tail.find(".")
        am = REF_ARTICLE_RE.search(tail[: end if end >= 0 else None])
        art = article if am is None or am.group(1) == "này" else am.group(1)
        pts = REF_POINT_RE.findall(points_str or "")
        if pts:
            refs += [chunk_id(doc_id, art, clause, p.lower()) for p in pts]
        else:
            refs.append(chunk_id(doc_id, art, clause))
    return refs


def build_chunks(articles: list[Article], doc_id: str, doc_number: str, max_clause_chars: int = 1500) -> list[Chunk]:
    chunks: list[Chunk] = []
    for a in articles:
        head = f"Điều {a.label}. {a.title}"
        if not a.clauses:
            body = "\n".join(a.intro)
            chunks.append(Chunk(
                id=chunk_id(doc_id, a.label), doc_id=doc_id, article=a.label, clause="", point="",
                text=f"{head}\n{body}".strip(), citation=_cite("", "", a.label, doc_number),
                refs=extract_refs(body, doc_id, a.label),
            ))
            continue
        for c in a.clauses:
            lead = f"{c.label}. {c.text}"
            extra = "\n".join(c.extra)
            full_clause = "\n".join([lead] + [f"{p.label}) {p.text}" for p in c.points] + ([extra] if extra else []))
            # khoản ngắn thì giữ nguyên cả khoản, đỡ mất ngữ cảnh giữa các điểm
            if not c.points or len(full_clause) <= max_clause_chars // 3:
                st = c.status or next((p.status for p in c.points if p.status), "") or ""
                chunks.append(Chunk(
                    id=chunk_id(doc_id, a.label, c.label), doc_id=doc_id, article=a.label, clause=c.label,
                    point="", text=f"{head}\n{full_clause}",
                    citation=_cite("", c.label, a.label, doc_number) + STATUS_NOTE.get(st, ""),
                    refs=extract_refs(full_clause, doc_id, a.label), status=st,
                ))
                continue
            for p in c.points:
                text = f"{head}\n{lead}\n{p.label}) {p.text}"
                st = p.status or c.status or ""
                chunks.append(Chunk(
                    id=chunk_id(doc_id, a.label, c.label, p.label), doc_id=doc_id, article=a.label,
                    clause=c.label, point=p.label, text=text,
                    citation=_cite(p.label, c.label, a.label, doc_number) + STATUS_NOTE.get(st, ""),
                    refs=extract_refs(p.text, doc_id, a.label), status=st,
                ))
            if extra:
                chunks.append(Chunk(
                    id=chunk_id(doc_id, a.label, c.label) + "/_", doc_id=doc_id, article=a.label,
                    clause=c.label, point="", text=f"{head}\n{lead}\n{extra}",
                    citation=_cite("", c.label, a.label, doc_number),
                    refs=extract_refs(extra, doc_id, a.label),
                ))
    # vbpl có chỗ đánh trùng nhãn điểm (vd. 2 điểm "đ)" liền nhau ở Điều 16 NĐ 168) -> thêm hậu tố cho id
    seen: dict[str, int] = {}
    for ch in chunks:
        if ch.id in seen:
            seen[ch.id] += 1
            ch.id = f"{ch.id}~{seen[ch.id]}"
        else:
            seen[ch.id] = 1
    return chunks


def build_backrefs(chunks: list[Chunk]) -> dict[str, list[str]]:
    """chunk X được những chunk nào nhắc tới.

    Trong NĐ 168, hình phạt bổ sung (trừ điểm GPLX, tước bằng) nằm ở khoản cuối
    của Điều và chỉ ghi "Thực hiện hành vi quy định tại điểm a khoản 1 Điều này...".
    Truy hồi điểm a mà không kéo theo khoản đó thì câu trả lời thiếu mất phần trừ điểm.
    """
    ids = {c.id for c in chunks}
    back: dict[str, list[str]] = {}
    for c in chunks:
        body = c.text.split("\n", 1)[-1]
        if not PENALTY_RE.search(body):
            continue  # "trừ các hành vi quy định tại..." là ngoại lệ, không phải hình phạt bổ sung
        for r in c.refs:
            targets = [r] if r in ids else [i for i in ids if i.startswith(r + "/")]
            if not targets and r.count("/") >= 3:
                # trỏ tới điểm nhưng khoản đó không bị tách theo điểm -> lấy cả khoản
                parent = r.rsplit("/", 1)[0]
                targets = [parent] if parent in ids else []
            for t in targets:
                if t == c.id:
                    continue
                back.setdefault(t, []).append(c.id)
    return back
