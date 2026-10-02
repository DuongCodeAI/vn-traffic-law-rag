"""Lấy văn bản từ vbpl.vn (CSDL quốc gia về pháp luật, Bộ Tư pháp).

Trang vbpl.vn bản mới (04/2026) là Next.js, nội dung render bằng JS nên
requests chỉ nhận được khung trang rỗng. Phải mở bằng trình duyệt (Playwright)
rồi đọc các thẻ <p class="prov-chapter|prov-section|prov-article|prov-clause|prov-item|prov-content">.

Điểm hay: vbpl hiển thị BẢN HỢP NHẤT (đã áp các văn bản sửa đổi) và gắn nhãn
"Điều khoản được sửa đổi, bổ sung / được bổ sung / được bãi bỏ" cho từng khoản.
Mình giữ lại nhãn này để citation ghi rõ "(đã được sửa đổi, bổ sung)".

python -m vnlaw_rag.vbpl --url https://vbpl.vn/van-ban/chi-tiet/nghi-dinh--173920 --out data/raw/nd168-2024.json
(cần: pip install playwright && playwright install chromium)
"""

import argparse
import json
from datetime import date
from pathlib import Path

EXTRACT_JS = r"""
() => [...document.querySelectorAll('[class*="prov-"]')].map(e => {
  const cls = [...e.classList].find(c => c.startsWith('prov-'));
  const hl = e.closest('[data-provision-highlighted="true"]');
  const btn = hl ? hl.querySelector('button') : null;
  return {cls, flag: btn ? btn.textContent.trim() : null, text: e.innerText.trim()};
})
"""

FLAG_MAP = {
    "Điều khoản được sửa đổi, bổ sung": "amended",
    "Điều khoản được bổ sung": "added",
    "Điều khoản được bãi bỏ": "repealed",
}


def fetch(url: str, timeout_ms: int = 60_000) -> list[dict]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        page.wait_for_selector(".prov-article", timeout=timeout_ms)
        rows = page.evaluate(EXTRACT_JS)
        browser.close()
    return rows


def rows_to_lines(rows: list[dict]) -> tuple[list[str], list[str | None]]:
    lines, flags = [], []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        # một <p> có thể chứa nhiều dòng (xuống dòng trong cùng một điểm)
        for i, line in enumerate(text.split("\n")):
            if line.strip():
                lines.append(line.strip())
                flags.append(FLAG_MAP.get(r.get("flag") or "", None) if i == 0 else None)
    return lines, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = fetch(args.url)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"url": args.url, "fetched": date.today().isoformat(), "rows": rows},
                                         ensure_ascii=False, indent=0), "utf-8")
    print(f"{len(rows)} dòng -> {args.out}")


if __name__ == "__main__":
    main()
