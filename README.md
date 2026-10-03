# vn-traffic-law-rag

Hỏi đáp Luật Giao thông Việt Nam có trích dẫn tới từng **Điểm / Khoản / Điều**, chạy được trên laptop không GPU.

```
hỏi: xe máy vượt đèn đỏ phạt bao nhiêu?
→ Điểm c, Khoản 7, Điều 7, NĐ 168/2024/NĐ-CP: phạt 4.000.000 – 6.000.000 đồng
→ (kéo theo) Điểm b, Khoản 13, Điều 7: trừ điểm giấy phép lái xe
```

Dữ liệu: Nghị định 168/2024/NĐ-CP (bản hợp nhất trên vbpl.vn, đã gồm sửa đổi của NĐ 238/2026 và NĐ 336/2025)
và Luật Trật tự, an toàn giao thông đường bộ 36/2024/QH15. Snapshot ngày 02/10/2026.

Đây là phần "bộ não pháp luật" của [viet-copilot](https://github.com/DuongCodeAI/viet-copilot).

## Kết quả

Bộ eval tự làm: 29 câu hỏi kiểu người dân hỏi + 29 bản gõ không dấu tương ứng + 3 câu ngoài phạm vi.
Đáp án (id điều khoản) tìm bằng cụm từ chính xác trong luật rồi kiểm tra tay, không dùng retriever để tạo đáp án.
"complete" = lấy được **đủ** các điều khoản cần (vd. vừa mức phạt vừa khoản trừ điểm GPLX).

Đo trên laptop Ryzen 5 5625U, không GPU, model ONNX int8:

| config | recall@1 | recall@5 | MRR | complete | ms/câu |
|---|---|---|---|---|---|
| BM25 | 0.172 | 0.741 | 0.383 | 0.379 | 3 |
| dense (e5-small) | 0.086 | 0.276 | 0.149 | 0.138 | 7 |
| hybrid (RRF) | 0.138 | 0.569 | 0.292 | 0.224 | 12 |
| + rerank | 0.207 | 0.517 | 0.314 | 0.224 | 970 |
| + lọc hiệu lực | 0.207 | 0.534 | 0.318 | 0.241 | 973 |
| + nhận biết loại xe | 0.310 | 0.638 | 0.417 | 0.276 | 988 |
| + từ điển đời thường → luật | **0.483** | **0.862** | **0.625** | 0.362 | 980 |
| + mở rộng tham chiếu (full) | **0.483** | **0.862** | **0.625** | **0.810** | 1003 |
| BM25 + từ điển (không model) | 0.224 | 0.862 | 0.447 | 0.414 | 9 |

Những gì mình rút ra (chi tiết trong [NOTES.md](NOTES.md)):

- **e5-small yếu với văn bản luật tiếng Việt**, thua cả BM25, và ghép vào hybrid còn kéo tụt kết quả.
  Sẽ thử embedding chuyên tiếng Việt (bkai) trên Kaggle.
- **Lỗi lớn nhất là khoảng cách từ vựng**: người dân nói "vượt đèn đỏ", luật viết "không chấp hành hiệu lệnh
  của đèn tín hiệu giao thông". Một bảng ~20 dòng tự soạn tăng recall@5 từ 0.64 lên 0.86.
- **Mở rộng tham chiếu** nâng "complete" từ 0.36 lên 0.81: phần trừ điểm GPLX luôn nằm ở khoản khác,
  chỉ ghi "thực hiện hành vi quy định tại điểm c khoản 7 Điều này".
- Bộ eval còn nhỏ, các số trên để so sánh tương đối giữa các config, không phải con số tuyệt đối.

## Cách hoạt động

```
câu hỏi ─► (thêm dấu nếu gõ không dấu) ─► từ điển đời thường → luật
        ─► lọc văn bản/điều khoản còn hiệu lực
        ─► BM25 (+ bản bỏ dấu) ┐
        ─► vector e5 (ONNX)    ┴► RRF ─► cross-encoder rerank ─► phạt lệch loại xe
        ─► kéo theo khoản "trừ điểm / tước bằng" trỏ ngược về hành vi ─► LLM trả lời + trích dẫn
        ─► kiểm tra mọi số tiền/số điểm trong câu trả lời có trong đoạn được trích
```

- **Chunk theo cấu trúc luật**, không theo số ký tự: mỗi chunk cấp điểm = tiêu đề Điều + câu dẫn Khoản
  (chứa mức phạt) + nội dung Điểm.
- **Nhãn sửa đổi từ vbpl**: khoản "được sửa đổi, bổ sung" ghi rõ trong trích dẫn; khoản "bãi bỏ" bị loại.
- **Kiểm tra số liệu**: LLM hay nhớ nhầm mức phạt của NĐ 100 cũ; mọi số tiền trong câu trả lời phải xuất
  hiện trong đoạn luật được trích, không thì gắn cảnh báo.
- **Không cần torch khi chạy**: embedding và reranker chạy bằng onnxruntime (~240MB model).

## Chạy thử

```bash
pip install -e ".[onnx,api]"
python scripts/prepare_local_models.py   # tải + quantize model, embed corpus (~2 phút)
cp .env.example .env                     # điền GEMINI_API_KEY / GROQ_API_KEY
uvicorn vnlaw_rag.api:app --port 8000    # mở http://localhost:8000
```

Không có API key vẫn dùng được chế độ "chỉ tìm điều khoản".

```python
from vnlaw_rag.pipeline import LawRAG
rag = LawRAG.from_dir("data")
rag.search("xe may vuot den do")
rag.lookup_sign("P.131a", vehicle="ô tô")   # biển báo -> điều khoản xử phạt (dùng trong viet-copilot)
```

Đánh giá lại: `python -m vnlaw_rag.eval.run_eval --dense-dir data/processed/dense --dense-backend onnx --dense-model models/embedder --reranker models/reranker --reranker-backend onnx`

Dựng lại index + ablation: `notebooks/01_build_index_eval` (Kaggle hoặc Colab, CPU là đủ).
Chạy trên Colab: mở notebook từ GitHub (File → Open notebook → GitHub → DuongCodeAI/vn-traffic-law-rag), runtime CPU,
thêm Secrets `GEMINI_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN` (biểu tượng chìa khoá, bật Notebook access).
Notebook tự clone code, kết quả lưu vào Google Drive (`MyDrive/ai-portfolio/vn-traffic-law-rag`).

## Hạn chế

- Luật 118/2025/QH15 sửa ~20 điều của Luật 36 (hiệu lực 01/07/2026) nhưng vbpl chưa cập nhật bản hợp nhất.
- Chưa có NĐ 100/2019 (văn bản cũ) nên chưa đo được phần lọc hiệu lực theo thời gian.
- Mức phạt riêng của Hà Nội (NQ 56/2026/NQ-HĐND) chưa có.
- Không thay thế tư vấn pháp lý.

## Cấu trúc

```
src/vnlaw_rag/   parse (luật -> chunk), bm25, dense/onnx_models, rerank, fusion (RRF), glossary,
                 vehicle, validity, retriever, answer (LLM + kiểm tra số), api, eval/
data/raw/        snapshot vbpl (JSON)        data/eval/  bộ câu hỏi
notebooks/       Kaggle: dựng index, so sánh embedding
space/           Dockerfile cho HF Space
```
