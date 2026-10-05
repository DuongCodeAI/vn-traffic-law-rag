# Ghi chú quyết định

Ghi lại vì sao chọn cách này, cái gì đã thử và thất bại.

## Dữ liệu

- vbpl.vn (bản 04/2026) render bằng JS, `requests` chỉ nhận khung trang rỗng. Đọc các thẻ
  `<p class="prov-article|prov-clause|prov-item">` sau khi trang render (Playwright, `vnlaw_rag/vbpl.py`).
- vbpl hiển thị **bản hợp nhất** + nhãn "Điều khoản được sửa đổi, bổ sung / được bổ sung / được bãi bỏ".
  NĐ 168: 54 khoản sửa đổi, 7 bổ sung, 2 bãi bỏ. Nhãn được đưa vào chunk và citation.
- `parent-id` của điểm trên vbpl trỏ thẳng lên Điều chứ không lên Khoản → dựng cây theo thứ tự dòng.
- Dữ liệu gốc có chỗ đánh trùng nhãn (2 điểm "đ)" liền nhau ở Điều 16) → thêm hậu tố `~2` cho id.
- Văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả (Luật SHTT, Điều 15 khoản 2) nên commit snapshot được.

## Chunking

- Cắt theo số ký tự thì "a) Không chấp hành hiệu lệnh của đèn tín hiệu" bị tách khỏi mức phạt ở câu dẫn khoản.
  → chunk cấp điểm = tiêu đề Điều + câu dẫn Khoản + Điểm. Khoản ngắn giữ nguyên cả khoản.
- 1.686 chunk, dài trung vị ~410 ký tự. Một số điểm rất dài (liệt kê 5-6 hành vi): "đỗ xe nơi có biển Cấm đỗ xe"
  nằm giữa Điểm e Khoản 3 Điều 6 dài 815 ký tự → embedding bị loãng.

## Tham chiếu ngược (refexp)

- Hình phạt bổ sung (trừ điểm, tước bằng) nằm ở khoản cuối Điều: "Thực hiện hành vi quy định tại điểm h,
  điểm i khoản 3; điểm a khoản 4; ... Điều này bị trừ điểm giấy phép lái xe 02 điểm".
- Nhiều nhóm "điểm.. khoản.." dùng chung một "Điều này" ở cuối câu → regex tìm "Điều" gần nhất phía sau trong câu.
- Phải phân biệt với "trừ các hành vi vi phạm quy định tại..." (ngoại lệ). Chỉ lấy tham chiếu từ câu có
  "còn bị / trừ điểm / tước quyền sử dụng / tịch thu / buộc". Không lọc thì kéo theo cả đống ngoại lệ vô ích.

## Retrieval

- BM25 tự viết: corpus nhỏ, <5ms/câu, chỉnh được tokenizer. Tokenizer = âm tiết + bigram âm tiết thay vì
  tách từ (underthesea/pyvi): bắt được phần lớn từ ghép 2 âm tiết mà không thêm dependency.
- Thêm một index BM25 trên text bỏ dấu cho câu gõ không dấu. Nhược điểm thấy ngay khi thử:
  "o to vuot den do" → "do" khớp cả "đỏ" lẫn "độ" (nồng độ cồn). Đây là lý do cần model thêm dấu (dự án 3).
- Vector search bằng numpy (ma trận 1686×384): nhanh hơn dựng Qdrant cho cỡ này.
- **e5-small ONNX int8**: bản int8 có sẵn trên HF là avx512_vnni, CPU AMD Zen 3 không chạy được → tự quantize
  dynamic uint8 (118MB). Embed 1.686 chunk mất 78s trên laptop.
- Kết quả bất ngờ: dense e5-small recall@5 chỉ 0.28, thua BM25 (0.74). Đoán: model đa ngữ nhỏ, chưa thấy văn
  phong luật Việt; câu hỏi đời thường còn xa văn bản luật hơn nữa. Việc tiếp: so với bkai bi-encoder (cần GPU Colab),
  và thử fine-tune e5 bằng cặp (câu hỏi, điều khoản) sinh từ LLM.
- Reranker mmarco-mMiniLMv2 quint8_avx2: 20 cặp ~1.3s, 12 cặp ~0.6-1s trên laptop. Giữ 12.

## Từ điển đời thường → luật

- Thử thật: "xe máy vượt đèn đỏ" → cả BM25, vector, reranker đều trả về "vượt xe", "vận chuyển hàng hoá vượt
  phía trước". "vượt" bị hiểu là vượt xe; luật viết "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
- Chọn bảng tra tự soạn thay vì gọi LLM viết lại câu hỏi: không tốn thêm ~1s + quota, giải thích được, test được.
  Nhược điểm: chỉ phủ các cách nói mình nghĩ ra → cần mở rộng dần theo log câu hỏi thật.
- Bẫy nghiệp vụ: "chở ba" = 3 người trên xe tính cả người lái = luật ghi "chở theo **02** người".
  Lần đầu mình map sai sang "từ 03 người trở lên".

## Loại phương tiện

- NĐ 168 chia Điều theo loại xe, nhiều khoản giữa Điều 6 (ô tô) và Điều 7 (xe máy) gần như trùng chữ, chỉ khác tiền.
- Bug đã gặp: "ô tô" là chuỗi con của "m**ô tô**" → tiêu đề Điều 7 bị nhận là cả ô tô. Phải so theo ranh giới từ.

## Sinh câu trả lời

- Gemini (`gemini-3.5-flash`, dòng 2.5 đã đóng với dự án mới), Groq `gpt-oss-120b` dự phòng. Gọi REST thẳng,
  không cài SDK.
- Sau khi sinh: mọi số tiền/số điểm trong câu trả lời phải có trong đoạn được trích, không thì cảnh báo.

## Việc còn lại

- Lấy NĐ 100/2019 (trang vbpl mở bằng tab mới, cần bấm tay) để đo lọc hiệu lực theo thời gian.
- Mở rộng bộ eval lên ~150 câu (script `eval/make_questions.py` sinh ứng viên, duyệt tay).
- Ưu tiên NĐ 168 khi câu hỏi có "phạt / bao nhiêu" (hiện Luật 36 đôi khi xếp trên).
