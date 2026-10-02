"""Cross-encoder rerank top ứng viên sau RRF.

Bi-encoder mã hoá câu hỏi và đoạn luật riêng rẽ nên bỏ lỡ chi tiết kiểu
"xe máy" vs "ô tô" (hai điều luật gần như giống hệt chỉ khác loại xe).
Cross-encoder đọc cả cặp cùng lúc nên phân biệt được, đổi lại chậm hơn ->
chỉ chạy trên ~20 ứng viên.
"""


class Reranker:
    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, texts: list[str], batch_size: int = 16) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=batch_size)]
