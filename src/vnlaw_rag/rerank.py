"""Cross-encoder rerank top ứng viên sau RRF.

Bi-encoder mã hoá câu hỏi và đoạn luật riêng rẽ nên bỏ lỡ chi tiết kiểu
"xe máy" vs "ô tô" (hai điều luật gần như giống hệt chỉ khác loại xe).
Cross-encoder đọc cả cặp cùng lúc nên phân biệt được, đổi lại chậm hơn ->
chỉ chạy trên ~20 ứng viên.

Mặc định dùng cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (đa ngữ, có bản ONNX
quint8_avx2 ~119MB chạy được trên CPU không có AVX-512).
"""

from pathlib import Path


class Reranker:
    def __init__(self, model: str, backend: str = "st", onnx_file: str = "model.onnx",
                 device: str | None = None, max_length: int = 512, threads: int | None = None):
        self.backend = backend
        if backend == "onnx":
            from .onnx_models import OnnxCrossEncoder

            self._m = OnnxCrossEncoder(Path(model), onnx_file, max_length, threads)
        else:
            from sentence_transformers import CrossEncoder

            self._m = CrossEncoder(model, device=device, max_length=max_length)

    def rerank(self, query: str, texts: list[str], batch_size: int = 16) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._m.predict([(query, t) for t in texts], batch_size=batch_size)]
