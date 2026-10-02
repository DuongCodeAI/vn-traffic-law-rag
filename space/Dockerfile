# HF Space (Docker SDK). Không cài torch: embedding + rerank chạy bằng onnxruntime.
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir "vnlaw-rag[onnx,api] @ git+https://github.com/DuongCodeAI/vn-traffic-law-rag" huggingface_hub

COPY --chown=1000 . /app
RUN useradd -m -u 1000 user && chown -R user /app
USER user

ENV VNLAW_DATA=/app/data VNLAW_MODELS=/app/models
# index + model tải lúc khởi động (để image nhẹ, đổi index không cần build lại image)
CMD python download.py && uvicorn vnlaw_rag.api:app --host 0.0.0.0 --port 7860
