# ---- build stage ----
FROM python:3.10-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ---- runtime stage ----
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY main.py chunker.py ./

# Copy model files (expected at models/BAAI/bge-large-zh-v1.5/)
COPY models/ ./models/

# Point to the local model path; override at runtime if needed
ENV EMBEDDING_MODEL=models/BAAI/bge-large-zh-v1.5

# Prevent HuggingFace from attempting network downloads at runtime
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 8003

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
