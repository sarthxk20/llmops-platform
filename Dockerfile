# ── Stage 1: builder — install all dependencies ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for tokenizers, faiss, bitsandbytes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install Python deps into a prefix directory (for easy copy)
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime — lean final image ───────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user (security best practice)
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy application source
COPY --chown=appuser:appuser serving/     ./serving/
COPY --chown=appuser:appuser training/    ./training/
COPY --chown=appuser:appuser monitoring/  ./monitoring/
COPY --chown=appuser:appuser data/        ./data/

# Model and adapter are mounted at runtime via K8s volume (not baked in)
# to keep the image small. Set via env vars:
ENV BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ENV ADAPTER_PATH="/mnt/model/adapter"
ENV FAISS_INDEX_PATH="/mnt/model/faiss_index"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check (matches K8s liveness probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
