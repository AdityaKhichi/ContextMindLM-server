# =============================================================================
# Stage 1: BUILDER
# =============================================================================
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false

# Install deps + fix torch + cleanup in ONE layer
RUN poetry install --only main --no-interaction --no-ansi --no-root \
    && pip uninstall -y torch torchvision \
    && pip install --no-cache-dir torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache \
    && rm -rf /usr/local/lib/python3.13/site-packages/nvidia* \
    && rm -rf /usr/local/lib/python3.13/site-packages/triton* \
    && rm -rf /usr/local/lib/python3.13/site-packages/cuda*

# =============================================================================
# Stage 2: RUNTIME
# =============================================================================
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]