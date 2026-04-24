# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Multi-stage build for the MFA auto-grading FastAPI service.
#
# The build is split in two so the final image does NOT carry compilers,
# .pyc caches, or the wheel cache:
#
#   1. `builder` - installs build toolchains and pip-installs every
#      dependency into a self-contained virtualenv at /opt/venv.
#   2. `runtime` - copies /opt/venv into a clean slim image that only
#      has the shared libraries the wheels need at runtime.
#
# Torch is pulled from a configurable wheel index so the same Dockerfile
# builds a CPU image by default and a CUDA image on GPU hosts:
#
#   # CPU (default, portable)
#   docker build -t mfa-auto-grading .
#
#   # NVIDIA CUDA 12.8 (Blackwell / RTX 50xx, matches requirements.txt note)
#   docker build \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#     -t mfa-auto-grading:cuda .
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# =========================================================================
# Stage 1: builder
# =========================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX_URL

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Build-time system dependencies:
#   * build-essential / gcc  - compile psycopg2 / protobuf fallbacks
#   * libpq-dev              - headers for psycopg2 (belt & suspenders)
#   * git                    - `pip install git+https://.../transformers.git`
#                              (GLM-OCR requires the transformers dev head)
#   * curl, ca-certificates  - fetch pip/wheel indexes over HTTPS
# We purposely do NOT install these into the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        git \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip setuptools wheel

WORKDIR /build

# Copy only the requirements file first so the (expensive) pip layer is
# cached across rebuilds whenever application source changes but deps
# don't.
COPY requirements.txt ./

# Install torch (and torchvision, which transformers' AutoVideoProcessor
# needs at import time for Qwen2.5-VL) first, from the correct wheel
# index, so the transitive torch pulled in by `sentence-transformers` /
# `transformers` does not override it with a different CUDA build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.1.0" \
        "torchvision>=0.16.0"

# Install everything else. `torch` is already satisfied so pip will skip it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# =========================================================================
# Stage 2: runtime
# =========================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Keep HuggingFace / Torch caches inside a writable volume mount so
    # models (Qwen2.5-VL, GLM-OCR, sentence-transformers) survive
    # container restarts.
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# Runtime shared libraries:
#   * libpq5                     - psycopg2 runtime
#   * libgl1, libglib2.0-0       - Pillow/OpenCV/PyMuPDF image rendering
#   * libgomp1                   - OpenMP for torch
#   * libstdc++6, libgcc-s1      - C++ runtime for torch
#   * curl                       - HEALTHCHECK probe
#   * tini                       - proper PID 1 / signal forwarding
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libpq5 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libstdc++6 \
        libgcc-s1 \
        curl \
        tini \
 && rm -rf /var/lib/apt/lists/*

# Copy the fully-populated virtualenv built in stage 1.
COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user.
RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

# Copy only what the service needs at runtime. Tests, venv, caches, etc.
# are filtered out by .dockerignore.
COPY --chown=app:app app/ ./app/
COPY --chown=app:app alembic/ ./alembic/
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app docker-entrypoint.sh ./

# Strip any stray CRs in case the script was saved on Windows with CRLF
# line endings (otherwise the `#!/usr/bin/env bash` shebang becomes
# `bash\r` and the container exits with "No such file or directory").
RUN sed -i 's/\r$//' ./docker-entrypoint.sh \
 && chmod +x ./docker-entrypoint.sh \
 && mkdir -p /app/.cache /app/chroma_db /app/uploads \
 && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/docs >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
