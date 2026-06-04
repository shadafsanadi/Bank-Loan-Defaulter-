# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies needed by LightGBM and CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: app ──────────────────────────────────────────────────────────────
FROM base AS app

WORKDIR /app

# Copy project source (exclude raw data and notebooks via .dockerignore)
COPY src/       ./src/
COPY app/       ./app/
COPY api/       ./api/
COPY configs/   ./configs/
COPY models/    ./models/

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ── Stage 3: api ──────────────────────────────────────────────────────────────
FROM base AS api

WORKDIR /app
COPY src/       ./src/
COPY api/       ./api/
COPY configs/   ./configs/
COPY models/    ./models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
