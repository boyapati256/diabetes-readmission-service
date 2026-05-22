# ── Base image ────────────────────────────────────────────────────────────────
# Slim Python image — no GPU needed, keeps image size down
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="Diabetes 30-day readmission prediction service"
LABEL version="1.0.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker layer caching means this only
# re-runs if requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY src/       ./src/
COPY api/       ./api/
COPY configs/   ./configs/
COPY scripts/   ./scripts/

# ── Copy model artifacts ──────────────────────────────────────────────────────
# Models must be trained locally first: python scripts/train.py
# Then Docker copies the artifact into the image at build time
COPY models/    ./models/

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker will mark container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]
# ── Base image ────────────────────────────────────────────────────────────────
# Slim Python image — no GPU needed, keeps image size down
FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="Diabetes 30-day readmission prediction service"
LABEL version="1.0.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker layer caching means this only
# re-runs if requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY src/       ./src/
COPY api/       ./api/
COPY configs/   ./configs/
COPY scripts/   ./scripts/

# ── Copy model artifacts ──────────────────────────────────────────────────────
# Models must be trained locally first: python scripts/train.py
# Then Docker copies the artifact into the image at build time
COPY models/    ./models/

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker will mark container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]