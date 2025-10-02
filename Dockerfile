# Multi-stage build for optimized Docker image
# Stage 1: Builder stage for dependencies
FROM nvcr.io/nvidia/tensorrt:25.04-py3 AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies with pip wheel for faster rebuilds
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# ============================================
# Stage 2: Runtime stage (smaller final image)
FROM nvcr.io/nvidia/tensorrt:25.04-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    git \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security with fixed UID/GID
RUN groupadd -g 10001 -r appuser && useradd -u 10001 -r -g appuser appuser \
    && mkdir -p /app /home/appuser /cache/hf \
    && chown -R appuser:appuser /app /home/appuser /cache/hf

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p static/uploads static/outputs logs \
    && chown -R appuser:appuser static logs \
    && chmod 755 static static/uploads static/outputs logs

# Switch to non-root user
USER appuser

# Health check for the application
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]