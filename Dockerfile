# Multi-stage build for minimal production image
# Stage 1: Build dependencies and compile packages
FROM python:3.12-slim-bullseye AS builder

# Install build dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel for faster builds
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install dependencies with optimizations
RUN pip install --no-cache-dir \
    --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
    -r requirements.txt

# Clean up pip cache and temporary files to reduce layer size
RUN pip cache purge && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete

# Stage 2: Runtime image - minimal production image
FROM python:3.12-slim-bullseye AS runtime

# Install only runtime dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgdal28 \
    libgeos-c1v5 \
    libproj19 \
    libspatialindex6 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user for security
RUN groupadd -g 1000 appgroup && \
    useradd -r -u 1000 -g appgroup -s /bin/false appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage (this is the key optimization)
COPY --from=builder /opt/venv /opt/venv

# Ensure we use the virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

# Create necessary directories with proper permissions
RUN mkdir -p logs data/outputs data/orthophotos && \
    chown -R appuser:appgroup logs data && \
    chmod -R 755 logs data

# Remove any development files and caches to minimize size
RUN find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type f -name "*.pyc" -delete && \
    find . -type f -name "*.pyo" -delete && \
    rm -rf .git .gitignore .dockerignore 2>/dev/null || true

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=20s --start-period=180s --retries=15 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
