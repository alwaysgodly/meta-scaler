# SupplyChainEnv — OpenEnv Dockerfile
# Multi-stage build for lean production image

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -----------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY tasks/ ./tasks/
COPY scripts/ ./scripts/
COPY openenv.yaml .

# Set Python path
ENV PYTHONPATH="/app/src:/app/tasks"
ENV TASK_ID="easy"
ENV PORT=8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

EXPOSE 8000

# Run the OpenEnv server
CMD ["python", "-m", "uvicorn", "supply_chain_env.server:app", "--host", "0.0.0.0", "--port", "8000"]
