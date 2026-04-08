# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final Image
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy EVERYTHING from your root (this gets supply_chain_env, inference.py, etc.)
COPY . .

# Set Python path to ensure it finds the supply_chain_env module
ENV PYTHONPATH="/app:/app/supply_chain_env"
ENV TASK_ID="easy"
# Port MUST be 7860 for Hugging Face and Scaler Validator
ENV PORT=7860

# Health check using the correct port
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

EXPOSE 7860

# Run the FastAPI app from YOUR inference.py file
# Note: it's "inference:app" because your file is named inference.py 
# and the FastAPI object inside it is named 'app'
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
