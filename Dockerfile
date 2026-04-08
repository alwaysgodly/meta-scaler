FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY supply_chain_env/src/ ./src/
COPY supply_chain_env/tasks/ ./tasks/
COPY supply_chain_env/scripts/ ./scripts/
COPY server/ ./server/
COPY openenv.yaml .
COPY inference.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH="/app/src:/app/tasks"
ENV TASK_ID="easy"

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
