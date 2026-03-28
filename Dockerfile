# Chatbot + OpenLLMetry → Elastic
# Official MCP URL fetch uses PyPI mcp-server-fetch (python -m mcp_server_fetch); see app/mcp_fetch.py and MCP_FETCH_* env.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-eval.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from deepeval.metrics import AnswerRelevancyMetric"

COPY app/ ./app/
COPY otel-collector-config.yaml ./

# Default port; override with PORT env
EXPOSE 8088

# Run the API (OTEL collector runs as separate service)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
