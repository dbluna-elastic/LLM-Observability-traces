# Chatbot + OpenLLMetry → Elastic
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-eval.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-eval.txt

COPY app/ ./app/
COPY otel-collector-config.yaml ./

# Default port; override with PORT env
EXPOSE 8088

# Run the API (OTEL collector runs as separate service)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
