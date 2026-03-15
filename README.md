# Chatbot with OpenLLMetry → Elastic

A small website with a chatbot running in Docker, instrumented with [OpenLLMetry](https://github.com/traceloop/openllmetry) so that **traces**, **tool calls**, and LLM metadata are sent to your **Elastic** cluster via the OpenTelemetry Collector and Elastic APM Server.

## Architecture

- **App**: FastAPI backend + static chat UI. Uses OpenAI (or compatible API) and the Traceloop SDK so every LLM call is traced.
- **OpenTelemetry Collector**: Receives OTLP from the app and forwards traces/metrics/logs to Elastic APM Server (8.x+ supports OTLP on port 8200).
- **Elastic**: Your existing cluster with APM Server and Kibana. Traces show up under **Observability → APM → Services** as `chatbot-service` (or your `OTEL_SERVICE_NAME`).

## Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Elastic cluster with **APM Server 8.x+** (OTLP enabled on port 8200)

## Quick start

1. **Copy env and set required variables**

   ```bash
   cp .env.example .env
   # Edit .env and set at least:
   #   OPENAI_API_KEY=sk-...
   #   ELASTIC_APM_SERVER_URL=https://your-apm-host:8200
   # If using Elastic Cloud or secured APM, set ELASTIC_APM_SECRET_TOKEN.
   ```

2. **Run with Docker Compose**

   ```bash
   docker compose up --build
   ```

3. **Open the app**

   - App (chat UI): [http://localhost:8000](http://localhost:8000)
   - Health: [http://localhost:8000/health](http://localhost:8000/health)

4. **View traces in Elastic**

   - Open Kibana → **Observability → APM → Services**
   - Select service `chatbot-service`
   - You’ll see transactions/spans for each chat and LLM call, including tool calls, token usage, and model name.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for the chatbot |
| `ELASTIC_APM_SERVER_URL` | Yes | APM Server URL (e.g. `https://your-host:8200` or Elastic Cloud URL) |
| `ELASTIC_APM_SECRET_TOKEN` | No | APM secret token for authenticated APM Server |
| `ELASTIC_APM_INSECURE` | No | Set to `true` for self-signed or dev TLS (default: `false`) |
| `OTEL_SERVICE_NAME` | No | Service name in Kibana (default: `chatbot-service`) |
| `OPENAI_MODEL` | No | Model name (default: `gpt-4o-mini`) |
| `CHATBOT_USE_TOOLS` | No | Enable tool-call demo for richer traces (default: `true`) |

## What gets traced

- **Chat workflow**: each `/api/chat` request is a workflow; the LLM call is a child span.
- **LLM spans**: provider (OpenAI), model, prompts/completions (if not disabled), token usage, latency.
- **Tool calls**: when the model uses tools, those appear as part of the trace.

To avoid sending prompt/completion content to Elastic (e.g. in production), set:

```bash
TRACELOOP_TRACE_CONTENT=false
```

in the `app` service in `docker-compose.yaml` or in your env.

## Local development (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set env vars (e.g. from `.env`) and point the app at a local collector:

```bash
export OPENAI_API_KEY=sk-...
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# Run OTEL Collector locally (e.g. docker run with otel-collector-config.yaml) with ELASTIC_APM_SERVER_URL set
uvicorn app.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## References

- [OpenLLMetry](https://github.com/traceloop/openllmetry) – OpenTelemetry-based LLM observability
- [Traceloop × Elastic APM](https://docs.traceloop.com/docs/openllmetry/integrations/elasticsearch-apm) – integration guide
- [Elastic APM – OpenTelemetry](https://www.elastic.co/guide/en/observability/current/apm-open-telemetry-direct.html) – OTLP intake
