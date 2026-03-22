# Chatbot with OpenLLMetry → Elastic

A small website with a chatbot running in Docker, instrumented with [OpenLLMetry](https://github.com/traceloop/openllmetry) so that **traces**, **tool calls**, and LLM metadata are sent to your **Elastic** cluster via the OpenTelemetry Collector and Elastic APM Server.

## Architecture

- **App**: FastAPI backend + static chat UI. Calls the **LiteLLM proxy** (default) or Ollama/OpenAI directly. Traceloop SDK traces every LLM call.
- **LiteLLM proxy**: OpenAI-compatible gateway on port 4000. Runs a **Presidio PII guardrail** (pre-call), then routes by model—default **`gpt-4o-mini`** (OpenAI API; requires **`OPENAI_API_KEY`**). **`tinyllama`** via Ollama remains available in [`litellm-config.yaml`](litellm-config.yaml) if you set `OPENAI_MODEL=tinyllama`.
- **Presidio**: Two self-hosted services. **From your host**, analyzer is on **5002** and anonymizer on **5001** (mapped to port **3000** inside each container). **LiteLLM must call** `http://presidio-analyzer:3000` and `http://presidio-anonymizer:3000` on the Docker network—not the host-mapped ports.
- **Ollama**: Optional local LLM; still in Compose so you can set `OPENAI_MODEL=tinyllama`. Pulls **`OLLAMA_MODEL`** (default `tinyllama`) on first use.
- **OpenTelemetry Collector**: Receives OTLP from the app (and optionally from LiteLLM) and forwards traces to Elastic APM Server (8.x+ supports OTLP on port 8200).
- **Elastic**: Your existing cluster with APM Server and Kibana. Traces show up under **Observability → APM → Services** as `chatbot-service` (or your `OTEL_SERVICE_NAME`).

### Demo RAG (Texas colleges)

The chat workflow runs a small **keyword retrieval** step over [`app/data/texas_colleges.json`](app/data/texas_colleges.json) (not embeddings). Relevant snippets are merged into the system prompt as **Context:** so the model can answer questions about sample Texas universities. Extend the JSON to grow the corpus; `retrieve_documents` in [`app/main.py`](app/main.py) scores token overlap plus a few phrase boosts.

## Prerequisites

- Docker and Docker Compose
- **Elastic** cluster with **APM Server 8.x+** (OTLP enabled on port 8200)
- **LLM**: default stack uses **OpenAI** (`gpt-4o-mini` through LiteLLM)—you need an **OpenAI API key**. Optional **Ollama** (`tinyllama`) for a fully local model.

## Quick start (LiteLLM + Presidio + OpenAI gpt-4o-mini)

The app calls **LiteLLM** by default; LiteLLM runs **Presidio**, then **OpenAI** for `gpt-4o-mini`. **Set a real `OPENAI_API_KEY`** in `.env` (it is passed to both the **app** and **litellm** services). The **Ollama** service still starts so you can switch to `OPENAI_MODEL=tinyllama` without editing compose.

1. **Copy env and set Elastic + OpenAI**

   ```bash
   cp .env.example .env
   # Edit .env and set:
   #   OPENAI_API_KEY=sk-...        # required for default gpt-4o-mini
   #   ELASTIC_APM_SERVER_URL=https://your-apm-host:8200
   # Optional: ELASTIC_APM_SECRET_TOKEN. OPENAI_API_BASE defaults to http://litellm:4000.
   ```

2. **Run with Docker Compose**

   ```bash
   docker compose up --build
   ```
   Presidio and LiteLLM start automatically. If you use **`tinyllama`**, the Ollama container will pull it (~600MB) on first use.

3. **Presidio guardrails (LiteLLM)**

   In [`litellm-config.yaml`](litellm-config.yaml), `presidio-pii` has **`default_on: true`**, so each prompt is run through Presidio (PII masked per `pii_entities_config`) before the model. Restart the **litellm** container after editing that file. Set `default_on: false` if you need to bypass the guardrail (e.g. local debugging).

4. **Open the app**

   - App (chat UI): [http://localhost:8088](http://localhost:8088)
   - Health: [http://localhost:8088/health](http://localhost:8088/health)

5. **View traces in Elastic**

   - Open Kibana → **Observability → APM → Services**
   - Select service `chatbot-service`
   - You’ll see transactions/spans for each chat and LLM call, including tool calls, token usage, and model name.

## Using Ollama only (no OpenAI key)

In `.env`, point at Ollama and use the tinyllama route (already in `litellm-config.yaml`):

```bash
OPENAI_API_BASE=http://litellm:4000
OPENAI_MODEL=tinyllama
OPENAI_API_KEY=ollama
CHATBOT_USE_TOOLS=false   # recommended for tinyllama
```

Or bypass LiteLLM: `OPENAI_API_BASE=http://ollama:11434/v1` (see compose comments). You can remove or stop the `ollama` service if you only use OpenAI and never `tinyllama`.

## Calling OpenAI without LiteLLM

In `.env`, clear the proxy base URL so the app talks to OpenAI directly (no Presidio on that path unless you add it elsewhere):

```bash
OPENAI_API_BASE=
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CHATBOT_USE_TOOLS=true
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_BASE` | No | LLM API base URL. Default `http://litellm:4000` (LiteLLM + Presidio → OpenAI or Ollama per model). Use `http://ollama:11434/v1` to bypass LiteLLM. Leave empty for direct OpenAI. |
| `OPENAI_API_KEY` | Yes (default stack) | **Required** for `gpt-4o-mini` (set on app and LiteLLM in Compose). Use `ollama` when using only the `tinyllama` route through LiteLLM. |
| `OPENAI_MODEL` | No | Default in Compose: **`gpt-4o-mini`**. Use **`tinyllama`** for local Ollama via LiteLLM. Must match a `model_name` in `litellm-config.yaml`. |
| `OLLAMA_MODEL` | No | Model for Ollama to pull on first start (default: `tinyllama`). |
| `PRESIDIO_ANALYZER_API_BASE` | No | Used by LiteLLM (default in compose: `http://presidio-analyzer:3000`). |
| `PRESIDIO_ANONYMIZER_API_BASE` | No | Used by LiteLLM (default in compose: `http://presidio-anonymizer:3000`). |
| `DEBUG_LLM_ERRORS` | No | If `true`, the app appends a short debug snippet to chat error messages (local dev only). |
| `ELASTIC_APM_SERVER_URL` | Yes | APM Server URL (e.g. `https://your-host:8200` or Elastic Cloud URL) |
| `ELASTIC_APM_SECRET_TOKEN` | No | APM secret token for authenticated APM Server |
| `ELASTIC_APM_INSECURE` | No | Set to `true` for self-signed or dev TLS (default: `false`) |
| `OTEL_SERVICE_NAME` | No | Service name in Kibana (default: `chatbot-service`) |
| `CHATBOT_USE_TOOLS` | No | Tool-calling agent + `agent_call` in traces (default: `true` in Docker Compose; set `false` if using tinyllama without a tool-capable model) |
| `CHATBOT_SYSTEM_INSTRUCTION` | No | Override the default system prompt (brief answers unless the user wants more). |

Replies default to **brief** answers via a built-in system instruction; set `CHATBOT_SYSTEM_INSTRUCTION` to override.

## What gets traced

- **Chat workflow**: each `/api/chat` request is a workflow; the LLM call is a child span.
- **LLM spans**: provider (OpenAI), model, prompts/completions (if not disabled), token usage, latency.
- **Tool calls**: when the model uses tools, those appear as part of the trace.
- **[TRACING.md](TRACING.md)** – Diagram of tracing capabilities: service/OTLP flow, span hierarchy (workflow → tasks → agent_call → chat_completion, tools), and env vars.

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

Set env vars (e.g. from `.env`) and point the app at a local collector. For Ollama (with `ollama serve` running locally):

```bash
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_MODEL=tinyllama
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# Run OTEL Collector locally with ELASTIC_APM_SERVER_URL set
uvicorn app.main:app --reload --port 8088
```

For OpenAI instead: set `OPENAI_API_KEY=sk-...` and leave `OPENAI_API_BASE` unset.

Then open [http://localhost:8088](http://localhost:8088).

## References

- [OpenLLMetry](https://github.com/traceloop/openllmetry) – OpenTelemetry-based LLM observability
- [Traceloop × Elastic APM](https://docs.traceloop.com/docs/openllmetry/integrations/elasticsearch-apm) – integration guide
- [Elastic APM – OpenTelemetry](https://www.elastic.co/guide/en/observability/current/apm-open-telemetry-direct.html) – OTLP intake
