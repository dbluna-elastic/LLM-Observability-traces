# Chatbot with OpenLLMetry → Elastic

A small website with a chatbot running in Docker, instrumented with [OpenLLMetry](https://github.com/traceloop/openllmetry) so that **traces**, **tool calls**, and LLM metadata are sent to your **Elastic** cluster via the OpenTelemetry Collector and Elastic APM Server.

## Architecture

- **App**: FastAPI backend + static chat UI. Calls the **LiteLLM proxy** (default) or Ollama/OpenAI directly. Traceloop SDK traces every LLM call.
- **LiteLLM proxy**: OpenAI-compatible gateway on port 4000. Runs a **Presidio PII guardrail** (pre-call) so user prompts are checked before being sent to the LLM; blocks or masks sensitive data. Forwards allowed requests to Ollama.
- **Presidio**: Two self-hosted services. **From your host**, analyzer is on **5002** and anonymizer on **5001** (mapped to port **3000** inside each container). **LiteLLM must call** `http://presidio-analyzer:3000` and `http://presidio-anonymizer:3000` on the Docker network—not the host-mapped ports.
- **Ollama**: Local LLM service; runs **tinyllama** by default and pulls it on first start.
- **OpenTelemetry Collector**: Receives OTLP from the app (and optionally from LiteLLM) and forwards traces to Elastic APM Server (8.x+ supports OTLP on port 8200).
- **Elastic**: Your existing cluster with APM Server and Kibana. Traces show up under **Observability → APM → Services** as `chatbot-service` (or your `OTEL_SERVICE_NAME`).

## Prerequisites

- Docker and Docker Compose
- **Elastic** cluster with **APM Server 8.x+** (OTLP enabled on port 8200)
- **LLM**: either **Ollama** (local, default) or an **OpenAI API key**

## Quick start (LiteLLM + Presidio + Ollama)

The stack uses **LiteLLM** as the app’s LLM endpoint by default. LiteLLM runs a **Presidio PII guardrail** (pre-call), then forwards requests to **Ollama** (tinyllama). On first run, the Ollama container pulls the model automatically.

1. **Copy env and set Elastic**

   ```bash
   cp .env.example .env
   # Edit .env and set:
   #   ELASTIC_APM_SERVER_URL=https://your-apm-host:8200
   # Optional: ELASTIC_APM_SECRET_TOKEN. OPENAI_API_BASE defaults to http://litellm:4000.
   ```

2. **Run with Docker Compose**

   ```bash
   docker compose up --build
   ```
   The first time, the Ollama service will pull `tinyllama` (~600MB). Presidio and LiteLLM start automatically.

3. **Presidio guardrails (LiteLLM)**

   In [`litellm-config.yaml`](litellm-config.yaml), `presidio-pii` has **`default_on: true`**, so each prompt is run through Presidio (PII masked per `pii_entities_config`) before Ollama. Restart the **litellm** container after editing that file. Set `default_on: false` if you need to bypass the guardrail (e.g. local debugging).

4. **Open the app**

   - App (chat UI): [http://localhost:8088](http://localhost:8088)
   - Health: [http://localhost:8088/health](http://localhost:8088/health)

5. **View traces in Elastic**

   - Open Kibana → **Observability → APM → Services**
   - Select service `chatbot-service`
   - You’ll see transactions/spans for each chat and LLM call, including tool calls, token usage, and model name.

## Using OpenAI instead of Ollama

In `.env`, clear the Ollama base URL and set your API key and model:

```bash
OPENAI_API_BASE=
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CHATBOT_USE_TOOLS=true
```

You can then remove or stop the `ollama` service in docker-compose if you don’t need it.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_BASE` | No | LLM API base URL. Default `http://litellm:4000` (LiteLLM + Presidio guardrail → Ollama). Use `http://ollama:11434/v1` to bypass LiteLLM. Leave unset for OpenAI. |
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key. Use any value (e.g. `ollama`) when using Ollama or LiteLLM. |
| `OPENAI_MODEL` | No | Model name: `tinyllama` for Ollama (default), or e.g. `gpt-4o-mini` for OpenAI. Must match a `model_name` in `litellm-config.yaml` when using LiteLLM. |
| `OLLAMA_MODEL` | No | Model for Ollama to pull on first start (default: `tinyllama`). |
| `PRESIDIO_ANALYZER_API_BASE` | No | Used by LiteLLM (default in compose: `http://presidio-analyzer:3000`). |
| `PRESIDIO_ANONYMIZER_API_BASE` | No | Used by LiteLLM (default in compose: `http://presidio-anonymizer:3000`). |
| `DEBUG_LLM_ERRORS` | No | If `true`, the app appends a short debug snippet to chat error messages (local dev only). |
| `ELASTIC_APM_SERVER_URL` | Yes | APM Server URL (e.g. `https://your-host:8200` or Elastic Cloud URL) |
| `ELASTIC_APM_SECRET_TOKEN` | No | APM secret token for authenticated APM Server |
| `ELASTIC_APM_INSECURE` | No | Set to `true` for self-signed or dev TLS (default: `false`) |
| `OTEL_SERVICE_NAME` | No | Service name in Kibana (default: `chatbot-service`) |
| `CHATBOT_USE_TOOLS` | No | Enable tool-call demo (default: `false` with Ollama; set `true` for OpenAI) |
| `CHATBOT_SYSTEM_INSTRUCTION` | No | Override the default system prompt that steers the model to short answers (2–3 sentences). |

Replies are steered to **short, concise** answers by default via a built-in system instruction; set `CHATBOT_SYSTEM_INSTRUCTION` to customize or tighten further.

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
