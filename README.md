# Chatbot with OpenLLMetry → Elastic

A small website with a chatbot running in Docker, instrumented with [OpenLLMetry](https://github.com/traceloop/openllmetry) so that **traces**, **tool calls**, and LLM metadata are sent to your **Elastic** cluster via the OpenTelemetry Collector and Elastic APM Server.

## Architecture

- **App**: FastAPI backend + static chat UI. OpenAI-compatible client (`OPENAI_API_BASE` / `OPENAI_MODEL`). Traceloop SDK traces every LLM call.
- **Elastic LiteLLM (hosted)**: Default chat path; `OPENAI_API_BASE` defaults to `https://elastic.litellm-prod.ai` (host only, per Elastic’s OpenAI client example) with **`OPENAI_API_KEY`** from your deployment.
- **Local LiteLLM** (port 4000): Optional; **Presidio** + [`litellm-config.yaml`](litellm-config.yaml) for **`tinyllama`** (Ollama) or **`gpt-4.1-mini`** with an API key on the **litellm** service.
- **Presidio**: Used with **local** LiteLLM only. Host ports **5002** / **5001** map to container **3000**; Docker DNS `presidio-analyzer:3000` / `presidio-anonymizer:3000`.
- **Ollama**: Local LLM for **`tinyllama`** through local LiteLLM; pulls **`OLLAMA_MODEL`** (~600MB) on first use.
- **OpenTelemetry Collector**: Receives OTLP from the app (and optionally from LiteLLM) and forwards traces to Elastic APM Server (8.x+ supports OTLP on port 8200).
- **Elastic**: Your existing cluster with APM Server and Kibana. Traces show up under **Observability → APM → Services** as `chatbot-service` (or your `OTEL_SERVICE_NAME`).

### Demo RAG (Texas colleges)

The chat workflow runs a small **keyword retrieval** step over [`app/data/texas_colleges.json`](app/data/texas_colleges.json) (not embeddings). Relevant snippets are merged into the system prompt as **Context:** so the model can answer questions about sample Texas universities. Extend the JSON to grow the corpus; `retrieve_documents` in [`app/main.py`](app/main.py) scores token overlap plus a few phrase boosts.

## Prerequisites

- Docker and Docker Compose
- **Elastic** cluster with **APM Server 8.x+** (OTLP enabled on port 8200)
- **LLM**: default Compose targets **Elastic LiteLLM prod**; you need a valid **`OPENAI_API_KEY`**. **`OPENAI_MODEL` must be the full `id` from `/v1/models`** (Elastic’s gateway often uses a prefix, e.g. **`llm-gateway/gpt-4.1-mini`**—not bare `gpt-4.1-mini`). List ids: set **`EXPOSE_LLM_MODELS=true`**, restart, then **GET [http://localhost:8088/api/llm/models](http://localhost:8088/api/llm/models)**, or `curl -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_API_BASE/v1/models"`. For fully **local** Ollama, switch **`OPENAI_API_BASE`** and **`OPENAI_MODEL`** (see `.env.example`).

## Quick start (Elastic LiteLLM + traces)

1. **Copy env and set keys**

   ```bash
   cp .env.example .env
   # Edit .env and set:
   #   OPENAI_API_KEY=<your Elastic LiteLLM / provider key>
   #   ELASTIC_APM_SERVER_URL=https://your-apm-host:8200
   # Defaults: OPENAI_API_BASE=https://elastic.litellm-prod.ai, OPENAI_MODEL=llm-gateway/gpt-4.1-mini (override using /v1/models ids)
   ```

2. **Run with Docker Compose**

   ```bash
   docker compose up --build
   ```
   The **app** does not wait on the local **litellm** container. Optional services (local LiteLLM, Presidio, Ollama) still start if you use the full file.

3. **Presidio (local LiteLLM only)**

   When **`OPENAI_API_BASE=http://litellm:4000`**, [`litellm-config.yaml`](litellm-config.yaml) runs **Presidio** `pre_call` on each request. Restart **litellm** after edits.

4. **Open the app**

   - App (chat UI): [http://localhost:8088](http://localhost:8088)
   - Health: [http://localhost:8088/health](http://localhost:8088/health)
   - **Run test questions** (in the UI) runs **15** prompts: core tool coverage plus five edge-case prompts (invalid timezone, non-allowlisted fetch, fake tool name, KB overreach, nonsense conversion). Each prompt uses a **rotating** chat model id (`llm-gateway/gpt-5-mini`, `gpt-4.1-nano`, `claude-sonnet-4-6`, `gemini-2.5-pro` in cycle) via the optional **`model`** field on **`POST /api/chat`**. Set **`CHATBOT_USE_TOOLS=true`**. For URL fetch prompts to call **`fetch_url`**, set **`MCP_FETCH_ENABLED=true`** and include **`example.com`** (or your target host) in **`MCP_FETCH_ALLOWLIST`**. You can also run **`python scripts/run_llm_tests.py`** against the same API (same questions and rotation).

5. **View traces in Elastic**

   - Open Kibana → **Observability → APM → Services**
   - Select service `chatbot-service`
   - You’ll see transactions/spans for each chat and LLM call, including tool calls, token usage, and model name.

## Local LiteLLM + Ollama (tinyllama)

```bash
OPENAI_API_BASE=http://litellm:4000
OPENAI_MODEL=tinyllama
OPENAI_API_KEY=ollama
CHATBOT_USE_TOOLS=false
```

Restart: `docker compose up -d --force-recreate app`. Ensure the **litellm** service is healthy before chatting.

## Local LiteLLM + OpenAI-backed model

```bash
OPENAI_API_BASE=http://litellm:4000
OPENAI_MODEL=gpt-4.1-mini   # must match `model_name` in litellm-config.yaml
OPENAI_API_KEY=sk-...       # also used by the litellm container in compose
CHATBOT_USE_TOOLS=true
```

## Other OpenAI-compatible bases

- **Elastic LiteLLM / LiteLLM proxy (typical)**: use the **host only** (e.g. `https://elastic.litellm-prod.ai` or `http://litellm:4000`) — same as the official OpenAI Python `base_url` examples.
- **Ollama**: use **`http://ollama:11434/v1`** (Ollama exposes the OpenAI API under `/v1/...`).
- **api.openai.com**: leave `OPENAI_API_BASE` unset; the client uses the default OpenAI URL.

Example direct OpenAI:

```bash
OPENAI_API_BASE=
OPENAI_API_KEY=sk-...
OPENAI_MODEL=llm-gateway/gpt-4.1-mini
CHATBOT_USE_TOOLS=true
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_BASE` | No | Default **`https://elastic.litellm-prod.ai`** (no `/v1`). For local Presidio + proxy use `http://litellm:4000`. Use `http://ollama:11434/v1` for Ollama only. Leave empty for default OpenAI API. |
| `OPENAI_API_KEY` | Yes (default stack) | Required for Elastic LiteLLM prod / cloud models. Use **`ollama`** only for local **`tinyllama`** via `http://litellm:4000`. Also passed to **litellm** service for local OpenAI-backed routes. |
| `OPENAI_MODEL` | No | Default in Compose: **`llm-gateway/gpt-4.1-mini`** (Elastic-style prefixed id). Use any **`id` from `/v1/models`**. Use **`tinyllama`** only with local `http://litellm:4000` + Ollama. Must match a `model_name` in `litellm-config.yaml` when using **local** LiteLLM. |
| `EXPOSE_LLM_MODELS` | No | If **`true`**, enables **GET `/api/llm/models`** to list ids for your key (default **`false`**). |
| `OLLAMA_MODEL` | No | Model for Ollama to pull on first start (default: `tinyllama`). |
| `PRESIDIO_ANALYZER_API_BASE` | No | Used by LiteLLM (default in compose: `http://presidio-analyzer:3000`). |
| `PRESIDIO_ANONYMIZER_API_BASE` | No | Used by LiteLLM (default in compose: `http://presidio-anonymizer:3000`). |
| `DEBUG_LLM_ERRORS` | No | If `true`, the app appends a short debug snippet to chat error messages (local dev only). |
| `ELASTIC_APM_SERVER_URL` | Yes | APM Server URL (e.g. `https://your-host:8200` or Elastic Cloud URL) |
| `ELASTIC_APM_SECRET_TOKEN` | No | APM secret token for authenticated APM Server |
| `ELASTIC_APM_INSECURE` | No | Set to `true` for self-signed or dev TLS (default: `false`) |
| `OTEL_SERVICE_NAME` | No | Service name in Kibana (default: `chatbot-service`) |
| `CHATBOT_USE_TOOLS` | No | Tool-calling agent + `agent_call` in traces (default **`false`**; set `true` for tool-capable cloud models) |
| `MCP_FETCH_ENABLED` | No | If **`true`**, registers **`fetch_url`** (official **mcp-server-fetch** over stdio). Requires **`CHATBOT_USE_TOOLS=true`** and outbound HTTPS from the app. Default **`false`**. |
| `MCP_FETCH_ALLOWLIST` | No | Comma-separated allowed URL hosts when set (empty = allow all — **SSRF risk**). Patterns: `example.com`, `*.wikipedia.org`. |
| `MCP_FETCH_TIMEOUT_SEC` | No | Per MCP session/tool timeout in seconds (default **`60`**, clamped 5–300). |
| `MCP_FETCH_RETRIES` | No | Retries on transport/init failures (default **`3`**, max 5). |
| `MCP_FETCH_COMMAND` | No | Executable to run the fetch server (default **`python`**). |
| `MCP_FETCH_ARGV` | No | Comma-separated argv after command (default **`-m,mcp_server_fetch`**). |
| `CHATBOT_PARALLEL_TOOL_CALLS` | No | If **`true`**, omit `parallel_tool_calls=false` on chat requests (OpenAI-style parallel tools). Default **`false`** (sequential tool rounds; more reliable on some LLM gateways). |
| `WEATHER_PROVIDER` | No | **`open_meteo`** (default): `get_current_weather` uses [Open-Meteo](https://open-meteo.com/) (outbound HTTPS, no key). Allow **`geocoding-api.open-meteo.com`** and **`api.open-meteo.com`** from the app container. **`stub`**: fixed demo temperatures (offline / CI). |
| `DEEPEVAL_SCORE_CHAT` | No | If **`true`**, runs [DeepEval](https://github.com/confident-ai/deepeval) **Answer Relevancy** on each successful `/api/chat` turn and returns scores in **`evaluation`** (extra latency + judge LLM cost via **`OPENAI_API_KEY`**). Default **`false`**. |
| `DEEPEVAL_JUDGE_MODEL` | No | Model id for DeepEval judges only (docker default **`llm-gateway/gemini-3.1-pro-preview`**; falls back to **`OPENAI_MODEL`**, then **`llm-gateway/gemini-3.1-pro-preview`** on hosted Elastic LiteLLM with no model set, or **`gpt-4.1`** on api.openai.com). Chat uses **`OPENAI_MODEL`** unless **`POST /api/chat`** JSON includes **`model`**. |
| `DEEPEVAL_INCLUDE_REASON` | No | If **`true`** (default), include a short textual reason in **`evaluation`**. |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | No | Set **`YES`** to opt out of DeepEval telemetry (default in Compose). |
| `CHATBOT_SYSTEM_INSTRUCTION` | No | Override the default system prompt (brief answers unless the user wants more). |

Replies default to **brief** answers via a built-in system instruction; set `CHATBOT_SYSTEM_INSTRUCTION` to override.

### DeepEval (optional)

The app image installs **DeepEval** via [`requirements.txt`](requirements.txt) (`deepeval>=2.5.0,<3`, plus explicit **`google-genai`** and **`posthog`** so `deepeval.metrics` imports succeed in a slim image). [`requirements-eval.txt`](requirements-eval.txt) is a compatibility shim (`-r requirements.txt`). With **`DEEPEVAL_SCORE_CHAT=true`**, each successful chat response includes an **`evaluation`** object (e.g. **`score`**, **`reason`**, **`judge_model`**) from the answer-relevancy metric. Local tests: `pip install -r requirements-dev.txt` then **`pytest tests/`** (tests do not call a live judge).

## What gets traced

- **Chat workflow**: each `/api/chat` request is a workflow; the LLM call is a child span. Optional **`category`** on the JSON body is validated against a fixed use-case list (default **`Other`**) and exported as OTEL **`prompt.category`**. Optional **`model`** overrides **`OPENAI_MODEL`** for that request only (demo / test runner; any caller with API access can use models your key allows). **`GET /api/prompt/categories`** returns the allowed labels for UIs.
- **LLM spans**: provider (OpenAI), model, prompts/completions (if not disabled), token usage, latency.
- **Tool calls**: when the model uses tools, those appear as part of the trace. The **`get_current_weather`** tool uses **Open-Meteo** for live conditions when **`WEATHER_PROVIDER=open_meteo`** (default); the app needs outbound HTTPS to **`geocoding-api.open-meteo.com`** and **`api.open-meteo.com`**. Use **`WEATHER_PROVIDER=stub`** for the previous fixed demo replies without calling the network.
- **MCP fetch** (`MCP_FETCH_ENABLED=true`): adds **`fetch_url`** backed by the official **`mcp-server-fetch`** subprocess. Traceloop still emits a **`fetch_url`** tool span; the app also records an OpenTelemetry span **`mcp.tool.fetch`** with **`mcp.server`**, **`mcp.tool.name`**, and **`url.host`**. Prefer **`MCP_FETCH_ALLOWLIST`** outside local demos.
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
# DeepEval is included above; use pip install -r requirements-dev.txt for pytest.
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
