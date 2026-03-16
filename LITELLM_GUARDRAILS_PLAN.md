# Add LiteLLM with Pre-Call Guardrails (Option A: Presidio)

## Overview

Add LiteLLM as an OpenAI-compatible proxy in front of Ollama and configure a **pre-call guardrail using Presidio** (self-hosted, no API key) so every user prompt is checked (PII detection/masking) before being sent to the main LLM.

## Current flow

- Web UI → FastAPI app → Ollama (via `OPENAI_API_BASE`).

## Target flow

- Web UI → FastAPI app → **LiteLLM proxy** → **Presidio pre_call guardrail** → Ollama.
- App keeps using the OpenAI client; only `OPENAI_API_BASE` points at LiteLLM.

---

## 1. LiteLLM config (Option A: Presidio)

**New file: `litellm-config.yaml`** at project root (or `config/litellm.yaml`).

- **model_list**: One entry for the app’s model (e.g. `tinyllama`):
  - `model_name: tinyllama` (match `OPENAI_MODEL`)
  - `litellm_params.model: ollama/tinyllama`
  - `litellm_params.api_base: http://ollama:11434`
  - `litellm_params.api_key: "ollama"` or `none`

- **guardrails**: One Presidio guardrail, pre_call, default_on:
  - `guardrail_name: presidio-pii`
  - `litellm_params.guardrail: presidio`
  - `litellm_params.mode: pre_call`
  - `litellm_params.presidio_language: "en"`
  - Optional: `pii_entities_config` (e.g. CREDIT_CARD: MASK, EMAIL_ADDRESS: MASK)
  - Env vars for Presidio services: `PRESIDIO_ANALYZER_API_BASE`, `PRESIDIO_ANONYMIZER_API_BASE` (or equivalent per LiteLLM/Presidio docs)

---

## 2. Docker Compose

**Edit `docker-compose.yaml`:**

- **presidio-analyser**: Presidio Analyzer image, expose port (e.g. 5002).
- **presidio-anonymizer**: Presidio Anonymizer image, expose port (e.g. 5001).
- **litellm**: Image `docker.litellm.ai/berriai/litellm:main-latest`, port `4000:4000`, mount `litellm-config.yaml`, env for Presidio URLs (and OTEL; see Observability). `depends_on`: `ollama`, `presidio-analyser`, `presidio-anonymizer`.
- **app**: Set `OPENAI_API_BASE=http://litellm:4000` (instead of `http://ollama:11434/v1`). App `depends_on` should include `litellm`.

---

## 3. App and env

- **No code changes** in `app/main.py` for the happy path; `default_on: true` runs the guardrail on every request.
- **Optional**: In `_chat_completion`, catch HTTP 400 (or guardrail block) from the proxy and return a user-friendly message (e.g. “Request blocked by content policy”).
- **`.env.example`**: Document `OPENAI_API_BASE=http://litellm:4000` when using Docker, and any Presidio/LiteLLM env vars.

---

## 4. Observability

- **App trace (default):** You will see the **call from the app to LiteLLM** in your existing trace: the chat_completion span (and the instrumented HTTP client) shows the request to `http://litellm:4000`. So “LiteLLM proxy” appears as the target of the outgoing call.

- **LiteLLM proxy in the same trace:** To see **LiteLLM’s own spans** (e.g. “Received Proxy Server Request”, guardrail, call to Ollama) in that same trace:
  1. **Enable OpenTelemetry in the LiteLLM container**: Configure LiteLLM to use the OTEL callback and set `OTEL_EXPORTER_OTLP_ENDPOINT` to your **existing** OTEL collector (e.g. `http://otel-collector:4318` for HTTP). Use the same collector that the FastAPI app uses so spans go to the same backend (Elastic).
  2. **Trace propagation:** The app’s Traceloop/OpenLLMetry-instrumented OpenAI client should send the W3C `traceparent` header to LiteLLM. Ensure LiteLLM is configured to use incoming trace context as the parent for its spans (so proxy spans are children of the app’s span). LiteLLM supports this; if there is an opt-in for traceparent handling in the proxy config, enable it.
  3. **LiteLLM env in docker-compose:** Add to the `litellm` service: `OTEL_EXPORTER_OTLP_ENDPOINT`, and any other OTEL vars needed (e.g. `OTEL_EXPORTER_OTLP_PROTOCOL=http` if using HTTP). No change to the app’s Traceloop or collector config.

Result: One trace from the app through LiteLLM (and guardrail) to Ollama, visible in Elastic.

---

## 5. Files to add or edit

| Item | Action |
|------|--------|
| `litellm-config.yaml` | **Add** – model_list (Ollama) + guardrails (Presidio pre_call, default_on). |
| `docker-compose.yaml` | **Edit** – add `presidio-analyser`, `presidio-anonymizer`, `litellm`; set app `OPENAI_API_BASE=http://litellm:4000`; add OTEL env to `litellm` for Observability. |
| `.env.example` | **Edit** – document LiteLLM and Presidio-related vars. |
| `app/main.py` | **Optional** – handle guardrail block (400) and return friendly message. |
| `README.md` | **Edit** – add “LiteLLM and guardrails (Presidio)” and Observability notes. |

---

## Summary

- **Option A (Presidio)** is the chosen approach: self-hosted PII guardrail, no external API keys.
- **Observability:** App trace already shows the call to LiteLLM; enable OTEL in the LiteLLM container and point it at the same collector with trace propagation so LiteLLM proxy appears in the same trace.
