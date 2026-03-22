"""
Chatbot API with OpenLLMetry tracing to Elastic.
Initialize Traceloop before any LLM client imports.
"""
import json
import logging
import re
import time
import uuid
from os import getenv
from pathlib import Path

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

# Must init Traceloop before importing OpenAI so the client is instrumented
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, tool, workflow

# Prompt/completion capture: default on. Set TRACELOOP_TRACE_CONTENT=false to disable.
# Metrics enrichment: default on (e.g. token usage for streaming). Set TRACELOOP_ENRICH_TOKENS=false to disable.
Traceloop.init(
    app_name=getenv("OTEL_SERVICE_NAME", "chatbot-service"),
    api_endpoint=getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318"),
    disable_batch=getenv("OTEL_DISABLE_BATCH", "false").lower() == "true",
    should_enrich_metrics=getenv("TRACELOOP_ENRICH_TOKENS", "true").lower() == "true",
    resource_attributes={"deployment.environment": getenv("SERVICE_ENVIRONMENT", "chatbot")},
)

from opentelemetry import trace
from opentelemetry.propagate import inject

tracer = trace.get_tracer(getenv("OTEL_SERVICE_NAME", "chatbot-service"), "1.0.0")

from openai import OpenAI
from openai import APIError as OpenAIAPIError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APIConnectionError as OpenAIAPIConnectionError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Support OpenAI or Ollama (OpenAI-compatible): set OPENAI_API_BASE for Ollama (e.g. http://ollama:11434/v1)
_api_base = getenv("OPENAI_API_BASE")
_client_kwargs = {"api_key": getenv("OPENAI_API_KEY", "ollama")}
if _api_base:
    _client_kwargs["base_url"] = _api_base.rstrip("/")
client = OpenAI(**_client_kwargs)

app = FastAPI(title="Chatbot with LLM Observability")

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    category: str | None = None  # optional, e.g. "code_review", "general"
    name: str | None = None  # optional; when set, tool + task personalize the prompt


class ChatResponse(BaseModel):
    message: str
    role: str = "assistant"
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Hierarchical trace for the Traceflow UI (OpenLLMetry-style names; timings from this request only).
    traceflow: dict | None = None


def _elapsed_ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


# Global brevity instruction for every LLM call; override with CHATBOT_SYSTEM_INSTRUCTION in env.
_DEFAULT_CHATBOT_SYSTEM_INSTRUCTION = "Keep answers brief unless the user asks for more detail."


def _chatbot_system_instruction() -> str:
    custom = (getenv("CHATBOT_SYSTEM_INSTRUCTION") or "").strip()
    return custom if custom else _DEFAULT_CHATBOT_SYSTEM_INSTRUCTION


def _ensure_brevity_system_message(messages: list[dict]) -> list[dict]:
    """Prepend global brevity instruction: merge into first system message or insert a system message at the front."""
    instruction = _chatbot_system_instruction()
    if not messages:
        return [{"role": "system", "content": instruction}]
    out = list(messages)
    if out[0].get("role") == "system":
        existing = (out[0].get("content") or "").strip()
        merged = f"{instruction}\n\n{existing}" if existing else instruction
        out[0] = {**out[0], "role": "system", "content": merged}
        return out
    return [{"role": "system", "content": instruction}] + out


# Optional: tool definitions so tool_calls appear in traces
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location (e.g. Texas, Houston TX, Austin)",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and region, e.g. Texas, Houston TX, Austin"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                },
                "required": ["location"],
            },
        },
    },
]


@tool(name="get_current_weather")
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Tool: return current weather for a location. Stub: returns Texas weather when location is in Texas."""
    location_lower = (location or "").strip().lower()
    if "texas" in location_lower or "tx" in location_lower or location_lower in ("austin", "houston", "dallas", "san antonio"):
        temp_f = 88
        temp_c = 31
        conditions = "Sunny and warm"
        if unit == "celsius":
            return f"{conditions}, {temp_c}°C in Texas."
        return f"{conditions}, {temp_f}°F in Texas."
    if unit == "celsius":
        return "Clear, 22°C."
    return "Clear, 72°F."


# Map tool names to callables for the agent loop
TOOL_FUNCTIONS = {"get_current_weather": get_current_weather}


def _chatbot_tools_enabled() -> bool:
    """Tool-calling agent loop; use with capable models. Default off—tinyllama often dumps schema as text."""
    return getenv("CHATBOT_USE_TOOLS", "false").lower() == "true"


def _requests_through_litellm() -> bool:
    """True when the OpenAI client targets LiteLLM (Presidio guardrail runs there pre_call)."""
    base = (getenv("OPENAI_API_BASE") or "").strip().lower()
    return bool(base and "litellm" in base)


def _guardrail_trace_node() -> dict:
    """Traceflow row for Presidio PII guardrail (runs inside LiteLLM before the provider call)."""
    return {
        "name": "presidio_pii_guardrail",
        "kind": "GUARDRAIL",
        "duration_ms": 0.0,
        "context": "MASK: CREDIT_CARD, EMAIL, PHONE_NUMBER, PERSON, US_SSN (litellm-config)",
        "detail": "LiteLLM pre_call: Presidio analyzer + anonymizer (MASK per litellm-config.yaml)",
        "children": [],
    }


def _trace_context_snippet(text: str, max_len: int = 160) -> str:
    """Single-line snippet for trace UI (no secrets—user-visible chat text only)."""
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _tool_trace_context(fn_name: str, kwargs: dict) -> str:
    if fn_name == "get_current_weather":
        loc = (kwargs.get("location") or "").strip()
        unit = kwargs.get("unit") or "fahrenheit"
        return f"location={loc!r}, unit={unit}"
    return ""


def _parse_tool_arguments(raw_args) -> dict:
    """Parse tool call arguments; handle string JSON, list (e.g. model returns schema), or dict."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, list) and raw_args and isinstance(raw_args[0], dict):
        return raw_args[0]
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except json.JSONDecodeError:
            pass
    return {}


def _looks_like_schema(obj: dict) -> bool:
    """True if obj looks like a JSON schema (properties, required) rather than actual arguments."""
    return bool(
        obj.get("properties") or obj.get("required") or obj.get("propeties")  # common model typo
        or (obj.get("type") == "object" and ("properties" in obj or "propeties" in obj))
    )


def _infer_weather_location(args: dict, messages: list[dict]) -> dict:
    """When the model returns schema instead of args, infer location from the last user message."""
    out = dict(args)
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content and ("weather" in content.lower() or "austin" in content.lower() or "texas" in content.lower()):
                if "austin" in content.lower():
                    out["location"] = "Austin, Texas"
                elif "texas" in content.lower():
                    out["location"] = "Texas"
                else:
                    out["location"] = content[:200]
                break
        if msg.get("role") == "assistant":
            break
    out.setdefault("location", "Texas")
    return out


def _looks_like_raw_tool_schema_in_reply(text: str) -> bool:
    """True when the model echoed tool/function JSON instead of a natural answer (common with small models)."""
    t = (text or "").strip()
    if len(t) < 40 or "get_current_weather" not in t:
        return False
    markers = (
        '"type"',
        "'type'",
        "descripion",
        "description",
        '"function"',
        "'function'",
        '"enum"',
        "City and region",
        "fahrenheit",
        "celsius",
    )
    return sum(1 for m in markers if m in t) >= 3


def _user_asks_about_weather(messages: list[dict]) -> bool:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            c = (msg.get("content") or "").lower()
            return bool(
                "weather" in c
                or "temperature" in c
                or "forecast" in c
                or "austin" in c
                or "houston" in c
                or "dallas" in c
                or ("texas" in c and len(c) < 200)
            )
        if msg.get("role") == "assistant":
            break
    return False


def _weather_reply_from_conversation(messages: list[dict]) -> str | None:
    """Plain-text weather line when we can infer location from the chat."""
    if not _user_asks_about_weather(messages):
        return None
    args = _infer_weather_location({}, messages)
    loc = str(args.get("location") or "Texas")
    unit = args.get("unit") if isinstance(args.get("unit"), str) else "fahrenheit"
    return get_current_weather(loc, unit if unit in ("celsius", "fahrenheit") else "fahrenheit")


def _sanitize_user_facing_reply(content: str, messages: list[dict]) -> str:
    """Replace schema-dump replies with a readable sentence."""
    if not _looks_like_raw_tool_schema_in_reply(content):
        return (content or "").strip()
    weather_line = _weather_reply_from_conversation(messages)
    if weather_line:
        return weather_line
    logger.warning(
        "Model returned tool-schema-like text instead of a reply (often tinyllama with tools on). "
        "Set CHATBOT_USE_TOOLS=false or use a stronger model."
    )
    return (
        "The model replied with tool metadata instead of a normal answer—common with small local models when "
        "tool-calling is enabled. Set CHATBOT_USE_TOOLS=false in .env or docker-compose, rebuild the app, "
        "and try again. (Enable tools only for capable models like GPT-4.)"
    )


_TEXAS_COLLEGES_PATH = Path(__file__).resolve().parent / "data" / "texas_colleges.json"
_TEXAS_COLLEGES_CORPUS: list[dict] | None = None


def _load_texas_colleges_corpus() -> list[dict]:
    """Load static Texas colleges JSON (small demo RAG corpus)."""
    global _TEXAS_COLLEGES_CORPUS
    if _TEXAS_COLLEGES_CORPUS is None:
        try:
            raw = _TEXAS_COLLEGES_PATH.read_text(encoding="utf-8")
            _TEXAS_COLLEGES_CORPUS = json.loads(raw).get("documents", [])
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load Texas colleges corpus: %s", e)
            _TEXAS_COLLEGES_CORPUS = []
    return _TEXAS_COLLEGES_CORPUS


def retrieve_documents(query: str, top_k: int = 4) -> list[str]:
    """Keyword RAG over bundled Texas colleges corpus (no embeddings; demo / local use)."""
    q = (query or "").strip()
    if not q:
        return []

    corpus = _load_texas_colleges_corpus()
    if not corpus:
        return []

    q_lower = q.lower()
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", q_lower) if len(t) > 2}
    if not q_tokens:
        q_tokens = {t for t in q_lower.split() if len(t) > 1}

    def doc_text(doc: dict) -> str:
        parts = [doc.get("title", ""), doc.get("text", "")]
        parts.extend(doc.get("keywords") or [])
        return " ".join(parts).lower()

    scored: list[tuple[float, dict]] = []
    for doc in corpus:
        hay = doc_text(doc)
        title_lower = (doc.get("title") or "").lower()
        score = 0.0
        for t in q_tokens:
            score += hay.count(t)
            if t in title_lower:
                score += 3.0
        # phrase / alias boosts (keyword RAG, not embeddings)
        if ("a&m" in q_lower or "tamu" in q_lower or "aggie" in q_lower) and (
            "texas a&m" in hay or "aggies" in hay
        ):
            score += 4.0
        if "rice" in q_lower and "rice" in hay:
            score += 2.0
        if "san antonio" in q_lower and "san antonio" in hay:
            score += 2.0
        if ("ut dallas" in q_lower or "utd" in q_tokens) and "dallas" in hay:
            score += 2.0
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    picked = [d for _, d in scored[:top_k]]

    if not picked:
        # Broad Texas / college questions: return overview + a few anchors
        broad = ("college", "university", "texas", "campus", "school", "degree", "student")
        if any(b in q_lower for b in broad):
            overview = next((d for d in corpus if d.get("id") == "utexas_overview"), None)
            rest = [d for d in corpus if d.get("id") != "utexas_overview"][:2]
            picked = ([overview] if overview else []) + rest
        if not picked:
            return []

    return [f"{d['title']}: {d['text']}" for d in picked]


def build_prompt(context: list[str], messages: list[dict]) -> list[dict]:
    """Build messages for the LLM. If context is non-empty, merge it into the first system message or add one."""
    if not context:
        return messages
    context_block = "\n\n".join(context)
    context_system = {"role": "system", "content": f"Context:\n{context_block}"}
    if messages and messages[0].get("role") == "system":
        # Merge so personalization (e.g. "You are speaking to X") stays visible; put it first.
        existing = messages[0].get("content", "")
        merged = f"{existing}\n\n{context_system['content']}"
        return [{"role": "system", "content": merged}] + messages[1:]
    return [context_system] + messages


@tool(name="get_user_preferences")
def get_user_preferences(name: str) -> dict:
    """Tool: resolve the user's display name (stub). Only `name` is passed into the chat prompt."""
    return {"name": name}


@task(name="personalize_prompt")
def personalize_prompt(tool_result: dict, messages: list[dict]) -> list[dict]:
    """Task: prepend a short personalized system line (name only); global brevity rule is added in generate_response."""
    name = (tool_result.get("name") or "").strip()
    system_content = f"The user's name is {name}."
    out = [{"role": "system", "content": system_content}] + list(messages)
    # Inject name into the latest user message so the model sees it in-conversation (helps small models).
    if name:
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                content = out[i].get("content", "")
                low = content.strip().lower()
                if content and not (low.startswith("(name:") or low.startswith("(the user's name is")):
                    out[i] = {"role": "user", "content": f"(Name: {name}.)\n\n{content}"}
                break
    return out


@workflow(name="handle_chat")
def handle_chat(
    query: str,
    messages: list[dict],
    name: str | None = None,
    trace_root: dict | None = None,
) -> tuple[str, dict]:
    """Workflow: optional tool+task for name/personalization, then retrieve context and generate response."""
    wf_start = time.perf_counter()
    children: list | None = trace_root["children"] if trace_root else None

    if name and name.strip():
        nm = name.strip()
        t0 = time.perf_counter()
        tool_result = get_user_preferences(nm)
        if children is not None:
            children.append(
                {
                    "name": "get_user_preferences",
                    "kind": "TOOL",
                    "duration_ms": _elapsed_ms(t0),
                    "context": f"returns {json.dumps(tool_result, ensure_ascii=False)}",
                    "children": [],
                }
            )
        t0 = time.perf_counter()
        messages = personalize_prompt(tool_result, messages)
        if children is not None:
            children.append(
                {
                    "name": "personalize_prompt",
                    "kind": "TASK",
                    "duration_ms": _elapsed_ms(t0),
                    "context": f'Adds system: The user\'s name is {nm}. · Prefixes last user message with (Name: {nm}.)',
                    "children": [],
                }
            )

    t0 = time.perf_counter()
    context = retrieve_context(query)
    if children is not None:
        q_snip = _trace_context_snippet(query)
        n_docs = len(context)
        children.append(
            {
                "name": "retrieve_context",
                "kind": "TASK",
                "duration_ms": _elapsed_ms(t0),
                "context": f"query: {q_snip!r} · chunks: {n_docs} (Texas colleges keyword RAG)",
                "children": [],
            }
        )

    gen_children: list | None = [] if children is not None else None
    gen_node: dict | None = None
    if children is not None:
        gen_node = {"name": "generate_response", "kind": "TASK", "duration_ms": 0.0, "children": gen_children}
        children.append(gen_node)

    t_gen = time.perf_counter()
    reply, usage = generate_response(context, messages, trace_children=gen_children, trace_meta=gen_node)
    if gen_node is not None:
        gen_node["duration_ms"] = _elapsed_ms(t_gen)
    if trace_root is not None:
        trace_root["duration_ms"] = _elapsed_ms(wf_start)
    return reply, usage


@task(name="retrieve_context")
def retrieve_context(query: str) -> list[str]:
    """Task: retrieve documents and set span attributes."""
    span = trace.get_current_span()
    span.set_attribute("retrieval.source", "texas_colleges.json")
    context = retrieve_documents(query)
    span.set_attribute("retrieval.num_results", len(context))
    return context


MAX_AGENT_TURNS = 10


def run_agent(messages: list[dict], trace_children: list | None = None) -> tuple[str, dict]:
    """Agent loop: call LLM with tools; when model returns tool_calls, execute tools and re-call until final answer (e.g. weather in Texas)."""
    with tracer.start_as_current_span("agent_call") as span:
        span.set_attribute("llm.agent", True)
        agent_start = time.perf_counter()
        inner: list | None = [] if trace_children is not None else None
        agent_node: dict | None = None
        if trace_children is not None:
            agent_node = {"name": "agent_call", "kind": "AGENT", "duration_ms": 0.0, "children": inner}
            trace_children.append(agent_node)

        messages_copy = list(messages)
        total_usage: dict = {}
        content = ""
        for turn in range(MAX_AGENT_TURNS):
            content, usage, assistant_msg = _chat_completion(messages_copy, trace_children=inner)
            for k, v in (usage or {}).items():
                if v is not None and isinstance(v, (int, float)):
                    total_usage[k] = total_usage.get(k, 0) + v
            if assistant_msg is None:
                span.set_attribute("agent.turns", turn + 1)
                content = _sanitize_user_facing_reply(content, messages_copy)
                if agent_node is not None:
                    agent_node["duration_ms"] = _elapsed_ms(agent_start)
                    agent_node["context"] = f"{turn + 1} LLM round(s); tools={'on' if _chatbot_tools_enabled() else 'off'}"
                return content, total_usage
            messages_copy.append(assistant_msg)
            tool_count = 0
            for tc in assistant_msg.get("tool_calls", []):
                tool_id = tc.get("id", "")
                name = (tc.get("function") or {}).get("name", "")
                raw_args = (tc.get("function") or {}).get("arguments", "{}")
                if name not in TOOL_FUNCTIONS:
                    messages_copy.append({"role": "tool", "tool_call_id": tool_id, "content": f"Unknown tool: {name}"})
                    continue
                args = _parse_tool_arguments(raw_args)
                if name == "get_current_weather" and (not args.get("location") or _looks_like_schema(args)):
                    args = _infer_weather_location(args, messages_copy)
                kwargs = {k.lower(): v for k, v in args.items() if isinstance(v, (str, int, float, type(None)))}
                t_tool = time.perf_counter()
                if name == "get_current_weather":
                    kwargs.setdefault("location", "")
                    kwargs.setdefault("unit", "fahrenheit")
                    # Only pass location/unit—models often echo schema keys like "type" into args.
                    loc = str(kwargs.get("location", ""))
                    unit = kwargs.get("unit", "fahrenheit")
                    if unit not in ("celsius", "fahrenheit"):
                        unit = "fahrenheit"
                    result = TOOL_FUNCTIONS[name](location=loc, unit=unit)
                else:
                    result = TOOL_FUNCTIONS[name](**kwargs)
                if inner is not None:
                    tctx = _tool_trace_context(name, kwargs)
                    tool_node: dict = {
                        "name": name or "tool",
                        "kind": "TOOL",
                        "duration_ms": _elapsed_ms(t_tool),
                        "children": [],
                    }
                    if tctx:
                        tool_node["context"] = tctx
                    inner.append(tool_node)
                messages_copy.append({"role": "tool", "tool_call_id": tool_id, "content": str(result)})
                tool_count += 1
            if tool_count:
                span.add_event("tool_calls_executed", {"count": tool_count})
        span.set_attribute("agent.turns", MAX_AGENT_TURNS)
        content = _sanitize_user_facing_reply(content or "", messages_copy)
        if agent_node is not None:
            agent_node["duration_ms"] = _elapsed_ms(agent_start)
            agent_node["context"] = f"{MAX_AGENT_TURNS} rounds (max); tools on"
        return content or "Max agent turns reached.", total_usage


@task(name="generate_response")
def generate_response(
    context: list[str],
    messages: list[dict],
    trace_children: list | None = None,
    trace_meta: dict | None = None,
) -> tuple[str, dict]:
    """Task: build prompt and call LLM (or run agent when tools enabled)."""
    span = trace.get_current_span()
    span.set_attribute("prompt.template", getenv("PROMPT_TEMPLATE", "rag_v2"))
    final_messages = build_prompt(context, messages)
    final_messages = _ensure_brevity_system_message(final_messages)
    span.set_attribute("prompt.num_messages", len(final_messages))
    if trace_meta is not None:
        model = getenv("OPENAI_MODEL", "gpt-4o-mini")
        rag = f"{len(context)} RAG chunk(s) in prompt" if context else "no RAG chunks"
        tools = "tools on" if _chatbot_tools_enabled() else "tools off"
        base = f"model={model} · {len(final_messages)} messages · {rag} · {tools}"
        if _requests_through_litellm():
            base += (
                " · Nested below: Presidio pre_call (guardrail) runs inside LiteLLM before each "
                "provider call—order is parent row first (this task), then children in call order."
            )
        else:
            base += " · No LiteLLM hop; presidio_pii_guardrail rows are omitted."
        trace_meta["context"] = base
    if _chatbot_tools_enabled():
        return run_agent(final_messages, trace_children=trace_children)
    content, usage, _ = _chat_completion(final_messages, trace_children=trace_children)
    content = _sanitize_user_facing_reply(content, final_messages)
    return content, usage


def _trace_headers() -> dict[str, str]:
    """Inject W3C trace context (traceparent, tracestate) so LiteLLM attaches to the same trace."""
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def _litellm_extra_headers() -> dict[str, str] | None:
    """If PROPAGATE_TRACE_TO_LITELLM=true, return trace context so LiteLLM shares the app trace; else None (separate traces)."""
    if getenv("PROPAGATE_TRACE_TO_LITELLM", "false").lower() != "true":
        return None
    return _trace_headers()


def _message_to_dict(msg) -> dict:
    """Build a message dict for the API from a ChatCompletionMessage (including tool_calls)."""
    out: dict = {"role": msg.role, "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


@task(name="chat_completion")
def _chat_completion(
    messages: list[dict],
    trace_children: list | None = None,
) -> tuple[str, dict, dict | None]:
    """LLM call as a task span; returns (content, usage, assistant_message_or_none). When tool_calls present, third is the assistant message to append for the agent loop."""
    if trace_children is not None and _requests_through_litellm():
        trace_children.append(_guardrail_trace_node())
    start = time.perf_counter()
    kwargs: dict = {
        "model": getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "tools": TOOLS if _chatbot_tools_enabled() else None,
        "tool_choice": "auto" if _chatbot_tools_enabled() else None,
    }
    extra = _litellm_extra_headers()
    if extra:
        kwargs["extra_headers"] = extra
    response = client.chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - start) * 1000)
    span = trace.get_current_span()
    span.add_event("first_token_received", {"latency_ms": latency_ms})
    span.add_event("stream_complete", {"total_chunks": 1})

    choice = response.choices[0]
    usage = {}
    if getattr(response, "usage", None) is not None:
        u = response.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    model = getenv("OPENAI_MODEL", "gpt-4o-mini")
    _cc_ctx = f"model={model} · request_messages={len(messages)}"
    if _chatbot_tools_enabled():
        _cc_ctx += " · OpenAI tools=auto (get_current_weather)"

    if choice.message.tool_calls:
        assistant_msg = _message_to_dict(choice.message)
        content = choice.message.content or f"[Tool calls: {[t.function.name for t in choice.message.tool_calls]}]"
        if trace_children is not None:
            ntc = [tc.function.name for tc in choice.message.tool_calls]
            trace_children.append(
                {
                    "name": "chat_completion",
                    "kind": "CHAT",
                    "duration_ms": _elapsed_ms(start),
                    "context": f"{_cc_ctx} · assistant_tool_calls={ntc}",
                    "children": [],
                }
            )
        return content, usage, assistant_msg
    if trace_children is not None:
        trace_children.append(
            {
                "name": "chat_completion",
                "kind": "CHAT",
                "duration_ms": _elapsed_ms(start),
                "context": _cc_ctx,
                "children": [],
            }
        )
    return choice.message.content or "", usage, None


@tool(name="search_knowledge_base")
def search_kb(query: str) -> str:
    """Tool: search knowledge base (stub). Returns context as a single string for use in prompts or agents."""
    docs = retrieve_documents(query)
    return "\n\n".join(docs) if docs else ""


@app.get("/health")
def health():
    return {"status": "ok"}


def _openai_error_detail_for_log(exc: OpenAIAPIError) -> str:
    """Short text from OpenAI SDK error for server logs (truncate; avoid huge bodies)."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        text = getattr(resp, "text", None)
        if text:
            return text.strip()[:800]
    body = getattr(exc, "body", None)
    if body is not None:
        if isinstance(body, str):
            return body.strip()[:800]
        return str(body)[:800]
    return str(exc)[:800]


def _log_llm_upstream_error(exc: BaseException, where: str) -> None:
    if isinstance(exc, OpenAIAPIStatusError):
        code = getattr(exc, "status_code", None)
        logger.warning("LLM upstream error [%s]: HTTP %s %s", where, code, _openai_error_detail_for_log(exc))
    elif isinstance(exc, OpenAIAPIError):
        logger.warning("LLM API error [%s]: %s", where, _openai_error_detail_for_log(exc))
    elif isinstance(exc, OpenAIAPIConnectionError):
        logger.warning("LLM connection error [%s]: %s", where, exc)


def _user_message_for_status_error(exc: OpenAIAPIStatusError) -> str:
    code = getattr(exc, "status_code", None) or 0
    detail = _openai_error_detail_for_log(exc).lower()
    if code == 429:
        return "The language model is rate-limited. Please wait a moment and try again."
    if code == 503:
        return "The language model is busy. Please try again in a few seconds."
    if "presidio" in detail or ("pii" in detail and "analysis" in detail):
        return (
            "Our PII safety check is temporarily unavailable. Please try again shortly. "
            "If this continues, verify Presidio services are reachable from LiteLLM (container port 3000)."
        )
    return "The language model is temporarily unavailable. Please try again later."


def _user_message_for_api_error(exc: OpenAIAPIError) -> str:
    detail = _openai_error_detail_for_log(exc).lower()
    if "presidio" in detail or ("pii" in detail and "failed" in detail):
        return (
            "Our PII safety check is temporarily unavailable. Please try again shortly. "
            "If this continues, verify Presidio services are reachable from LiteLLM (container port 3000)."
        )
    return "The language model is temporarily unavailable. Please try again later."


def _debug_llm_error_suffix(exc: BaseException) -> str:
    if getenv("DEBUG_LLM_ERRORS", "").lower() not in ("1", "true", "yes"):
        return ""
    try:
        if isinstance(exc, OpenAIAPIStatusError):
            return f" [debug HTTP {getattr(exc, 'status_code', '?')} {_openai_error_detail_for_log(exc)[:180]}]"
        if isinstance(exc, OpenAIAPIError):
            return f" [debug {_openai_error_detail_for_log(exc)[:180]}]"
        return f" [debug {str(exc)[:180]}]"
    except Exception:
        return ""


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    turn_number = sum(1 for m in messages if m.get("role") == "user")
    session_id = request.headers.get("X-Session-ID") or str(uuid.uuid4())
    prompt_category = req.category or "general"

    span = trace.get_current_span()
    span.set_attribute("user.session_id", session_id)
    span.set_attribute("conversation.turn", turn_number)
    span.set_attribute("prompt.category", prompt_category)

    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            query = m.get("content", "")
            break

    name = req.name or None
    trace_root: dict = {
        "name": "handle_chat",
        "kind": "WORKFLOW",
        "duration_ms": 0.0,
        "children": [],
    }
    try:
        reply, usage = handle_chat(query, messages, name=name, trace_root=trace_root)
    except OpenAIAPIStatusError as e:
        if getattr(e, "status_code", None) == 400:
            return ChatResponse(
                message="Request blocked by content policy. Please avoid sharing sensitive personal information.",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
            )
        _log_llm_upstream_error(e, "chat")
        msg = _user_message_for_status_error(e) + _debug_llm_error_suffix(e)
        return ChatResponse(
            message=msg,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
    except OpenAIAPIConnectionError as e:
        _log_llm_upstream_error(e, "chat")
        return ChatResponse(
            message="Cannot reach the language model. Please try again later." + _debug_llm_error_suffix(e),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
    except OpenAIAPIError as e:
        _log_llm_upstream_error(e, "chat")
        msg = _user_message_for_api_error(e) + _debug_llm_error_suffix(e)
        return ChatResponse(
            message=msg,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
    except Exception as e:
        logger.exception("Unexpected error in chat: %s", e)
        return ChatResponse(
            message="Something went wrong. Please try again.",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )

    span = trace.get_current_span()
    span.set_attribute("response.quality_score", 0.0)  # placeholder; set from feedback when available

    return ChatResponse(
        message=reply,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        traceflow=trace_root,
    )


@app.get("/")
def index():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve AI favicon (SVG) for browsers that request /favicon.ico."""
    path = Path(__file__).resolve().parent / "static" / "favicon.svg"
    return FileResponse(path, media_type="image/svg+xml")


def main():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(getenv("PORT", "8088")),
        reload=getenv("ENV", "development") == "development",
    )


if __name__ == "__main__":
    main()
