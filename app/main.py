"""
Chatbot API with OpenLLMetry tracing to Elastic.
Initialize Traceloop before any LLM client imports.
"""
import json
import logging
import re
import time
import uuid
import urllib.error
import urllib.request
from urllib.parse import urlparse
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime
from os import getenv
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

# Fallback when OPENAI_MODEL is unset (direct OpenAI / generic proxies). Hosted Elastic LiteLLM keys often use
# different model ids—set OPENAI_MODEL to an id from GET /api/llm/models (EXPOSE_LLM_MODELS=true) or curl /v1/models.
_DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from app.categories import PROMPT_CATEGORIES
from app.mcp_fetch import call_fetch_sync, mcp_fetch_enabled, url_host_allowed
from app.deepeval_score import score_agent_task_completion, score_chat_turn
from app.weather_open_meteo import fetch_current_weather


def _deepeval_scoring_enabled() -> bool:
    return getenv("DEEPEVAL_SCORE_CHAT", "").lower() in ("1", "true", "yes")


def _deepeval_agent_scoring_enabled() -> bool:
    """Task-completion judge when the agent loop invoked tools (extra LLM calls)."""
    if not _deepeval_scoring_enabled():
        return False
    return getenv("DEEPEVAL_SCORE_AGENT", "true").lower() in ("1", "true", "yes")


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    # Log on a logger Uvicorn already configures so startup lines always appear in docker logs.
    _boot = logging.getLogger("uvicorn.error")
    logging.getLogger("app").setLevel(logging.INFO)
    if _deepeval_scoring_enabled():
        agent_note = (
            " + Task Completion for agent tool calls (deepeval_task_completion)"
            if _deepeval_agent_scoring_enabled()
            else " (DEEPEVAL_SCORE_AGENT=false: skipping task-completion judge for tool runs)"
        )
        _boot.info(
            "DEEPEVAL_SCORE_CHAT enabled: Answer Relevancy (deepeval_answer_relevancy)%s",
            agent_note,
        )
    else:
        _boot.info(
            "DEEPEVAL_SCORE_CHAT is off (default). Set DEEPEVAL_SCORE_CHAT=true in .env and recreate the app container."
        )
    yield


# Support OpenAI or Ollama (OpenAI-compatible): set OPENAI_API_BASE for Ollama (e.g. http://ollama:11434/v1)
_api_base = getenv("OPENAI_API_BASE")
_client_kwargs = {"api_key": getenv("OPENAI_API_KEY", "ollama")}
if _api_base:
    _client_kwargs["base_url"] = _api_base.rstrip("/")
client = OpenAI(**_client_kwargs)

app = FastAPI(title="Chatbot with LLM Observability", lifespan=_app_lifespan)

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
    # Use-case label for tracing (prompt.category). Default Other; must be one of PROMPT_CATEGORIES.
    category: str = "Other"
    name: str | None = None  # optional; when set, tool + task personalize the prompt
    # Optional model id for this request only (e.g. test runner); otherwise OPENAI_MODEL / default.
    model: str | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _validate_prompt_category(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return "Other"
        s = str(v).strip()
        if s not in PROMPT_CATEGORIES:
            allowed = ", ".join(PROMPT_CATEGORIES)
            raise ValueError(f"Invalid category {s!r}. Allowed: {allowed}")
        return s


class ChatResponse(BaseModel):
    message: str
    role: str = "assistant"
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Hierarchical trace for the Traceflow UI (OpenLLMetry-style names; timings from this request only).
    traceflow: dict | None = None
    # DeepEval when DEEPEVAL_SCORE_CHAT=true: answer_relevancy fields + optional nested task_completion (agent/tools).
    evaluation: dict | None = None


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
    if _chatbot_tools_enabled():
        instruction += (
            " Use tools when they help: get_current_weather for weather (Open-Meteo live or stub); "
            "search_knowledge_base for Texas college facts from the demo corpus; get_current_time for "
            "local time (default timezone America/Chicago); convert_units for miles/km, °F/°C, lb/kg. "
            "If get_current_weather returns text starting with \"Current weather (Open-Meteo):\", treat "
            "that as success and summarize—do not apologize."
        )
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
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the bundled Texas colleges knowledge base (keyword RAG). "
                "Use for questions about UT, A&M, Rice, Texas universities, campuses, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. Texas A&M engineering, Rice University Houston",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time in a named IANA timezone (e.g. America/Chicago for Texas)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone id, e.g. America/Chicago, America/New_York, UTC",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": (
                "Convert a numeric amount between supported units: miles/km, fahrenheit/celsius, pounds/kg."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Numeric value to convert"},
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit: miles, km, fahrenheit, celsius, pounds, kg (common aliases accepted)",
                    },
                    "to_unit": {"type": "string", "description": "Target unit (same set)"},
                },
                "required": ["amount", "from_unit", "to_unit"],
            },
        },
    },
]

FETCH_URL_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch a public HTTP(S) URL and return page content as markdown (via official MCP fetch server). "
            "Use for live web facts when the knowledge base is insufficient. "
            "Respects MCP_FETCH_ALLOWLIST when set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch (https://...)"},
                "max_length": {
                    "type": "integer",
                    "description": "Max characters to return (default on server ~5000)",
                },
                "start_index": {
                    "type": "integer",
                    "description": "Character offset for chunked reads of long pages",
                },
                "raw": {
                    "type": "boolean",
                    "description": "If true, skip markdown conversion (raw content)",
                },
            },
            "required": ["url"],
        },
    },
}


def _weather_stub_reply(location: str, unit: str) -> str:
    """Fixed demo weather when WEATHER_PROVIDER=stub (offline / tests)."""
    location_lower = (location or "").strip().lower()
    if "texas" in location_lower or "tx" in location_lower or location_lower in (
        "austin",
        "houston",
        "dallas",
        "san antonio",
    ):
        temp_f = 88
        temp_c = 31
        conditions = "Sunny and warm"
        if unit == "celsius":
            return f"{conditions}, {temp_c}°C in Texas."
        return f"{conditions}, {temp_f}°F in Texas."
    if unit == "celsius":
        return "Clear, 22°C."
    return "Clear, 72°F."


@tool(name="get_current_weather")
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Tool: current weather for a location (Open-Meteo live, or stub if WEATHER_PROVIDER=stub)."""
    provider = (getenv("WEATHER_PROVIDER") or "open_meteo").strip().lower()
    if provider == "stub":
        u = unit if unit in ("celsius", "fahrenheit") else "fahrenheit"
        return _weather_stub_reply(location, u)
    try:
        return fetch_current_weather(location, unit)
    except Exception:
        logger.exception("get_current_weather (Open-Meteo) failed")
        return "Could not fetch current weather right now. Please try again later."


def _chatbot_tools_enabled() -> bool:
    """Tool-calling agent loop; use with capable models. Default off—tinyllama often dumps schema as text."""
    return getenv("CHATBOT_USE_TOOLS", "false").lower() == "true"


def _requests_through_litellm() -> bool:
    """True when the OpenAI client targets the local Compose LiteLLM proxy (Presidio pre_call in litellm-config)."""
    base = (getenv("OPENAI_API_BASE") or "").strip().lower()
    if not base:
        return False
    # Hosted e.g. elastic.litellm-prod.ai is LiteLLM but not our Presidio-in-docker setup
    return "litellm:4000" in base or "localhost:4000" in base or "127.0.0.1:4000" in base


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
    if fn_name == "search_knowledge_base":
        q = (kwargs.get("query") or "").strip()
        return f"query={_trace_context_snippet(q, 120)!r}"
    if fn_name == "get_current_time":
        tz = (kwargs.get("timezone") or "America/Chicago").strip()
        return f"timezone={tz!r}"
    if fn_name == "convert_units":
        return (
            f"amount={kwargs.get('amount')!r}, "
            f"from_unit={kwargs.get('from_unit')!r}, to_unit={kwargs.get('to_unit')!r}"
        )
    if fn_name == "fetch_url":
        u = (kwargs.get("url") or "").strip()
        return f"url={_trace_context_snippet(u, 120)!r}"
    return ""


def _parse_tool_arguments(raw_args) -> dict:
    """Parse tool call arguments; handle string JSON, list (e.g. model returns schema), or dict."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, list) and raw_args and isinstance(raw_args[0], dict):
        return raw_args[0]
    if isinstance(raw_args, str):
        s = raw_args.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except json.JSONDecodeError:
            pass
        # Some gateways/models return near-JSON or extra text — recover location/unit if present
        m = re.search(r'"location"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
        if m:
            loc = m.group(1).replace("\\n", " ").replace('\\"', '"').replace("\\\\", "\\")
            out = {"location": loc}
            um = re.search(r'"unit"\s*:\s*"([^"]+)"', s)
            if um and um.group(1).lower() in ("celsius", "fahrenheit"):
                out["unit"] = um.group(1).lower()
            return out
    return {}


def _looks_like_schema(obj: dict) -> bool:
    """True if obj looks like a JSON schema (properties, required) rather than actual arguments."""
    return bool(
        obj.get("properties") or obj.get("required") or obj.get("propeties")  # common model typo
        or (obj.get("type") == "object" and ("properties" in obj or "propeties" in obj))
    )


def _user_text_is_weather_related(content: str) -> bool:
    """True if the user message is plausibly asking for weather (used for location inference)."""
    c = (content or "").lower()
    return bool(
        "weather" in c
        or "temperature" in c
        or "forecast" in c
        or "austin" in c
        or "houston" in c
        or "dallas" in c
        or ("texas" in c and len(c) < 200)
    )


# Longer aliases first so "san antonio" wins over "antonio" if extended later
_TEXAS_CITY_QUALIFIED: tuple[tuple[str, str], ...] = (
    ("san antonio", "San Antonio, Texas, United States"),
    ("fort worth", "Fort Worth, Texas, United States"),
    ("el paso", "El Paso, Texas, United States"),
    ("corpus christi", "Corpus Christi, Texas, United States"),
    ("mckinney", "McKinney, Texas, United States"),
    ("brownsville", "Brownsville, Texas, United States"),
    ("amarillo", "Amarillo, Texas, United States"),
    ("garland", "Garland, Texas, United States"),
    ("irving", "Irving, Texas, United States"),
    ("laredo", "Laredo, Texas, United States"),
    ("lubbock", "Lubbock, Texas, United States"),
    ("plano", "Plano, Texas, United States"),
    ("arlington", "Arlington, Texas, United States"),
    ("frisco", "Frisco, Texas, United States"),
    ("dallas", "Dallas, Texas, United States"),
    ("houston", "Houston, Texas, United States"),
    ("austin", "Austin, Texas, United States"),
)


def _texas_city_qualified_from_user_text(text: str) -> str | None:
    """Map a major Texas city mention to an unambiguous geocoding query."""
    tl = (text or "").lower()
    for alias, qualified in _TEXAS_CITY_QUALIFIED:
        if re.search(rf"\b{re.escape(alias)}\b", tl):
            return qualified
    return None


def _extract_location_hint(text: str) -> str | None:
    """Pull a place name from natural language (e.g. 'weather in Paris' -> 'Paris')."""
    t = (text or "").strip()
    if not t:
        return None
    patterns = (
        r"(?:weather|forecast|temperature)(?:\s+is|\s+like)?\s+(?:in|for|at)\s+([^?.!\n]+)",
        r"(?:what(?:'s| is)|how(?:'s| is))\s+the\s+weather\s+(?:in|for|at)\s+([^?.!\n]+)",
        r"\b(?:in|for|at)\s+([A-Za-z0-9][^?.!\n]{0,120})",
    )
    for pat in patterns:
        m = re.search(pat, t, re.I)
        if m:
            loc = m.group(1).strip().rstrip("?.!, ")
            # Drop trailing filler like "today" / "right now"
            loc = re.sub(r"\s+(today|now|right\s+now|please)\s*$", "", loc, flags=re.I).strip()
            if len(loc) >= 2:
                return loc[:200]
    return None


def _infer_weather_location(args: dict, messages: list[dict]) -> dict:
    """When the model returns schema instead of args, infer location from the last user message."""
    out = dict(args)
    last_user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_content = (msg.get("content") or "").strip()
            break
    if last_user_content and _user_text_is_weather_related(last_user_content):
        tx_place = _texas_city_qualified_from_user_text(last_user_content)
        if tx_place:
            out["location"] = tx_place
        else:
            hint = _extract_location_hint(last_user_content)
            if hint:
                out["location"] = hint
            elif "austin" in last_user_content.lower():
                out["location"] = "Austin, Texas, United States"
            elif "texas" in last_user_content.lower() or re.search(
                r"\btx\b", last_user_content.lower()
            ):
                out["location"] = "Texas"
            else:
                out["location"] = last_user_content[:200]
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
            return _user_text_is_weather_related(msg.get("content") or "")
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
        "and try again. (Enable tools only for tool-capable cloud models.)"
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


_UNIT_CANON = {
    "miles": "miles",
    "mi": "miles",
    "mile": "miles",
    "km": "kilometers",
    "kilometer": "kilometers",
    "kilometers": "kilometers",
    "f": "fahrenheit",
    "fahrenheit": "fahrenheit",
    "c": "celsius",
    "celsius": "celsius",
    "lb": "pounds",
    "lbs": "pounds",
    "pound": "pounds",
    "pounds": "pounds",
    "kg": "kilograms",
    "kilogram": "kilograms",
    "kilograms": "kilograms",
}

_CONVERTERS = {
    ("miles", "kilometers"): lambda x: x * 1.609344,
    ("kilometers", "miles"): lambda x: x / 1.609344,
    ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
    ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
    ("pounds", "kilograms"): lambda x: x * 0.45359237,
    ("kilograms", "pounds"): lambda x: x / 0.45359237,
}


def _canonical_unit(u: str) -> str | None:
    key = re.sub(r"[^\w]", "", (u or "").strip().lower())
    return _UNIT_CANON.get(key)


@tool(name="search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Tool: search Texas colleges corpus (same keyword RAG as server-side context)."""
    q = (query or "").strip()
    if not q:
        return "No query provided."
    docs = retrieve_documents(q)
    if not docs:
        return "No matching documents in the knowledge base for that query."
    return "\n\n---\n\n".join(docs)


@tool(name="get_current_time")
def get_current_time(timezone: str = "America/Chicago") -> str:
    """Tool: current local date/time in an IANA timezone."""
    tz_name = (timezone or "America/Chicago").strip() or "America/Chicago"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return f"Unknown timezone {tz_name!r}. Use an IANA id such as America/Chicago or UTC."
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (IANA %z)")


@tool(name="convert_units")
def convert_units(amount: float | int | str, from_unit: str, to_unit: str) -> str:
    """Tool: convert between miles/km, °F/°C, lb/kg."""
    try:
        x = float(amount)
    except (TypeError, ValueError):
        return "Invalid amount: pass a number."
    a = _canonical_unit(from_unit)
    b = _canonical_unit(to_unit)
    if not a or not b:
        return f"Unknown unit(s): from_unit={from_unit!r}, to_unit={to_unit!r}. Use miles, km, fahrenheit, celsius, pounds, kg."
    if a == b:
        return f"{x:g} ({a} unchanged)"
    fn = _CONVERTERS.get((a, b))
    if not fn:
        return f"No conversion defined from {a} to {b}. Supported: miles↔km, fahrenheit↔celsius, pounds↔kg."
    y = fn(x)
    return f"{x:g} {a} = {y:.4g} {b}"


@tool(name="fetch_url")
def fetch_url(
    url: str,
    max_length: int | None = None,
    start_index: int | None = None,
    raw: bool | None = None,
) -> str:
    """Tool: fetch URL via official MCP mcp-server-fetch (stdio)."""
    if not mcp_fetch_enabled():
        return "URL fetch is disabled (set MCP_FETCH_ENABLED=true)."
    u = (url or "").strip()
    if not u:
        return "No URL provided."
    ok, reason = url_host_allowed(u)
    if not ok:
        return f"URL not allowed: {reason}"
    ml = None if max_length is None else int(max_length)
    si = None if start_index is None else int(start_index)
    rw = None if raw is None else bool(raw)
    with tracer.start_as_current_span("mcp.tool.fetch") as mspan:
        mspan.set_attribute("mcp.server", "mcp_server_fetch")
        mspan.set_attribute("mcp.tool.name", "fetch")
        try:
            parsed = urlparse(u)
            if parsed.hostname:
                mspan.set_attribute("url.host", parsed.hostname)
        except Exception:
            pass
        return call_fetch_sync(u, max_length=ml, start_index=si, raw=rw)


# Map tool names to callables for the agent loop (must follow all @tool defs above).
TOOL_FUNCTIONS: dict[str, object] = {
    "get_current_weather": get_current_weather,
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "convert_units": convert_units,
}
if mcp_fetch_enabled():
    TOOL_FUNCTIONS["fetch_url"] = fetch_url
    TOOLS.append(FETCH_URL_OPENAI_TOOL)


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
) -> tuple[str, dict, list[dict]]:
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
    reply, usage, agent_tools = generate_response(
        context, messages, trace_children=gen_children, trace_meta=gen_node
    )
    if gen_node is not None:
        gen_node["duration_ms"] = _elapsed_ms(t_gen)
    if trace_root is not None:
        trace_root["duration_ms"] = _elapsed_ms(wf_start)
    return reply, usage, agent_tools


@task(name="retrieve_context")
def retrieve_context(query: str) -> list[str]:
    """Task: retrieve documents and set span attributes."""
    span = trace.get_current_span()
    span.set_attribute("retrieval.source", "texas_colleges.json")
    context = retrieve_documents(query)
    span.set_attribute("retrieval.num_results", len(context))
    return context


MAX_AGENT_TURNS = 10


def run_agent(messages: list[dict], trace_children: list | None = None) -> tuple[str, dict, list[dict]]:
    """Agent loop: call LLM with tools; when model returns tool_calls, execute tools and re-call until final answer (e.g. weather in Texas).

    Returns ``(reply, usage, tools_invoked)`` where ``tools_invoked`` is a list of
    ``{"name", "input_parameters", "output"}`` dicts for DeepEval task-completion scoring.
    """
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
        tools_invoked: list[dict] = []
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
                return content, total_usage, tools_invoked
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
                kwargs = {
                    k.lower(): v
                    for k, v in args.items()
                    if isinstance(v, (str, int, float, type(None), bool))
                }
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
                    record_params = {"location": loc, "unit": unit}
                else:
                    result = TOOL_FUNCTIONS[name](**kwargs)
                    record_params = {k: v for k, v in kwargs.items()}
                tools_invoked.append(
                    {
                        "name": name,
                        "input_parameters": record_params,
                        "output": str(result),
                    }
                )
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
        return content or "Max agent turns reached.", total_usage, tools_invoked


@task(name="generate_response")
def generate_response(
    context: list[str],
    messages: list[dict],
    trace_children: list | None = None,
    trace_meta: dict | None = None,
) -> tuple[str, dict, list[dict]]:
    """Task: build prompt and call LLM (or run agent when tools enabled)."""
    span = trace.get_current_span()
    span.set_attribute("prompt.template", getenv("PROMPT_TEMPLATE", "rag_v2"))
    final_messages = build_prompt(context, messages)
    final_messages = _ensure_brevity_system_message(final_messages)
    span.set_attribute("prompt.num_messages", len(final_messages))
    if trace_meta is not None:
        model = _openai_model()
        rag = f"{len(context)} RAG chunk(s) in prompt" if context else "no RAG chunks"
        tools = "tools on" if _chatbot_tools_enabled() else "tools off"
        if _chatbot_tools_enabled() and mcp_fetch_enabled():
            tools += " + MCP fetch"
        base = f"model={model} · {len(final_messages)} messages · {rag} · {tools}"
        if _requests_through_litellm():
            base += (
                " · Nested below: Presidio pre_call (guardrail) runs inside local LiteLLM before each "
                "provider call—order is parent row first (this task), then children in call order."
            )
        else:
            base += " · Remote or direct LLM base URL; local presidio_pii_guardrail rows omitted."
        trace_meta["context"] = base
    if _chatbot_tools_enabled():
        return run_agent(final_messages, trace_children=trace_children)
    content, usage, _ = _chat_completion(final_messages, trace_children=trace_children)
    content = _sanitize_user_facing_reply(content, final_messages)
    return content, usage, []


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
        "model": _openai_model(),
        "messages": messages,
    }
    if _chatbot_tools_enabled():
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"
        # Default off: some gateways handle sequential tool rounds more reliably than parallel
        if getenv("CHATBOT_PARALLEL_TOOL_CALLS", "false").lower() not in ("1", "true", "yes"):
            kwargs["parallel_tool_calls"] = False
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
    model = _openai_model()
    _cc_ctx = f"model={model} · request_messages={len(messages)}"
    if _chatbot_tools_enabled():
        _cc_ctx += " · OpenAI tools=auto (weather, knowledge_base, time, convert_units"
        if mcp_fetch_enabled():
            _cc_ctx += ", fetch_url (MCP)"
        _cc_ctx += ")"

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/prompt/categories")
def api_prompt_categories():
    """Allowed values for ChatRequest.category (use-case taxonomy for OTEL prompt.category)."""
    return {"categories": list(PROMPT_CATEGORIES)}


@app.get("/api/llm/models")
def api_llm_models():
    """
    List model ids your API key can use on the configured OPENAI_API_BASE (OpenAI /v1/models shape).
    Enable with EXPOSE_LLM_MODELS=true (off by default).
    """
    if getenv("EXPOSE_LLM_MODELS", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Not found")
    return _fetch_upstream_model_catalog()


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


# Per-request chat model override (set by /api/chat when body includes ``model``).
_chat_model_override: ContextVar[str | None] = ContextVar("chat_model_override", default=None)


@contextmanager
def _chat_model_scope(model: str | None):
    token = None
    explicit = (model or "").strip() or None
    if explicit:
        token = _chat_model_override.set(explicit)
    try:
        yield
    finally:
        if token is not None:
            _chat_model_override.reset(token)


def _openai_model() -> str:
    override = _chat_model_override.get()
    if override:
        return override
    explicit = (getenv("OPENAI_MODEL") or "").strip()
    if explicit:
        return explicit
    return _DEFAULT_OPENAI_MODEL


def _upstream_error_user_hint(exc: OpenAIAPIError) -> str:
    """Extract a short provider/LiteLLM message for the chat UI (OpenAI + LiteLLM JSON shapes)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        data = body
    else:
        raw = _openai_error_detail_for_log(exc)
        if not raw:
            return ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    # LiteLLM often returns {"error": "plain string message..."}
    if isinstance(err, str) and err.strip():
        return err.strip().replace("\n", " ")[:400]
    if isinstance(err, dict):
        for k in ("message", "msg"):
            v = err.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().replace("\n", " ")[:400]
    m = data.get("message")
    if isinstance(m, str) and m.strip():
        return m.strip().replace("\n", " ")[:400]
    return ""


def _llm_models_list_urls() -> list[str]:
    """Try common OpenAI-compatible model catalog paths (LiteLLM uses /v1/models)."""
    base = (getenv("OPENAI_API_BASE") or "").strip().rstrip("/")
    if not base:
        return []
    if base.endswith("/v1"):
        return [f"{base}/models"]
    return [f"{base}/v1/models", f"{base}/models"]


def _fetch_upstream_model_catalog() -> dict:
    """GET models from the configured LLM base (Bearer OPENAI_API_KEY)."""
    key = (getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set.")
    urls = _llm_models_list_urls()
    if not urls:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_BASE is not set; cannot list models for a custom endpoint.",
        )
    last_err: str | None = None
    for url in urls:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw = resp.read().decode()
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode()[:500]
            except Exception:
                body = str(e)
            last_err = f"{url} -> HTTP {e.code} {body}"
            logger.warning("List models failed: %s", last_err)
        except Exception as e:
            last_err = f"{url} -> {e}"
            logger.warning("List models failed: %s", last_err)
    raise HTTPException(
        status_code=502,
        detail=last_err or "Could not reach the LLM /models endpoint.",
    )


def _is_likely_content_policy_block(exc: OpenAIAPIStatusError) -> bool:
    """True only when upstream body suggests moderation/content policy—not every HTTP 400."""
    d = _openai_error_detail_for_log(exc).lower()
    if not d:
        return False
    keywords = (
        "content_policy",
        "content policy",
        "content_filter",
        "moderation",
        "safety",
        "blocked by",
        "disallowed",
        "violation",
        "inappropriate",
        "personal information",
        "guardrail",
    )
    return any(k in d for k in keywords)


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
    if code == 401:
        return (
            "LLM authentication failed (HTTP 401). Check OPENAI_API_KEY and that OPENAI_API_BASE "
            "matches your provider (e.g. https://elastic.litellm-prod.ai). "
            "Set DEBUG_LLM_ERRORS=true for a short debug hint."
        )
    if code == 429:
        return "The language model is rate-limited. Please wait a moment and try again."
    if code == 503:
        return "The language model is busy. Please try again in a few seconds."
    if "presidio" in detail or ("pii" in detail and "analysis" in detail):
        return (
            "Our PII safety check is temporarily unavailable. Please try again shortly. "
            "If this continues, verify Presidio services are reachable from LiteLLM (container port 3000)."
        )
    if code == 400:
        # Most 400s are bad model, auth scope, or tools/schema—not "you said something wrong"
        hint = _upstream_error_user_hint(exc)
        base = (
            "The language model rejected the request (HTTP 400). Common causes: the model name is not enabled "
            "on this endpoint, the API key cannot access that model, or tool-calling failed. "
            "Try CHATBOT_USE_TOOLS=false for a plain answer. Set DEBUG_LLM_ERRORS=true for a short server hint."
        )
        if hint:
            base = f"{base} Details: {hint}"
        if "invalid model" in detail or "model name" in detail:
            base += (
                " To see allowed model ids: set EXPOSE_LLM_MODELS=true, restart the app, then GET /api/llm/models "
                "(or curl OPENAI_API_BASE/v1/models with Authorization: Bearer your key)."
            )
        return base
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
    prompt_category = req.category

    span = trace.get_current_span()
    span.set_attribute("user.session_id", session_id)
    span.set_attribute("conversation.turn", turn_number)
    span.set_attribute("prompt.category", prompt_category)
    span.set_attribute("chat.use_case", prompt_category)
    # Flat aliases improve discoverability in backends that normalize dotted attribute names.
    span.set_attribute("prompt_category", prompt_category)
    span.set_attribute("use_case", prompt_category)
    span.add_event("chat.request", {"prompt.category": prompt_category})

    sc = span.get_span_context()
    trace_id_hex = format(sc.trace_id, "032x") if sc.is_valid else ""
    logger.info(
        "chat request use_case=%r session_id=%s trace_id=%s turn=%s",
        prompt_category,
        session_id,
        trace_id_hex or "-",
        turn_number,
    )

    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            query = m.get("content", "")
            break

    name = req.name or None
    session_short = session_id[:8] if len(session_id) >= 8 else session_id
    trace_root: dict = {
        "name": "handle_chat",
        "kind": "WORKFLOW",
        "duration_ms": 0.0,
        "children": [],
        "context": f"use case: {prompt_category} · session={session_short}",
    }
    model_override = (req.model or "").strip() or None
    if model_override:
        span.set_attribute("llm.request.model", model_override[:256])

    with _chat_model_scope(req.model):
        try:
            reply, usage, agent_tools = handle_chat(query, messages, name=name, trace_root=trace_root)
        except OpenAIAPIStatusError as e:
            _log_llm_upstream_error(e, "chat")
            if getattr(e, "status_code", None) == 400 and _is_likely_content_policy_block(e):
                msg = (
                    "Request blocked by content policy. Please avoid sharing sensitive personal information."
                    + _debug_llm_error_suffix(e)
                )
            else:
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

        evaluation: dict | None = None
        if _deepeval_scoring_enabled():
            # Own span: after @workflow(handle_chat) returns the "current" OTEL span is unreliable for attributes
            # in some setups; a dedicated child span shows up clearly in APM (e.g. Elastic) waterfalls.
            with tracer.start_as_current_span("deepeval_answer_relevancy") as eval_span:
                evaluation = score_chat_turn(query, reply)
                if evaluation:
                    eval_span.set_attribute("deepeval.metric", "answer_relevancy")
                    if isinstance(evaluation.get("score"), (int, float)):
                        s = float(evaluation["score"])
                        eval_span.set_attribute("deepeval.score", s)
                        eval_span.set_attribute("deepeval.answer_relevancy", s)
                    if evaluation.get("skipped"):
                        eval_span.set_attribute("deepeval.skipped", True)
                    if evaluation.get("error"):
                        eval_span.set_attribute(
                            "deepeval.error", str(evaluation["error"])[:512]
                        )
                    jm = evaluation.get("judge_model")
                    if jm:
                        eval_span.set_attribute("deepeval.judge_model", str(jm)[:128])
            if evaluation:
                if isinstance(evaluation.get("score"), (int, float)):
                    logger.info("DeepEval answer_relevancy score=%s", evaluation["score"])
                elif evaluation.get("skipped"):
                    logger.info("DeepEval skipped: %s", evaluation.get("reason", ""))
                elif evaluation.get("error"):
                    logger.warning("DeepEval failed: %s", evaluation.get("error"))

            tc_eval = None
            if _deepeval_agent_scoring_enabled() and agent_tools:
                with tracer.start_as_current_span("deepeval_task_completion") as tc_span:
                    tc_eval = score_agent_task_completion(query, reply, agent_tools)
                    if tc_eval is not None:
                        tc_span.set_attribute("deepeval.metric", "task_completion")
                        if isinstance(tc_eval.get("score"), (int, float)):
                            tcs = float(tc_eval["score"])
                            tc_span.set_attribute("deepeval.score", tcs)
                            tc_span.set_attribute("deepeval.task_completion", tcs)
                        if tc_eval.get("error"):
                            tc_span.set_attribute(
                                "deepeval.error", str(tc_eval["error"])[:512]
                            )
                        jm = tc_eval.get("judge_model")
                        if jm:
                            tc_span.set_attribute("deepeval.judge_model", str(jm)[:128])
                        tc_span.set_attribute("deepeval.tools_invoked_count", len(agent_tools))
                        evaluation = {**evaluation, "task_completion": tc_eval}
                if tc_eval is not None:
                    if isinstance(tc_eval.get("score"), (int, float)):
                        logger.info("DeepEval task_completion score=%s", tc_eval["score"])
                    elif tc_eval.get("error"):
                        logger.warning("DeepEval task_completion failed: %s", tc_eval.get("error"))

        return ChatResponse(
            message=reply,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            traceflow=trace_root,
            evaluation=evaluation,
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
