"""
Chatbot API with OpenLLMetry tracing to Elastic.
Initialize Traceloop before any LLM client imports.
"""
import json
import logging
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


# Global brevity instruction for every LLM call; override with CHATBOT_SYSTEM_INSTRUCTION in env.
_DEFAULT_CHATBOT_SYSTEM_INSTRUCTION = (
    "Answer in at most 2-3 short sentences unless the user asks for more detail. No long lists or essays."
)


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


def retrieve_documents(query: str) -> list[str]:
    """Stub: return empty list. Replace with real Elasticsearch/retrieval later."""
    return []


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
    """Tool: return user preferences for the given name (stub). Returns dict with 'name' and 'preferences'."""
    return {
        "name": name,
        "preferences": "Preferred language: English. Interests: general.",
    }


@task(name="personalize_prompt")
def personalize_prompt(tool_result: dict, messages: list[dict]) -> list[dict]:
    """Task: prepend a short personalized system line (name); global brevity rule is added in generate_response."""
    name = (tool_result.get("name") or "").strip()
    preferences = tool_result.get("preferences", "")
    system_content = f"User's name: {name}. Use their name when relevant. Preferences: {preferences}"
    out = [{"role": "system", "content": system_content}] + list(messages)
    # Inject name into the latest user message so the model sees it in-conversation (helps small models).
    if name:
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                content = out[i].get("content", "")
                if content and not content.strip().lower().startswith("(the user's name is"):
                    out[i] = {"role": "user", "content": f"(The user's name is {name}.)\n\n{content}"}
                break
    return out


@workflow(name="handle_chat")
def handle_chat(query: str, messages: list[dict], name: str | None = None) -> tuple[str, dict]:
    """Workflow: optional tool+task for name/personalization, then retrieve context and generate response."""
    if name and name.strip():
        tool_result = get_user_preferences(name.strip())
        messages = personalize_prompt(tool_result, messages)
    context = retrieve_context(query)
    return generate_response(context, messages)


@task(name="retrieve_context")
def retrieve_context(query: str) -> list[str]:
    """Task: retrieve documents and set span attributes."""
    span = trace.get_current_span()
    span.set_attribute("retrieval.source", "elasticsearch")
    context = retrieve_documents(query)
    span.set_attribute("retrieval.num_results", len(context))
    return context


MAX_AGENT_TURNS = 10


def run_agent(messages: list[dict]) -> tuple[str, dict]:
    """Agent loop: call LLM with tools; when model returns tool_calls, execute tools and re-call until final answer (e.g. weather in Texas)."""
    with tracer.start_as_current_span("agent_call") as span:
        span.set_attribute("llm.agent", True)
        messages_copy = list(messages)
        total_usage: dict = {}
        content = ""
        for turn in range(MAX_AGENT_TURNS):
            content, usage, assistant_msg = _chat_completion(messages_copy)
            for k, v in (usage or {}).items():
                if v is not None and isinstance(v, (int, float)):
                    total_usage[k] = total_usage.get(k, 0) + v
            if assistant_msg is None:
                span.set_attribute("agent.turns", turn + 1)
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
                if name == "get_current_weather":
                    kwargs.setdefault("location", "")
                    kwargs.setdefault("unit", "fahrenheit")
                result = TOOL_FUNCTIONS[name](**kwargs)
                messages_copy.append({"role": "tool", "tool_call_id": tool_id, "content": str(result)})
                tool_count += 1
            if tool_count:
                span.add_event("tool_calls_executed", {"count": tool_count})
        span.set_attribute("agent.turns", MAX_AGENT_TURNS)
        return content or "Max agent turns reached.", total_usage


@task(name="generate_response")
def generate_response(context: list[str], messages: list[dict]) -> tuple[str, dict]:
    """Task: build prompt and call LLM (or run agent when tools enabled)."""
    span = trace.get_current_span()
    span.set_attribute("prompt.template", getenv("PROMPT_TEMPLATE", "rag_v2"))
    final_messages = build_prompt(context, messages)
    final_messages = _ensure_brevity_system_message(final_messages)
    span.set_attribute("prompt.num_messages", len(final_messages))
    if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true":
        return run_agent(final_messages)
    content, usage, _ = _chat_completion(final_messages)
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
def _chat_completion(messages: list[dict]) -> tuple[str, dict, dict | None]:
    """LLM call as a task span; returns (content, usage, assistant_message_or_none). When tool_calls present, third is the assistant message to append for the agent loop."""
    start = time.perf_counter()
    kwargs: dict = {
        "model": getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "tools": TOOLS if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true" else None,
        "tool_choice": "auto" if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true" else None,
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
    if choice.message.tool_calls:
        assistant_msg = _message_to_dict(choice.message)
        content = choice.message.content or f"[Tool calls: {[t.function.name for t in choice.message.tool_calls]}]"
        return content, usage, assistant_msg
    return choice.message.content or "", usage, None


@tool(name="search_knowledge_base")
def search_kb(query: str) -> str:
    """Tool: search knowledge base (stub). Returns context as a single string for use in prompts or agents."""
    docs = retrieve_documents(query)
    return "\n\n".join(docs) if docs else ""


@app.get("/health")
def health():
    return {"status": "ok"}


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
    try:
        reply, usage = handle_chat(query, messages, name=name)
    except OpenAIAPIStatusError as e:
        if getattr(e, "status_code", None) == 400:
            return ChatResponse(
                message="Request blocked by content policy. Please avoid sharing sensitive personal information.",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
            )
        return ChatResponse(
            message="The language model is temporarily unavailable. Please try again later.",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
    except OpenAIAPIConnectionError:
        return ChatResponse(
            message="Cannot reach the language model. Please try again later.",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
    except OpenAIAPIError:
        return ChatResponse(
            message="The language model is temporarily unavailable. Please try again later.",
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
    )


@app.get("/")
def index():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)


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
