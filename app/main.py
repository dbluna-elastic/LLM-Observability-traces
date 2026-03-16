"""
Chatbot API with OpenLLMetry tracing to Elastic.
Initialize Traceloop before any LLM client imports.
"""
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Must init Traceloop before importing OpenAI so the client is instrumented
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(
    app_name=getenv("OTEL_SERVICE_NAME", "chatbot-service"),
    api_endpoint=getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318"),
    disable_batch=getenv("OTEL_DISABLE_BATCH", "false").lower() == "true",
)

from opentelemetry import trace

tracer = trace.get_tracer(getenv("OTEL_SERVICE_NAME", "chatbot-service"), "1.0.0")

from openai import OpenAI
from fastapi import FastAPI
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


class ChatResponse(BaseModel):
    message: str
    role: str = "assistant"
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


# Optional: tool definitions so tool_calls appear in traces
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                },
                "required": ["location"],
            },
        },
    },
]


def retrieve_documents(query: str) -> list[str]:
    """Stub: return empty list. Replace with real Elasticsearch/retrieval later."""
    return []


def build_prompt(context: list[str], messages: list[dict]) -> list[dict]:
    """Build messages for the LLM. If context is non-empty, prepend a system message with it."""
    if not context:
        return messages
    context_block = "\n\n".join(context)
    return [{"role": "system", "content": f"Context:\n{context_block}"}] + messages


@workflow(name="chat_workflow")
def _call_llm(messages: list[dict]) -> tuple[str, dict]:
    """Single workflow span that wraps the LLM call for clearer traces. Returns (content, usage)."""
    return _chat_completion(messages)


@task(name="chat_completion")
def _chat_completion(messages: list[dict]) -> tuple[str, dict]:
    """LLM call as a task span; tool_calls are automatically traced by OpenLLMetry. Returns (content, usage)."""
    response = client.chat.completions.create(
        model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        tools=TOOLS if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true" else None,
        tool_choice="auto" if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true" else None,
    )
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
        content = choice.message.content or f"[Tool calls: {[t.function.name for t in choice.message.tool_calls]}]"
        return content, usage
    return choice.message.content or "", usage


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            query = m.get("content", "")
            break

    with tracer.start_as_current_span("retrieve_context") as span:
        span.set_attribute("retrieval.source", "elasticsearch")
        context = retrieve_documents(query)
        span.set_attribute("retrieval.num_results", len(context))

    with tracer.start_as_current_span("build_prompt") as span:
        span.set_attribute("prompt.template", getenv("PROMPT_TEMPLATE", "rag_v2"))
        final_messages = build_prompt(context, messages)
        span.set_attribute("prompt.num_messages", len(final_messages))

    reply, usage = _call_llm(final_messages)
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
