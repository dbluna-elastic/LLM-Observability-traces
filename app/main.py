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

from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

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


@workflow(name="chat_workflow")
def _call_llm(messages: list[dict]) -> str:
    """Single workflow span that wraps the LLM call for clearer traces."""
    return _chat_completion(messages)


@task(name="chat_completion")
def _chat_completion(messages: list[dict]) -> str:
    """LLM call as a task span; tool_calls are automatically traced by OpenLLMetry."""
    response = client.chat.completions.create(
        model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        tools=TOOLS if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true" else None,
        tool_choice="auto" if getenv("CHATBOT_USE_TOOLS", "true").lower() == "true" else None,
    )
    choice = response.choices[0]
    if choice.message.tool_calls:
        # Optional: in a full implementation you could run tool calls and re-call the model
        return choice.message.content or f"[Tool calls: {[t.function.name for t in choice.message.tool_calls]}]"
    return choice.message.content or ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply = _call_llm(messages)
    return ChatResponse(message=reply)


@app.get("/")
def index():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)


def main():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(getenv("PORT", "8000")),
        reload=getenv("ENV", "development") == "development",
    )


if __name__ == "__main__":
    main()
