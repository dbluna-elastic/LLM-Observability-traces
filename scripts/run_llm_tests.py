#!/usr/bin/env python3
"""
Run 15 tool- and edge-case test questions against the chatbot API (weather, KB, time, convert, fetch, combos).
Rotates chat models across questions (same list as the UI test runner). Assumes CHATBOT_USE_TOOLS=true;
fetch prompts need MCP_FETCH_ENABLED=true and allowlisted hosts where applicable.
Usage: python scripts/run_llm_tests.py [--base-url http://localhost:8088] [--output results.md]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODELS = [
    "llm-gateway/gpt-5-mini",
    "llm-gateway/gpt-4.1-nano",
    "llm-gateway/claude-sonnet-4-6",
    "llm-gateway/gemini-2.5-pro",
]

QUESTIONS = [
    {
        "id": 1,
        "name": "Tool: current weather (Houston)",
        "question": "What is the current weather in Houston, Texas? Use your tools and report temperature in Fahrenheit.",
        "why": "Should invoke get_current_weather; validates Open-Meteo path and Traceflow tool row.",
    },
    {
        "id": 2,
        "name": "Tool: Texas colleges knowledge base (Rice)",
        "question": "Using the Texas colleges knowledge base, what notable facts can you share about Rice University in Houston?",
        "why": "Should invoke search_knowledge_base over texas_colleges.json.",
    },
    {
        "id": 3,
        "name": "Tool: time (America/Chicago)",
        "question": "What is the current date and time in the America/Chicago timezone? Use your tool with that exact IANA timezone name.",
        "why": "Should invoke get_current_time with America/Chicago.",
    },
    {
        "id": 4,
        "name": "Tool: unit conversion (miles to km)",
        "question": "Convert exactly 100 miles to kilometers using your convert_units tool and report the numeric result.",
        "why": "Should invoke convert_units (miles to kilometers).",
    },
    {
        "id": 5,
        "name": "Tool: fetch URL (example.com)",
        "question": "Fetch https://example.com with your URL fetch tool if you have one, then give a one-sentence summary. If you cannot fetch URLs, say so clearly.",
        "why": "When MCP_FETCH_ENABLED=true and example.com is allowlisted, should invoke fetch_url and mcp.tool.fetch span.",
    },
    {
        "id": 6,
        "name": "Multi: weather Dallas + time Chicago",
        "question": "What is the current weather in Dallas, Texas in Fahrenheit, and what is the current local time in America/Chicago? Use a separate tool call for each.",
        "why": "Two tools: get_current_weather and get_current_time.",
    },
    {
        "id": 7,
        "name": "Multi: KB Texas A&M + weather College Station",
        "question": "Give one fact about Texas A&M University from your Texas colleges knowledge base, then report the current weather in College Station, Texas in Fahrenheit.",
        "why": "search_knowledge_base then get_current_weather.",
    },
    {
        "id": 8,
        "name": "Multi: convert F to C + UTC time",
        "question": "Convert 212 degrees Fahrenheit to Celsius using your tool, and tell me the current time in UTC using your time tool.",
        "why": "convert_units and get_current_time (UTC).",
    },
    {
        "id": 9,
        "name": "Multi: compare two schools (KB only)",
        "question": "Compare The University of Texas at Austin and Texas A&M University using only facts from your Texas colleges knowledge base: city or location and one academic or campus strength for each.",
        "why": "One or more search_knowledge_base calls; no weather/time required.",
    },
    {
        "id": 10,
        "name": "Multi: orchestration (weather, time, convert, KB, optional fetch)",
        "question": "In one turn, use your tools for all of: (1) current weather in San Antonio, Texas in Fahrenheit, (2) current date and time in UTC, (3) convert 50 miles to kilometers, (4) one sentence about UT Austin from the Texas colleges knowledge base. If you have a web fetch tool, also fetch https://example.com and quote one short line from the page.",
        "why": "Exercises all five tools in one user message when MCP fetch is enabled.",
    },
    {
        "id": 11,
        "name": "Edge: invalid IANA timezone",
        "question": "What is the current date and time in the America/Fakeville timezone? Use your time tool with that exact IANA timezone string.",
        "why": "Invalid zone; tool or model should error or admit the zone is invalid.",
    },
    {
        "id": 12,
        "name": "Edge: fetch non-allowlisted host",
        "question": "Fetch https://example.org/ with your URL fetch tool and summarize the page. If your policy blocks that host, say so clearly.",
        "why": "Stresses allowlist when only e.g. example.com is permitted.",
    },
    {
        "id": 13,
        "name": "Edge: nonexistent tool",
        "question": "Use your get_stock_quote tool to fetch the latest price of AAPL. If you do not have that tool, say so and do not invent a price.",
        "why": "No such tool; checks hallucination vs honest refusal.",
    },
    {
        "id": 14,
        "name": "Edge: KB stats not in corpus",
        "question": "Using only the Texas colleges knowledge base, list the exact current undergraduate enrollment numbers for UT Austin, Texas A&M, and Rice University for the current academic year.",
        "why": "Corpus unlikely to have exact enrollments; invites grounding errors.",
    },
    {
        "id": 15,
        "name": "Edge: nonsense convert_units",
        "question": "Use convert_units to convert the text hello exactly into kilograms and report the numeric result.",
        "why": "Nonsense input; stresses argument parsing or refusal.",
    },
]


def call_chat(base_url: str, messages: list[dict], model: str | None = None) -> dict:
    """POST to /api/chat and return the JSON response."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict = {"messages": messages}
    if model:
        payload["model"] = model
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_tests(base_url: str, output_path: Optional[Path]) -> list[dict]:
    """Run all questions and return list of {question, response, tokens, error}."""
    results = []
    n = len(QUESTIONS)
    for q in QUESTIONS:
        model = MODELS[(q["id"] - 1) % len(MODELS)]
        print(f"  [{q['id']}/{n}] {q['name']} ({model})...", end=" ", flush=True)
        try:
            data = call_chat(base_url, [{"role": "user", "content": q["question"]}], model=model)
            results.append({
                "id": q["id"],
                "name": q["name"],
                "question": q["question"],
                "why": q["why"],
                "model": model,
                "response": data.get("message", ""),
                "input_tokens": data.get("input_tokens"),
                "output_tokens": data.get("output_tokens"),
                "total_tokens": data.get("total_tokens"),
                "error": None,
            })
            tok = data.get("total_tokens") or (data.get("output_tokens") or "?")
            print(f"OK ({tok} tokens)")
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else str(e)
            results.append({
                "id": q["id"],
                "name": q["name"],
                "question": q["question"],
                "why": q["why"],
                "model": model,
                "response": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": f"HTTP {e.code}: {body}",
            })
            print(f"HTTP {e.code}")
        except URLError as e:
            results.append({
                "id": q["id"],
                "name": q["name"],
                "question": q["question"],
                "why": q["why"],
                "model": model,
                "response": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": str(e.reason),
            })
            print(f"Error: {e.reason}")
        except Exception as e:
            results.append({
                "id": q["id"],
                "name": q["name"],
                "question": q["question"],
                "why": q["why"],
                "model": model,
                "response": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": str(e),
            })
            print(f"Error: {e}")
    return results


def write_markdown(results: list[dict], path: Path) -> None:
    """Write results to a Markdown file."""
    with path.open("w", encoding="utf-8") as f:
        f.write("# LLM Test Results\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
        total_ok = sum(1 for r in results if r["error"] is None)
        total_tokens = sum(r["total_tokens"] or 0 for r in results)
        f.write(f"- **Passed:** {total_ok}/{len(results)}\n")
        f.write(f"- **Total tokens:** {total_tokens}\n\n---\n\n")
        for r in results:
            f.write(f"## {r['id']}. {r['name']}\n\n")
            f.write(f"**Model:** `{r.get('model', '')}`\n\n")
            f.write(f"**Why:** {r['why']}\n\n")
            f.write("**Question:**\n\n")
            f.write(f"> {r['question']}\n\n")
            if r["error"]:
                f.write(f"**Error:** `{r['error']}`\n\n")
            else:
                if r.get("total_tokens"):
                    f.write(f"**Tokens:** {r.get('input_tokens', '')} in / {r.get('output_tokens', '')} out ({r['total_tokens']} total)\n\n")
                f.write("**Response:**\n\n")
                f.write(f"{r['response']}\n\n")
            f.write("---\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM test questions against the chatbot API")
    parser.add_argument("--base-url", default="http://localhost:8088", help="Chat API base URL")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output Markdown file (default: llm-test-results-<timestamp>.md)")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"llm-test-results-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.md")
    print(f"Base URL: {args.base_url}")
    print(f"Output:   {args.output}")
    print(f"Running {len(QUESTIONS)} questions...\n")
    results = run_tests(args.base_url, args.output)
    write_markdown(results, args.output)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone. {len(results) - errors}/{len(results)} succeeded. Results written to {args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
