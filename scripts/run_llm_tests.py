#!/usr/bin/env python3
"""
Run 10 high-quality LLM test questions against the chatbot API and save results.
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

QUESTIONS = [
    {
        "id": 1,
        "name": "No-Letter Constraint (Instruction Following)",
        "question": "Write a 50-word summary of the plot of The Matrix without using the letter 'e'.",
        "why": "Tests hard constraints; tokenization makes it difficult to track letters while maintaining grammar.",
    },
    {
        "id": 2,
        "name": "Multi-Step Counterfactual (Reasoning)",
        "question": "If Steve Jobs had decided to become a professional pastry chef in 1976 instead of starting Apple, how might the current landscape of smartphone interface design be different today? Provide three specific examples.",
        "why": "Requires understanding Jobs' influence on design and projecting a world where that influence was absent.",
    },
    {
        "id": 3,
        "name": "Logical Fallacy Trap (Critical Thinking)",
        "question": "Argue in favor of the statement: 'Since every part of this machine is light, the entire machine must be light.' Then, identify the logical fallacy you just used.",
        "why": "Tests recognition of the Fallacy of Composition; a strong model will flag why the argument is unsound.",
    },
    {
        "id": 4,
        "name": "Moral Dilemma with a Twist (Ethics & Nuance)",
        "question": "A self-driving car's brakes fail. It must choose between hitting a group of five elderly pedestrians or swerving into a wall, which will certainly kill the single passenger who is a renowned cancer researcher. Do not give me a 'it's a complex issue' answer; pick a side and justify it using Utilitarianism.",
        "why": "Forces the model out of neutrality to perform a specific philosophical analysis.",
    },
    {
        "id": 5,
        "name": "Complex Code Refactoring (Technical Skill)",
        "question": "Here is a Python function that uses nested loops to find duplicates in a list. Refactor it to O(n) time complexity, ensure it is PEP8 compliant, and add type hinting.",
        "why": "Tests optimization logic and modern professional standards (PEP8, type hints).",
    },
    {
        "id": 6,
        "name": "In-World Consistency Test (Creative Writing)",
        "question": "Describe a sunset, but you are a Victorian-era coal miner who has never seen the sky because you've lived underground your whole life. Use metaphors only related to mining.",
        "why": "Tests persona consistency; model must filter vocabulary through a limited perspective.",
    },
    {
        "id": 7,
        "name": "Zero-Shot Translation of Slang (Linguistics)",
        "question": "Translate the phrase 'That's mid, no cap, he's just chasing clout' into 18th-century Shakespearean English.",
        "why": "Requires mapping modern slang nuances into a historical dialect.",
    },
    {
        "id": 8,
        "name": "Data Synthesis Challenge (Information Density)",
        "question": "Compare the economic policies of the Roman Empire under Augustus to the United States' 'New Deal' in a 3-column table format, focusing on infrastructure, currency debasement, and social welfare.",
        "why": "Tests retrieving disparate facts and organizing into a structured comparative framework.",
    },
    {
        "id": 9,
        "name": "Recursive Prompt (Self-Awareness)",
        "question": "Write a prompt that would be difficult for an LLM like yourself to answer accurately, then explain why that prompt is difficult.",
        "why": "Reveals the model's understanding of its own limitations.",
    },
    {
        "id": 10,
        "name": "Theory of Mind Test (Social Intelligence)",
        "question": "Sally puts a ball in a red basket and leaves the room. While she is gone, Anne moves the ball to a blue box. Anne then leaves. Sally returns. Where will Sally look for the ball, and why would she be surprised if she looked in the blue box?",
        "why": "Classic False Belief task; tests whether the AI can model a character with incorrect information.",
    },
]


def call_chat(base_url: str, messages: list[dict]) -> dict:
    """POST to /api/chat and return the JSON response."""
    url = f"{base_url.rstrip('/')}/api/chat"
    body = json.dumps({"messages": messages}).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_tests(base_url: str, output_path: Optional[Path]) -> list[dict]:
    """Run all questions and return list of {question, response, tokens, error}."""
    results = []
    for q in QUESTIONS:
        print(f"  [{q['id']}/10] {q['name']}...", end=" ", flush=True)
        try:
            data = call_chat(base_url, [{"role": "user", "content": q["question"]}])
            results.append({
                "id": q["id"],
                "name": q["name"],
                "question": q["question"],
                "why": q["why"],
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
    print("Running 10 questions...\n")
    results = run_tests(args.base_url, args.output)
    write_markdown(results, args.output)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone. {len(results) - errors}/{len(results)} succeeded. Results written to {args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
