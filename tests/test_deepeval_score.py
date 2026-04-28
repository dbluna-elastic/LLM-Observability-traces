"""Unit tests for DeepEval scoring helpers (no live judge calls)."""

import pytest

from app.deepeval_score import _resolve_judge_model, score_chat_turn


def test_score_skips_empty_query():
    r = score_chat_turn("", "hello")
    assert r is not None
    assert r.get("skipped")


def test_score_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = score_chat_turn("What is 2+2?", "Four.")
    assert r is not None
    assert "error" in r
    assert "OPENAI_API_KEY" in r["error"]


def test_resolve_judge_model_prefixes_elastic_short_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://elastic.litellm-prod.ai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert _resolve_judge_model() == "llm-gateway/gpt-4.1-mini"


def test_resolve_judge_model_respects_full_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://elastic.litellm-prod.ai")
    monkeypatch.setenv("OPENAI_MODEL", "llm-gateway/gpt-4.1-mini")
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert _resolve_judge_model() == "llm-gateway/gpt-4.1-mini"


def test_resolve_judge_model_no_prefix_for_openai_com(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert _resolve_judge_model() == "gpt-4o-mini"


def test_resolve_judge_model_default_elastic_without_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://elastic.litellm-prod.ai")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert _resolve_judge_model() == "llm-gateway/gemini-3.1-pro-preview"


def test_resolve_judge_model_default_openai_com_without_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DEEPEVAL_JUDGE_MODEL", raising=False)
    assert _resolve_judge_model() == "gpt-4.1"


def test_gateway_judge_preserves_prefixed_model_id():
    pytest.importorskip("deepeval")
    from app.deepeval_openai_judge import GatewayOpenAIJudge

    j = GatewayOpenAIJudge(
        "llm-gateway/gpt-4.1-mini",
        "sk-test",
        base_url="https://elastic.litellm-prod.ai",
    )
    assert j.get_model_name() == "llm-gateway/gpt-4.1-mini"
