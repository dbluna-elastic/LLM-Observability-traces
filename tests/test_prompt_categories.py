"""Prompt use-case categories (API validation + list endpoint)."""

import pytest
from pydantic import ValidationError

from app.categories import PROMPT_CATEGORIES, is_allowed_prompt_category
from app.main import ChatRequest, ChatMessage


def test_prompt_categories_count_and_other():
    assert "Other" in PROMPT_CATEGORIES
    assert "Technical" in PROMPT_CATEGORIES
    assert len(PROMPT_CATEGORIES) == 11


def test_is_allowed_prompt_category():
    assert is_allowed_prompt_category("Sales")
    assert not is_allowed_prompt_category("general")


def test_chat_request_category_defaults_to_other():
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.category == "Other"


def test_chat_request_category_accepts_valid():
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        category="Technical",
    )
    assert req.category == "Technical"


def test_chat_request_category_rejects_invalid():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            category="NotARealCategory",
        )


def test_chat_request_category_blank_becomes_other():
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        category="   ",
    )
    assert req.category == "Other"
