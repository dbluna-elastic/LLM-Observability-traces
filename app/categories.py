"""Allowed use-case labels for prompt.category (OTEL + API validation)."""

from __future__ import annotations

# Keep in sync with product taxonomy; order is UI/API list order.
PROMPT_CATEGORIES: tuple[str, ...] = (
    "Technical",
    "Content",
    "Sales",
    "Customer Success",
    "Product",
    "Services and Consulting",
    "Marketing",
    "Legal",
    "Operations",
    "Human Capital",
    "Other",
)

_PROMPT_CATEGORY_SET = frozenset(PROMPT_CATEGORIES)


def is_allowed_prompt_category(value: str) -> bool:
    return value in _PROMPT_CATEGORY_SET
