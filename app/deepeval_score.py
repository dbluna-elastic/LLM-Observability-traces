"""Optional DeepEval scoring for chat turns (LLM-as-judge). Requires OPENAI_API_KEY (+ optional OPENAI_API_BASE)."""

from __future__ import annotations

import logging
import os
from os import getenv
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _api_base_hostname(api_base: str) -> str:
    u = (api_base or "").strip().rstrip("/")
    if not u:
        return ""
    if "://" not in u:
        u = "https://" + u
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""


def _elastic_hosted_litellm_base(api_base: str) -> bool:
    """True for default Elastic LiteLLM URL; short model ids need llm-gateway/ prefix."""
    host = _api_base_hostname(api_base)
    if not host:
        return False
    return "litellm-prod.ai" in host or host.endswith(".litellm.ai") or "elastic.litellm" in host


def _resolve_judge_model() -> str:
    """
    Model id for DeepEval's OpenAI-compatible judge calls.
    Hosted Elastic LiteLLM expects provider-prefixed names (e.g. llm-gateway/gpt-4o-mini); bare gpt-* ids 400.
    """
    api_base = (getenv("OPENAI_API_BASE") or "").strip()
    explicit = (getenv("DEEPEVAL_JUDGE_MODEL") or "").strip()
    from_openai = (getenv("OPENAI_MODEL") or "").strip()
    if _elastic_hosted_litellm_base(api_base):
        default = "llm-gateway/gpt-4o-mini"
    else:
        default = "gpt-4o-mini"
    raw = explicit or from_openai or default
    if "/" in raw:
        return raw
    if _elastic_hosted_litellm_base(api_base):
        return f"llm-gateway/{raw}"
    return raw


def _deepeval_openai_base_url() -> str:
    return (getenv("OPENAI_API_BASE") or "").strip().rstrip("/")


def make_gateway_judge() -> tuple[Any, str, str]:
    """Build ``GatewayOpenAIJudge`` and return (judge, resolved_model_id, base_url_str)."""
    from app.deepeval_openai_judge import GatewayOpenAIJudge

    base = _deepeval_openai_base_url()
    if base:
        os.environ["OPENAI_BASE_URL"] = base
    model = _resolve_judge_model()
    api_key = getenv("OPENAI_API_KEY", "").strip()
    judge = GatewayOpenAIJudge(
        model_id=model,
        api_key=api_key,
        base_url=base or None,
        temperature=0.0,
    )
    return judge, model, base


def _include_reason() -> bool:
    return getenv("DEEPEVAL_INCLUDE_REASON", "true").lower() in ("1", "true", "yes")


def score_chat_turn(user_query: str, assistant_reply: str) -> dict | None:
    """
    Run Answer Relevancy on the last user question vs assistant reply.
    Returns a JSON-serializable dict, or None if deepeval is not installed.
    """
    uq = (user_query or "").strip()
    ar = (assistant_reply or "").strip()
    if not uq or not ar:
        return {"skipped": True, "reason": "empty user query or assistant reply"}

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    if not (getenv("OPENAI_API_KEY") or "").strip():
        return {"error": "OPENAI_API_KEY is required for DeepEval judge model"}

    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning("deepeval is not installed; skip scoring")
        return {"error": "deepeval is not installed (image should include requirements-eval.txt)"}

    judge, model, _base = make_gateway_judge()
    include_reason = _include_reason()

    try:
        try:
            metric = AnswerRelevancyMetric(
                model=judge,
                include_reason=include_reason,
                verbose_mode=False,
                async_mode=False,
            )
        except TypeError:
            metric = AnswerRelevancyMetric(
                model=judge,
                include_reason=include_reason,
                verbose_mode=False,
            )
        tc = LLMTestCase(input=uq[:8000], actual_output=ar[:8000])
        metric.measure(tc)
        out: dict = {
            "metric": "answer_relevancy",
            "judge_model": model,
        }
        if metric.score is not None:
            out["score"] = float(metric.score)
        thr = getattr(metric, "threshold", None)
        if thr is not None:
            out["threshold"] = float(thr)
        success = getattr(metric, "success", None)
        if success is not None:
            out["success"] = bool(success)
        reason = getattr(metric, "reason", None)
        if reason:
            out["reason"] = str(reason)[:2000]
        return out
    except Exception as e:
        logger.exception("DeepEval AnswerRelevancyMetric failed")
        return {"error": str(e)[:800]}


def score_agent_task_completion(
    user_query: str,
    assistant_reply: str,
    tools_invoked: list[dict],
) -> dict | None:
    """
    DeepEval Task Completion (LLM-as-judge) using user message, final reply, and tool calls
    from the agent loop. Requires non-empty ``tools_invoked``.
    Each item: ``{"name": str, "input_parameters": dict, "output": str}``.
    """
    uq = (user_query or "").strip()
    ar = (assistant_reply or "").strip()
    if not uq or not ar or not tools_invoked:
        return None

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    if not (getenv("OPENAI_API_KEY") or "").strip():
        return {"error": "OPENAI_API_KEY is required for DeepEval judge model"}

    try:
        from deepeval.metrics import TaskCompletionMetric
        from deepeval.test_case import LLMTestCase, ToolCall
    except ImportError:
        logger.warning("deepeval is not installed; skip agent scoring")
        return {"error": "deepeval is not installed (image should include requirements-eval.txt)"}

    tool_calls: list[ToolCall] = []
    for row in tools_invoked:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        params = row.get("input_parameters")
        if not isinstance(params, dict):
            params = {}
        out = row.get("output")
        if out is None:
            out = ""
        tool_calls.append(
            ToolCall(
                name=name,
                input_parameters=params,
                output=str(out)[:8000],
            )
        )
    if not tool_calls:
        return None

    judge, model, _base = make_gateway_judge()
    include_reason = _include_reason()

    try:
        try:
            metric = TaskCompletionMetric(
                model=judge,
                include_reason=include_reason,
                async_mode=False,
                verbose_mode=False,
            )
        except TypeError:
            metric = TaskCompletionMetric(
                model=judge,
                include_reason=include_reason,
                verbose_mode=False,
            )
        tc = LLMTestCase(
            input=uq[:8000],
            actual_output=ar[:8000],
            tools_called=tool_calls,
        )
        metric.measure(tc)
        out: dict = {
            "metric": "task_completion",
            "judge_model": model,
        }
        if metric.score is not None:
            out["score"] = float(metric.score)
        thr = getattr(metric, "threshold", None)
        if thr is not None:
            out["threshold"] = float(thr)
        success = getattr(metric, "success", None)
        if success is not None:
            out["success"] = bool(success)
        reason = getattr(metric, "reason", None)
        if reason:
            out["reason"] = str(reason)[:2000]
        ug = getattr(metric, "user_goal", None)
        if ug:
            out["user_goal"] = str(ug)[:1500]
        to = getattr(metric, "task_outcome", None)
        if to:
            out["task_outcome"] = str(to)[:1500]
        return out
    except Exception as e:
        logger.exception("DeepEval TaskCompletionMetric failed")
        return {"error": str(e)[:800], "metric": "task_completion"}
