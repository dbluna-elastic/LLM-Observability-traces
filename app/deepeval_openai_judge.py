"""OpenAI-compatible LLM wrapper for DeepEval that preserves full gateway model ids.

DeepEval's built-in GPTModel runs parse_model_name(), which strips the segment before
the first "/". Hosted Elastic LiteLLM requires ids like llm-gateway/gpt-4.1-mini; the
stripped form is rejected with HTTP 400.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM

logger = logging.getLogger(__name__)


class GatewayOpenAIJudge(DeepEvalBaseLLM):
    """DeepEval judge LLM: sends ``model`` exactly as given (no prefix stripping)."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        # Do not call DeepEvalBaseLLM.__init__: it assigns model_name = parse_model_name(...)
        self._api_model_id = (model_id or "").strip()
        self._api_key = api_key
        self._base_url = base_url.strip().rstrip("/") if base_url else None
        self._temperature = temperature
        self.model_name = self._api_model_id
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._api_model_id

    def load_model(self, async_mode: bool = False, *args: Any, **kwargs: Any):
        if async_mode:
            if self._async_client is None:
                kw: dict = {"api_key": self._api_key}
                if self._base_url:
                    kw["base_url"] = self._base_url
                self._async_client = AsyncOpenAI(**kw)
            return self._async_client
        if self._sync_client is None:
            kw = {"api_key": self._api_key}
            if self._base_url:
                kw["base_url"] = self._base_url
            self._sync_client = OpenAI(**kw)
        return self._sync_client

    def _extract_prompt_and_schema(
        self, args: tuple, kwargs: dict
    ) -> tuple[str | None, Type[BaseModel] | None]:
        schema = kwargs.pop("schema", None)
        prompt = kwargs.pop("prompt", None)
        if args:
            if prompt is None and isinstance(args[0], str):
                prompt = args[0]
            if schema is None and len(args) > 1:
                schema = args[1]
        return prompt, schema

    def _structured_sync(
        self, client: OpenAI, prompt: str, schema: Type[BaseModel]
    ) -> BaseModel:
        try:
            completion = client.beta.chat.completions.parse(
                model=self._api_model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=self._temperature,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception as e:
            logger.debug("beta.chat.completions.parse failed, trying json_object: %s", e)
        completion = client.chat.completions.create(
            model=self._api_model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self._temperature,
        )
        raw = completion.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(raw))

    async def _structured_async(
        self, client: AsyncOpenAI, prompt: str, schema: Type[BaseModel]
    ) -> BaseModel:
        try:
            completion = await client.beta.chat.completions.parse(
                model=self._api_model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=self._temperature,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception as e:
            logger.debug("async beta.chat.completions.parse failed, trying json_object: %s", e)
        completion = await client.chat.completions.create(
            model=self._api_model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self._temperature,
        )
        raw = completion.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(raw))

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        prompt, schema = self._extract_prompt_and_schema(args, dict(kwargs))
        if not prompt:
            raise ValueError("GatewayOpenAIJudge.generate: missing prompt")
        client = self.load_model(async_mode=False)
        if schema is not None:
            return self._structured_sync(client, prompt, schema)
        completion = client.chat.completions.create(
            model=self._api_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        return completion.choices[0].message.content

    async def a_generate(self, *args: Any, **kwargs: Any) -> Any:
        prompt, schema = self._extract_prompt_and_schema(args, dict(kwargs))
        if not prompt:
            raise ValueError("GatewayOpenAIJudge.a_generate: missing prompt")
        client = self.load_model(async_mode=True)
        if schema is not None:
            return await self._structured_async(client, prompt, schema)
        completion = await client.chat.completions.create(
            model=self._api_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        return completion.choices[0].message.content
