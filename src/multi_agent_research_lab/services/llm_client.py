"""LLM client abstraction — OpenAI backend with tenacity retry and token tracking."""

import logging
from dataclasses import dataclass

from openai import APIError, APITimeoutError, BadRequestError, RateLimitError

try:
    from langfuse.openai import OpenAI
except ImportError:
    from openai import OpenAI  # type: ignore[no-redef]
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# gpt-4o-mini pricing (USD per token) — used as fallback estimate
_INPUT_PRICE = 0.150 / 1_000_000
_OUTPUT_PRICE = 0.600 / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client — OpenAI implementation.

    Handles models that do not support custom temperature (e.g. newer reasoning models)
    by automatically falling back to the model default (omitting the parameter).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._client: OpenAI | None = None
        self._temperature_supported: bool = True  # flipped to False on first 400

    def _get_client(self) -> OpenAI:
        if self._client is None:
            from multi_agent_research_lab.core.config import get_settings

            api_key = get_settings().openai_api_key
            self._client = OpenAI(api_key=api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the LLM and return content + usage stats."""
        logger.debug("LLM call model=%s prompt_len=%d", self._model, len(user_prompt))
        kwargs: dict = dict(
            model=self._model,
            timeout=self._timeout,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self._temperature_supported:
            kwargs["temperature"] = self._temperature

        try:
            response = self._get_client().chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "temperature" in str(exc).lower() and self._temperature_supported:
                # Model does not support custom temperature; retry without it
                logger.warning("Model %s does not support temperature — using default", self._model)
                self._temperature_supported = False
                kwargs.pop("temperature", None)
                response = self._get_client().chat.completions.create(**kwargs)
            else:
                raise

        usage = response.usage
        content = response.choices[0].message.content or ""
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        cost = (
            in_tok * _INPUT_PRICE + out_tok * _OUTPUT_PRICE
            if in_tok is not None and out_tok is not None
            else None
        )
        logger.debug("LLM response tokens=%s/%s cost_usd=%.6f", in_tok, out_tok, cost or 0)
        return LLMResponse(
            content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost
        )
