"""Model providers and paths.

Zhipu is reached through its OpenAI-compatible endpoint (langchain-openai); Gemini through the
native LangChain integration (langchain-google-genai), which keeps the thought signatures that
Gemini 3 requires on multi-turn tool calls. The rest of the agent only sees a BaseChatModel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str  # "openai_compatible" or "google_genai"
    base_url: str | None
    key_envs: tuple[str, ...]

    def api_key(self) -> str:
        for env in self.key_envs:
            if os.environ.get(env):
                return os.environ[env]
        raise RuntimeError(
            f"No API key for provider {self.name}: set one of {', '.join(self.key_envs)} "
            "(a .env file in the repo root is also read)."
        )


ZHIPU = Provider("zhipu", "openai_compatible", "https://open.bigmodel.cn/api/paas/v4", ("ZHIPUAI_API_KEY", "GLM_KEY"))
GOOGLE = Provider("google", "google_genai", None, ("GEMINI_API_KEY", "GOOGLE_API_KEY"))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: Provider
    # USD per million tokens, used only for cost reporting.
    usd_per_m_input: float
    usd_per_m_output: float
    extra_body: dict = field(default_factory=dict)
    rpm: int | None = None  # client-side request cap per minute (free tiers), shared by every session

    def chat_model(self, temperature: float = 0.0) -> BaseChatModel:
        if self.provider.kind == "google_genai":
            return ChatGoogleGenerativeAI(
                model=self.name,
                rate_limiter=_rate_limiter(self.name, self.rpm),
                google_api_key=self.provider.api_key(),
                temperature=temperature,
                max_tokens=1500,
                timeout=120,
                max_retries=6,
            )
        return ChatOpenAI(
            model=self.name,
            base_url=self.provider.base_url,
            api_key=self.provider.api_key(),
            temperature=temperature,
            max_tokens=1500,
            timeout=120,
            max_retries=6,
            extra_body=self.extra_body or None,
            rate_limiter=_rate_limiter(self.name, self.rpm),
        )


@cache
def _rate_limiter(model: str, rpm: int | None) -> InMemoryRateLimiter | None:
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=1) if rpm else None


_NO_THINKING = {"thinking": {"type": "disabled"}}

# Prices from the Z.ai price list (https://docs.z.ai/guides/overview/pricing), checked 2026-11-25;
# cache discounts are ignored, so costs are upper bounds. Gemini 3.5 Flash-Lite runs on the free
# tier of the Gemini API.
MODELS: dict[str, ModelSpec] = {
    "glm-4.7-flashx": ModelSpec("glm-4.7-flashx", ZHIPU, 0.07, 0.4, _NO_THINKING),
    "glm-4.5-air": ModelSpec("glm-4.5-air", ZHIPU, 0.2, 1.1, _NO_THINKING),
    # Free tier: 5 requests per minute for this model (429 RESOURCE_EXHAUSTED above that), so cap at 4.
    "gemini-3.5-flash-lite": ModelSpec("gemini-3.5-flash-lite", GOOGLE, 0.0, 0.0, rpm=4),
}

DEFAULT_MODEL = "glm-4.7-flashx"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines, optional `export`), never overrides real env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
