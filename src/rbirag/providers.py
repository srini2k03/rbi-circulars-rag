"""Presets for OpenAI-compatible LLM providers (free tiers).

A preset supplies the endpoint, the name of the environment variable holding the key,
the default model chain per role, and conservative client-side limits. Every value can be
overridden through settings; ``"auto"`` / empty values mean "use the preset".

Limits are set a little *below* the providers' published free-tier limits (as of late
2026), so the client-side limiter, not the provider, is normally what paces requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    api_key_env: str
    generator: tuple[str, ...]  # primary first, then fallbacks
    judge: tuple[str, ...]
    qa_writer: tuple[str, ...]
    rpm: float
    default_rpd: int
    # "json_object" is supported by every Groq chat model; strict "json_schema" is not.
    json_mode: str = "json_schema"
    rpd: dict[str, int] = field(default_factory=dict)
    tpm: dict[str, int] = field(default_factory=dict)  # tokens/minute; absent = no budget

    def rpd_for(self, model: str) -> int:
        return self.rpd.get(model, self.default_rpd)

    def tpm_for(self, model: str) -> int:
        return self.tpm.get(model, 0)


GROQ = ProviderPreset(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    # Fast 8B generator (high daily quota), different from the judge models.
    generator=("llama-3.1-8b-instant", "openai/gpt-oss-20b"),
    judge=("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"),
    qa_writer=("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"),
    rpm=25,  # published free tier: 30 RPM (60 for qwen3-32b)
    json_mode="json_object",
    default_rpd=950,  # published: 1,000/day for the larger models
    rpd={"llama-3.1-8b-instant": 14_000},  # published: 14,400/day
    # published free-tier TPM: 70B 12k, 8B 6k, gpt-oss 8k, qwen3-32b 6k
    tpm={
        "llama-3.3-70b-versatile": 11_000,
        "llama-3.1-8b-instant": 5_500,
        "openai/gpt-oss-120b": 7_500,
        "openai/gpt-oss-20b": 7_500,
        "qwen/qwen3-32b": 5_500,
    },
)

GEMINI = ProviderPreset(
    name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key_env="GEMINI_API_KEY",
    generator=("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"),
    judge=("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"),
    qa_writer=("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"),
    rpm=10,
    default_rpd=200,
)

PRESETS: dict[str, ProviderPreset] = {p.name: p for p in (GROQ, GEMINI)}


def get_preset(name: str) -> ProviderPreset | None:
    return PRESETS.get(name)
