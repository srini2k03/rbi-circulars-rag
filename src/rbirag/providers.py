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
    # Chains use only chat models listed for the project's key (GET /openai/v1/models,
    # 2026-11): llama-3.3-70b-versatile and llama-3.1-8b-instant returned 404 / are not
    # listed. The generator (20B) differs from the judge (120B).
    generator=("openai/gpt-oss-20b", "qwen/qwen3.8-27b"),
    judge=("openai/gpt-oss-120b", "qwen/qwen3.8-27b"),
    qa_writer=("openai/gpt-oss-120b", "qwen/qwen3.8-27b"),
    rpm=25,  # published free tier: 30 RPM
    json_mode="json_object",
    default_rpd=950,  # published: 1,000/day
    # published free-tier TPM: gpt-oss 8k; the Qwen budget is a conservative guess
    tpm={
        "openai/gpt-oss-120b": 7_500,
        "openai/gpt-oss-20b": 7_500,
        "qwen/qwen3.8-27b": 5_500,
        "llama-3.3-70b-versatile": 11_000,
        "llama-3.1-8b-instant": 5_500,
    },
    rpd={"llama-3.1-8b-instant": 14_000},
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
