from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rbirag.config import Settings, get_settings


def test_defaults_and_paths() -> None:
    s = Settings()
    assert s.rrf_k == 60
    assert s.fetch_rate_limit_seconds >= 1.0
    assert s.fetch_start_id - s.fetch_min_id >= 2000
    assert s.manifest_path == Path("data/raw/manifest.jsonl")


def test_env_and_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("RBIRAG_TOP_K=9\nRBIRAG_RETRIEVAL_MODE=bm25\n")
    monkeypatch.setenv("RBIRAG_TOP_K", "7")
    s = get_settings()
    assert s.top_k == 7  # env beats .env
    assert s.retrieval_mode == "bm25"


def test_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=64, chunk_overlap=64)
    with pytest.raises(ValidationError):
        Settings(fetch_rate_limit_seconds=0.2)
    with pytest.raises(ValidationError):
        Settings(fetch_start_id=10, fetch_min_id=20)


def test_local_judge_needs_no_provider_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RBIRAG_JUDGE_BACKEND", "ollama")
    monkeypatch.setenv("RBIRAG_LLM_PROVIDER", "groq")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.judge_model == "qwen2.5:7b-instruct"
    assert s.judge_model_fallbacks == []
    assert s.oai_model == "openai/gpt-oss-20b"  # the generator keeps its provider chain
    monkeypatch.setenv("RBIRAG_JUDGE_MODEL", "qwen2.5:14b-instruct")
    assert Settings(_env_file=None).judge_model == "qwen2.5:14b-instruct"  # type: ignore[call-arg]
