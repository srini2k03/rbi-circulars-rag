"""Disk cache for LLM calls, keyed by a hash of the complete request.

Reruns and resumed workflows reuse earlier responses instead of spending quota. Entries
are small JSON files under ``<dir>/<hash[:2]>/<hash>.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from rbirag.generation.llm import LLM, LLMResponse, Message, StreamChunk

logger = logging.getLogger(__name__)
CACHE_VERSION = 1


def request_key(
    *,
    backend: str,
    model: str,
    system: str | None,
    messages: Sequence[Message],
    max_tokens: int,
    json_schema: dict[str, Any] | None,
) -> str:
    payload = {
        "v": CACHE_VERSION,
        "backend": backend,
        "model": model,
        "system": system,
        "messages": [[m.role, m.content] for m in messages],
        "max_tokens": max_tokens,
        "json_schema": json_schema,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class CachedLLM:
    """Wraps any :class:`LLM`; ``generate`` and ``stream`` both read/write the cache."""

    def __init__(self, inner: LLM, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0

    @property
    def backend(self) -> str:
        return self.inner.backend

    @property
    def model(self) -> str:
        return self.inner.model

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("ignoring unreadable cache entry %s", path.name)
            return None

    def _put(self, key: str, response: LLMResponse) -> None:
        if response.refused or not response.text:
            return  # never cache failures; they may succeed on a rerun
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json(), encoding="utf-8")
        tmp.replace(path)

    def _key(
        self,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> str:
        return request_key(
            backend=self.backend,
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        key = self._key(messages, system, max_tokens, json_schema)
        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            return cached.model_copy(update={"cached": True, "latency_ms": 0.0})
        self.misses += 1
        response = self.inner.generate(
            messages, system=system, max_tokens=max_tokens, json_schema=json_schema
        )
        self._put(key, response)
        return response

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        key = self._key(messages, system, max_tokens, None)
        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            for piece in re.findall(r"\S+\s*", cached.text):
                yield StreamChunk(text=piece)
            yield StreamChunk(final=cached.model_copy(update={"cached": True, "latency_ms": 0.0}))
            return
        self.misses += 1
        for chunk in self.inner.stream(messages, system=system, max_tokens=max_tokens):
            if chunk.final is not None:
                self._put(key, chunk.final)
            yield chunk
