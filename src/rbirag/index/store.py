"""A tiny in-memory chunk store persisted as JSONL (id -> :class:`Chunk`)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from rbirag.chunking.base import Chunk


class ChunkStore:
    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self._chunks: dict[str, Chunk] = {}
        self.add(chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._chunks.values())

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for c in self._chunks.values():
                fh.write(c.model_dump_json() + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ChunkStore:
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk.model_validate(json.loads(line)))
        return cls(chunks)
