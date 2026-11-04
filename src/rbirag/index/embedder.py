"""Embedders: a protocol, the real bge-small model, and a deterministic fake for tests."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Returns L2-normalised float32 matrices of shape ``(n, dim)``."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> FloatArray: ...

    def embed_query(self, text: str) -> FloatArray: ...


def _l2_normalise(mat: FloatArray) -> FloatArray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return cast(FloatArray, (mat / np.clip(norms, 1e-12, None)).astype(np.float32))


class SentenceTransformerEmbedder:
    """``BAAI/bge-small-en-v1.5`` (384-d) via sentence-transformers, loaded lazily.

    bge models expect an instruction prefix on *queries only*.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None
        self._dim: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        return self._dim

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> FloatArray:
        return cast(FloatArray, self._encode([self.query_instruction + text])[0])


class FakeEmbedder:
    """Deterministic, network-free embedder for tests and CI.

    Uses signed feature hashing of lower-cased word unigrams and bigrams, so texts that
    share words get similar vectors (dense retrieval tests stay meaningful) while the output
    is identical across processes and platforms (``blake2b``, not Python's ``hash``).
    """

    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.name = f"fake-hash-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> FloatArray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = self._token.findall(text.lower())
        features = words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]
        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign
        if not features:
            vec[0] = 1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> FloatArray:
        return cast(FloatArray, _l2_normalise(self._vector(text)[None, :])[0])


class CachedEmbedder:
    """Disk cache of document embeddings, keyed by model name and text.

    One ``.npz`` file per model holds the vectors; several CI jobs can share it (one job
    embeds the corpus, the others download the file) instead of each re-embedding it.
    Queries are not cached (they are cheap and few)."""

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", inner.name)
        self.path = cache_dir / f"emb-{slug}.npz"
        self._keys: dict[str, int] = {}
        self._vecs: FloatArray | None = None
        if self.path.exists():
            with np.load(self.path) as data:
                keys = [str(k) for k in data["keys"]]
                self._vecs = np.asarray(data["vecs"], dtype=np.float32)
            self._keys = {k: i for i, k in enumerate(keys)}

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        if self._vecs is not None and self._vecs.shape[0]:
            return int(self._vecs.shape[1])
        return self.inner.dim

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.inner.name}\0{text}".encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._keys)

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        missing = sorted({k: i for i, k in enumerate(keys) if k not in self._keys}.values())
        if missing:
            new = self.inner.embed_documents([texts[i] for i in missing])
            start = len(self._keys)
            for j, i in enumerate(missing):
                self._keys[keys[i]] = start + j
            self._vecs = new if self._vecs is None else np.concatenate([self._vecs, new])
            self.save()
        assert self._vecs is not None
        return self._vecs[[self._keys[k] for k in keys]]

    def embed_query(self, text: str) -> FloatArray:
        return self.inner.embed_query(text)

    def save(self) -> None:
        if self._vecs is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._keys, key=self._keys.__getitem__)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, keys=np.array(ordered), vecs=self._vecs)
        tmp.replace(self.path)


def build_embedder(
    backend: str,
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    query_instruction: str = BGE_QUERY_INSTRUCTION,
    batch_size: int = 32,
    device: str | None = None,
    fake_dim: int = 256,
) -> Embedder:
    if backend == "fake":
        return FakeEmbedder(fake_dim)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name, query_instruction=query_instruction, batch_size=batch_size, device=device
        )
    raise ValueError(f"unknown embedder backend: {backend!r}")
