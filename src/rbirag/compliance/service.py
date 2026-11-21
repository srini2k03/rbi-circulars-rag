"""Compliance checks behind the API: one regulation store, cached checkers, background jobs.

Checking a policy takes ~5 LLM calls per clause (tens of seconds each on free tiers), so
``POST /comply/check`` starts a job and the client polls it. Limits keep a public demo
inside free-tier quotas: at most ``compliance_api_max_clauses`` clauses per check and
``compliance_api_max_jobs`` checks running at once (further checks queue).
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from rbirag.compliance.checker import (
    ClauseChecker,
    ClauseFinding,
    ComplianceReport,
    check_policy,
    render_markdown,
)
from rbirag.compliance.clauses import Clause, split_clauses
from rbirag.compliance.policy import PolicyDocument
from rbirag.compliance.regulations import RegulationStore
from rbirag.config import Settings

logger = logging.getLogger(__name__)

System = Literal["agent", "baseline"]
JobState = Literal["queued", "running", "succeeded", "failed"]
CheckerFactory = Callable[[str, str, Settings, RegulationStore], ClauseChecker]


class ComplianceUnavailableError(RuntimeError):
    """The corpus or index is missing, or the requested backend cannot be used."""


class ComplianceJob(BaseModel):
    job_id: str
    state: JobState
    system: System
    backend: str
    policy_name: str
    clauses_total: int
    clauses_done: int = 0
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    findings: list[ClauseFinding] = Field(default_factory=list)  # filled in as clauses finish
    report: ComplianceReport | None = None
    markdown: str | None = None


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def default_checker(
    system: str, backend: str, settings: Settings, store: RegulationStore
) -> ClauseChecker:
    from rbirag.compliance.agent import ComplianceAgent
    from rbirag.compliance.baseline import FixedPipeline
    from rbirag.compliance.llm import build_chat_model

    chat = build_chat_model(settings, backend)
    if system == "agent":
        return ComplianceAgent.from_settings(settings, store, chat)
    return FixedPipeline.from_settings(settings, store, chat)


class ComplianceService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: RegulationStore | None = None,
        checker_factory: CheckerFactory = default_checker,
    ) -> None:
        self.settings = settings
        self._store = store
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self.checker_factory = checker_factory
        self._checkers: dict[tuple[str, str], ClauseChecker] = {}
        self._checker_lock = threading.Lock()
        self._jobs: dict[str, ComplianceJob] = {}
        self._jobs_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, settings.compliance_api_max_jobs), thread_name_prefix="comply"
        )

    # -- store -----------------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._store is not None

    @property
    def notifications(self) -> int:
        return len(self._store.documents) if self._store is not None else 0

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> RegulationStore:
        """Load the regulation store once (index + documents + cached reference graph)."""
        if self._store is not None:
            return self._store
        with self._load_lock:
            if self._store is None:
                try:
                    self._store = RegulationStore.from_settings(self.settings)
                    self._load_error = None
                except FileNotFoundError as exc:
                    self._load_error = str(exc)
                    raise ComplianceUnavailableError(str(exc)) from exc
                logger.info(
                    "compliance store loaded (%d notifications)", len(self._store.documents)
                )
            return self._store

    def preload(self) -> None:
        """Load in the background so the first request does not wait (errors are kept)."""

        def run() -> None:
            try:
                self.load()
            except Exception as exc:  # reported by status()
                self._load_error = str(exc)
                logger.warning("compliance store not loaded: %s", exc)

        threading.Thread(target=run, name="comply-load", daemon=True).start()

    # -- checkers --------------------------------------------------------------------------
    def backends(self) -> dict[str, bool]:
        from rbirag.api.service import backend_configured

        names = ["gemini", "groq", "ollama", "fake"]
        return {b: backend_configured(b, self.settings) for b in names}

    def default_backend(self) -> str:
        configured = self.settings.compliance_llm_backend
        available = self.backends()
        if available.get(configured):
            return configured
        return next((b for b in ("gemini", "groq") if available.get(b)), "fake")

    def checker(self, system: str, backend: str) -> ClauseChecker:
        key = (system, backend)
        with self._checker_lock:
            if key not in self._checkers:
                store = self.load()
                try:
                    self._checkers[key] = self.checker_factory(
                        system, backend, self.settings, store
                    )
                except Exception as exc:  # missing key, unknown backend
                    raise ComplianceUnavailableError(
                        f"backend {backend!r} is not available: {exc}"
                    ) from exc
            return self._checkers[key]

    # -- clauses ---------------------------------------------------------------------------
    def clauses(self, doc: PolicyDocument) -> list[Clause]:
        return split_clauses(doc.text)

    # -- jobs ------------------------------------------------------------------------------
    def submit(
        self,
        doc: PolicyDocument,
        *,
        system: System = "agent",
        backend: str | None = None,
        max_clauses: int | None = None,
    ) -> ComplianceJob:
        backend = backend or self.default_backend()
        if not self.backends().get(backend):
            raise ComplianceUnavailableError(
                f"backend {backend!r} is not configured (set its free API key)"
            )
        limit = self.settings.compliance_api_max_clauses
        n = min(len(self.clauses(doc)), max_clauses or limit, limit)
        if n == 0:
            raise ValueError("no clauses found in the document")
        checker = self.checker(system, backend)  # fail now, not in the background
        job = ComplianceJob(
            job_id=uuid.uuid4().hex[:12],
            state="queued",
            system=system,
            backend=backend,
            policy_name=doc.name,
            clauses_total=n,
            created_at=_now(),
        )
        with self._jobs_lock:
            self._jobs[job.job_id] = job
        self._pool.submit(self._run, job.job_id, doc, checker, n)
        return job

    def _update(self, job_id: str, **changes: object) -> None:
        with self._jobs_lock:
            self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)

    def _run(self, job_id: str, doc: PolicyDocument, checker: ClauseChecker, n: int) -> None:
        self._update(job_id, state="running", started_at=_now())
        store = self.load()

        def progress(done: int, _total: int, finding: ClauseFinding) -> None:
            with self._jobs_lock:
                job = self._jobs[job_id]
                self._jobs[job_id] = job.model_copy(
                    update={"clauses_done": done, "findings": [*job.findings, finding]}
                )

        try:
            job = self.get(job_id)
            assert job is not None
            report = check_policy(
                doc,
                checker,
                store,
                as_of=self.settings.compliance_as_of,
                entity=self.settings.compliance_entity,
                system=job.system,
                max_clauses=n,
                on_progress=progress,
            )
            self._update(
                job_id,
                state="succeeded",
                finished_at=_now(),
                report=report,
                markdown=render_markdown(report),
            )
        except Exception as exc:  # quota, provider outage: the clauses done so far are kept
            logger.exception("compliance job %s failed", job_id)
            self._update(job_id, state="failed", finished_at=_now(), error=str(exc)[:500])

    def get(self, job_id: str) -> ComplianceJob | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
