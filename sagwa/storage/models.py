"""Run history schema (PRD FR-6, FR-7, FR-7a).

Runs are immutable and append-only: nothing here is ever updated in place
after a run completes (NFR "Auditability").
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)

    # Reproducibility (FR-7, FR-7a): both repos version independently, so a
    # regression could originate on either side — pin both git SHAs.
    sagwa_git_sha: Mapped[str] = mapped_column(String)
    target_pipeline_git_sha: Mapped[str | None] = mapped_column(String, nullable=True)

    target_name: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    dataset_path: Mapped[str] = mapped_column(String)
    dataset_sha256: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String, default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    results: Mapped[list["Result"]] = relationship(back_populates="run")


class Result(Base):
    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    case_id: Mapped[str] = mapped_column(String)

    input: Mapped[str] = mapped_column(String)
    output: Mapped[str] = mapped_column(String)
    context: Mapped[str | None] = mapped_column(String, nullable=True)

    latency_ms: Mapped[int] = mapped_column(Integer)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Populated by the metrics layer (Week 3-6) — empty dict until then.
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="results")
