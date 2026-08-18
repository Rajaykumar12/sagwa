"""initial: runs, results

Revision ID: 0001
Revises:
Create Date: 2026-08-18
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("sagwa_git_sha", sa.String(), nullable=False),
        sa.Column("target_pipeline_git_sha", sa.String(), nullable=True),
        sa.Column("target_name", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("dataset_path", sa.String(), nullable=False),
        sa.Column("dataset_sha256", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "results",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("input", sa.String(), nullable=False),
        sa.Column("output", sa.String(), nullable=False),
        sa.Column("context", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("trace_id", sa.String(), nullable=True),
    )
    op.create_index("ix_results_run_id", "results", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_results_run_id", table_name="results")
    op.drop_table("results")
    op.drop_table("runs")
