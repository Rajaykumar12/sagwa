"""trace_steps: one row per step of an agentic case's trajectory

Additive, with no backfill: runs from before the agentic work legitimately
have no trajectory.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trace_steps",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("result_id", sa.String(), sa.ForeignKey("results.id"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("input", sa.String(), nullable=True),
        sa.Column("output", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
    )
    op.create_index("ix_trace_steps_result_id", "trace_steps", ["result_id"])


def downgrade() -> None:
    op.drop_index("ix_trace_steps_result_id", table_name="trace_steps")
    op.drop_table("trace_steps")
