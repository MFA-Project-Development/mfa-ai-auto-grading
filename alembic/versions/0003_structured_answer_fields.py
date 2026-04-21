"""Add structured answer-key fields to answer_key_items.

Adds a cleaned-up, comparison-ready projection alongside the raw chunk:

* ``problem_text``       TEXT  - first non-``=`` line after the question number.
* ``solution_steps_json`` JSONB - ordered list of ``=`` step lines.
* ``final_answer``       TEXT  - raw final answer (what followed ``답``).
* ``normalized_answer``  TEXT  - normalized final answer for exact matching.
* ``answer_type``        VARCHAR(32) - rule-based classification
  (numeric / polynomial_expression / factored_expression /
  equation_solution / multiple_part_answer / unknown).
* ``formula_list_json``  JSONB - every distinct formula fragment found in
  the chunk, used for partial / step-level grading.

The existing ``content`` and ``answer_text`` columns are kept as-is so we
can always re-derive the structured projection from the original raw
chunk.

Revision ID: 0003_structured_answer_fields
Revises: 0002_minio_storage
Create Date: 2026-04-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_structured_answer_fields"
down_revision: Union[str, None] = "0002_minio_storage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "answer_key_items",
        sa.Column("problem_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "answer_key_items",
        sa.Column(
            "solution_steps_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "answer_key_items",
        sa.Column("final_answer", sa.Text(), nullable=True),
    )
    op.add_column(
        "answer_key_items",
        sa.Column("normalized_answer", sa.Text(), nullable=True),
    )
    op.add_column(
        "answer_key_items",
        sa.Column("answer_type", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "answer_key_items",
        sa.Column(
            "formula_list_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_index(
        op.f("ix_answer_key_items_answer_type"),
        "answer_key_items",
        ["answer_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_answer_key_items_answer_type"),
        table_name="answer_key_items",
    )
    op.drop_column("answer_key_items", "formula_list_json")
    op.drop_column("answer_key_items", "answer_type")
    op.drop_column("answer_key_items", "normalized_answer")
    op.drop_column("answer_key_items", "final_answer")
    op.drop_column("answer_key_items", "solution_steps_json")
    op.drop_column("answer_key_items", "problem_text")
