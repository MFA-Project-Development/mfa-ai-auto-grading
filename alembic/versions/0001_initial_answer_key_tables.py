"""Initial tables: answer_key_files and answer_key_items.

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "answer_key_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_name", sa.String(length=512), nullable=False),
        sa.Column("original_file_name", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("pdf_mode", sa.String(length=16), nullable=True),
        sa.Column("parser_used", sa.String(length=64), nullable=True),
        sa.Column(
            "ingestion_status",
            sa.String(length=16),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("total_pages", sa.Integer(), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("subject", sa.String(length=128), nullable=True),
        sa.Column("grade", sa.String(length=64), nullable=True),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_answer_key_files")),
    )
    op.create_index(
        op.f("ix_answer_key_files_file_name"),
        "answer_key_files",
        ["file_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_answer_key_files_ingestion_status"),
        "answer_key_files",
        ["ingestion_status"],
        unique=False,
    )

    op.create_table(
        "answer_key_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question_no", sa.String(length=64), nullable=False),
        sa.Column("heading_text", sa.String(length=512), nullable=True),
        sa.Column("chapter", sa.String(length=256), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("page_numbers", sa.String(length=512), nullable=True),
        sa.Column("parser_used", sa.String(length=64), nullable=True),
        sa.Column("chunk_type", sa.String(length=32), nullable=False),
        sa.Column("vector_id", sa.String(length=256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["file_id"],
            ["answer_key_files.id"],
            name=op.f("fk_answer_key_items_file_id_answer_key_files"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_answer_key_items")),
        sa.UniqueConstraint(
            "file_id",
            "question_no",
            name="uq_answer_key_items_file_id_question_no",
        ),
        sa.UniqueConstraint(
            "vector_id",
            name=op.f("uq_answer_key_items_vector_id"),
        ),
    )
    op.create_index(
        op.f("ix_answer_key_items_file_id"),
        "answer_key_items",
        ["file_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_answer_key_items_question_no"),
        "answer_key_items",
        ["question_no"],
        unique=False,
    )
    op.create_index(
        op.f("ix_answer_key_items_vector_id"),
        "answer_key_items",
        ["vector_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_answer_key_items_vector_id"), table_name="answer_key_items")
    op.drop_index(op.f("ix_answer_key_items_question_no"), table_name="answer_key_items")
    op.drop_index(op.f("ix_answer_key_items_file_id"), table_name="answer_key_items")
    op.drop_table("answer_key_items")

    op.drop_index(op.f("ix_answer_key_files_ingestion_status"), table_name="answer_key_files")
    op.drop_index(op.f("ix_answer_key_files_file_name"), table_name="answer_key_files")
    op.drop_table("answer_key_files")
