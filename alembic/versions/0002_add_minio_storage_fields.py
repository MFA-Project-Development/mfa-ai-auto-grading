"""Add MinIO object-storage fields to answer_key_files.

Revision ID: 0002_minio_storage
Revises: 0001_initial
Create Date: 2026-04-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_minio_storage"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "answer_key_files",
        sa.Column("storage_provider", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "answer_key_files",
        sa.Column("bucket_name", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "answer_key_files",
        sa.Column("object_name", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "answer_key_files",
        sa.Column("object_etag", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "answer_key_files",
        sa.Column("stored_file_name", sa.String(length=512), nullable=True),
    )
    op.create_index(
        op.f("ix_answer_key_files_object_name"),
        "answer_key_files",
        ["object_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_answer_key_files_object_name"),
        table_name="answer_key_files",
    )
    op.drop_column("answer_key_files", "stored_file_name")
    op.drop_column("answer_key_files", "object_etag")
    op.drop_column("answer_key_files", "object_name")
    op.drop_column("answer_key_files", "bucket_name")
    op.drop_column("answer_key_files", "storage_provider")
