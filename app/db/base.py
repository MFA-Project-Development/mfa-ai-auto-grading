"""SQLAlchemy declarative base.

All ORM models inherit from :class:`Base`. This module also imports every
model package so tables are registered on the metadata by the time Alembic
runs autogenerate / ``Base.metadata.create_all()`` is called.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Predictable constraint naming makes Alembic diffs deterministic across
# environments, which matters as soon as you add indexes / FKs later.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Root SQLAlchemy 2.x declarative base with shared metadata."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
