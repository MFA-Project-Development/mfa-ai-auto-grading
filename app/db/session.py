"""SQLAlchemy engine / session plumbing.

Exposes:

* :data:`engine` - process-wide engine configured from :mod:`app.core.config`.
* :data:`SessionLocal` - session factory.
* :func:`get_db` - FastAPI dependency yielding a scoped session.
* :func:`init_db` - one-shot metadata creation (useful in tests / first boot
  when Alembic is not yet configured).
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.sqlalchemy_url,
    echo=settings.db_echo,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a single transactional session.

    The session is closed after the request finishes regardless of whether
    the endpoint raised. Endpoints are expected to commit explicitly (or
    delegate commits to the repository / service layer) so failures cleanly
    roll back.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables declared on :class:`app.db.base.Base`.

    Primarily intended for local bootstrapping and unit tests. Production
    deployments should rely on Alembic migrations instead.
    """
    # Import models so they are registered on Base.metadata before
    # create_all runs.
    from app.db import models  # noqa: F401  (side-effect import)
    from app.db.base import Base

    Base.metadata.create_all(bind=engine)
