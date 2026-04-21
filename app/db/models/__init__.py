"""SQLAlchemy model package.

Import models here so ``Base.metadata`` is fully populated whenever
``app.db.models`` is imported (Alembic autogenerate + ``init_db()``
both rely on this).
"""

from app.db.models.answer_key_file import AnswerKeyFile
from app.db.models.answer_key_item import AnswerKeyItem

__all__ = ["AnswerKeyFile", "AnswerKeyItem"]
