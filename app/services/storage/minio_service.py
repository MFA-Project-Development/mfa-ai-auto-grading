"""MinIO-backed object storage for answer-key source files.

PostgreSQL owns file metadata, ChromaDB owns vectors, and MinIO owns the
original uploaded bytes. This module is the only place in the app that
talks to the MinIO SDK - every other module goes through the
:class:`MinioStorageService` facade so we can swap implementations later
without ripping through the codebase.

Design notes:

* The client is initialised lazily. Importing this module from the CLI
  (e.g. Alembic) must not require MinIO to be reachable.
* Errors from ``minio.S3Error`` are wrapped in our own exception
  hierarchy so route handlers never import the SDK directly.
* ``build_object_name`` is deterministic and only depends on ``file_id``
  plus the original filename extension. That makes objects trivially
  linkable to ``AnswerKeyFile`` rows and keeps reprocessing possible
  without extra lookups.
"""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.core.config import Settings, settings as _default_settings
from app.services.storage.exceptions import (
    StorageBucketError,
    StorageDeleteError,
    StorageDownloadError,
    StorageUploadError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadedObject:
    """Result of a successful upload.

    Mirrors what we persist on ``AnswerKeyFile`` so the ingestion
    pipeline can copy these fields straight through.
    """

    bucket_name: str
    object_name: str
    etag: str | None
    size: int
    content_type: str | None


class MinioStorageService:
    """Thin wrapper over the MinIO Python SDK."""

    def __init__(self, app_settings: Settings | None = None) -> None:
        self._settings = app_settings or _default_settings
        self._client: Minio | None = None
        self._bucket_ready: bool = False

    # ================================================================== client
    @property
    def client(self) -> Minio:
        """Return a lazily-initialised MinIO client."""
        if self._client is None:
            self._client = Minio(
                endpoint=self._settings.minio_endpoint,
                access_key=self._settings.minio_access_key,
                secret_key=self._settings.minio_secret_key,
                secure=self._settings.minio_secure,
                region=self._settings.minio_region,
            )
        return self._client

    @property
    def bucket_name(self) -> str:
        return self._settings.minio_bucket_name

    # ================================================================== bucket
    def ensure_bucket_exists(self, *, force: bool = False) -> None:
        """Create the configured bucket when absent. Idempotent.

        Caches a positive result so subsequent requests do not hit MinIO
        again. Pass ``force=True`` to re-check after an operator has
        wiped the bucket in the background.

        Raises:
            StorageBucketError: when the MinIO server is unreachable or
                rejects the create request.
        """
        if self._bucket_ready and not force:
            return

        try:
            if not self.client.bucket_exists(self.bucket_name):
                logger.info(
                    "minio: bucket=%s missing, creating", self.bucket_name
                )
                self.client.make_bucket(
                    self.bucket_name,
                    location=self._settings.minio_region,
                )
            else:
                logger.debug("minio: bucket=%s already exists", self.bucket_name)
        except S3Error as exc:
            raise StorageBucketError(
                f"Failed to inspect/create bucket {self.bucket_name!r}: {exc}"
            ) from exc
        except Exception as exc:  # connection errors etc.
            raise StorageBucketError(
                f"MinIO is unreachable at {self._settings.minio_endpoint!r}: {exc}"
            ) from exc

        self._bucket_ready = True

    # ================================================================== naming
    def build_object_name(
        self,
        file_id: uuid.UUID | str,
        original_filename: str,
    ) -> str:
        """Canonical object name for a given ``file_id``.

        Layout: ``{prefix}/{file_id}/original{ext}``. The file_id anchors
        the object to its SQL row so reprocessing never needs the
        original filename to locate the bytes. The extension is preserved
        (lowercased) so CDNs / object browsers still display the right
        MIME hint.
        """
        ext = Path(original_filename).suffix.lower()
        safe_file_id = str(file_id)
        prefix = self._settings.minio_object_prefix.strip("/")
        return f"{prefix}/{safe_file_id}/original{ext}" if prefix else (
            f"{safe_file_id}/original{ext}"
        )

    # ================================================================== upload
    def upload_file_bytes(
        self,
        *,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
        extra_metadata: dict[str, str] | None = None,
    ) -> UploadedObject:
        """Put raw bytes into the configured bucket.

        The SDK wants a stream, so we wrap the payload in ``BytesIO``
        once and pass the explicit length to avoid the SDK's fallback
        multipart logic (which is wasteful for typical answer-key PDFs).

        Raises:
            StorageUploadError: on any SDK / transport failure.
        """
        payload = io.BytesIO(data)
        size = len(data)
        try:
            result = self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=payload,
                length=size,
                content_type=content_type or "application/octet-stream",
                metadata=extra_metadata,
            )
        except S3Error as exc:
            raise StorageUploadError(
                f"MinIO rejected upload object={object_name!r}: {exc}"
            ) from exc
        except Exception as exc:  # connection errors
            raise StorageUploadError(
                f"MinIO upload failed object={object_name!r}: {exc}"
            ) from exc
        finally:
            payload.close()

        logger.info(
            "minio: uploaded bucket=%s object=%s size=%d etag=%s",
            self.bucket_name, object_name, size, result.etag,
        )

        return UploadedObject(
            bucket_name=self.bucket_name,
            object_name=object_name,
            etag=result.etag,
            size=size,
            content_type=content_type,
        )

    # ================================================================== delete
    def delete_object(self, object_name: str) -> bool:
        """Remove an object. Returns ``True`` when something was deleted.

        MinIO's ``remove_object`` is a no-op for missing keys, so we
        probe first via ``stat_object`` to report whether anything was
        actually there.

        Raises:
            StorageDeleteError: on SDK failures other than ``NoSuchKey``.
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject"}:
                logger.info(
                    "minio: delete skipped object=%s (not present)", object_name
                )
                return False
            raise StorageDeleteError(
                f"MinIO stat failed object={object_name!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise StorageDeleteError(
                f"MinIO stat error object={object_name!r}: {exc}"
            ) from exc

        try:
            self.client.remove_object(self.bucket_name, object_name)
        except S3Error as exc:
            raise StorageDeleteError(
                f"MinIO delete failed object={object_name!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise StorageDeleteError(
                f"MinIO delete error object={object_name!r}: {exc}"
            ) from exc

        logger.info(
            "minio: deleted bucket=%s object=%s", self.bucket_name, object_name
        )
        return True

    # ================================================================== presign
    def get_presigned_url(
        self,
        object_name: str,
        expires_seconds: int = 3600,
        response_headers: dict[str, str] | None = None,
    ) -> str:
        """Generate a time-limited GET URL for ``object_name``.

        When ``MINIO_PUBLIC_ENDPOINT`` is set the resulting URL is
        rewritten so browsers receive the externally reachable host
        instead of the internal Docker service name.
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=timedelta(seconds=expires_seconds),
                response_headers=response_headers,
            )
        except S3Error as exc:
            raise StorageDownloadError(
                f"MinIO presign failed object={object_name!r}: {exc}"
            ) from exc

        public = self._settings.minio_public_endpoint
        if public and public != self._settings.minio_endpoint:
            url = url.replace(
                self._settings.minio_endpoint,
                public,
                1,
            )
        return url


# ====================================================================== module-level singleton
_service: MinioStorageService | None = None


def get_storage_service() -> MinioStorageService:
    """Return the process-wide :class:`MinioStorageService` instance."""
    global _service
    if _service is None:
        _service = MinioStorageService()
    return _service
