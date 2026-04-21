"""Object-storage package.

Currently only exposes the MinIO-backed implementation, but the split
keeps the door open for an S3 / GCS variant behind the same interface
without touching callers.
"""

from app.services.storage.exceptions import (
    StorageBucketError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageUploadError,
)
from app.services.storage.minio_service import (
    MinioStorageService,
    UploadedObject,
    get_storage_service,
)

__all__ = [
    "MinioStorageService",
    "UploadedObject",
    "get_storage_service",
    "StorageError",
    "StorageBucketError",
    "StorageDeleteError",
    "StorageDownloadError",
    "StorageUploadError",
]
