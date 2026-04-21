"""Storage-layer exceptions.

Raising these (instead of leaking the SDK's ``S3Error``) keeps FastAPI
route handlers free from SDK-specific imports and makes it obvious what
the failure mode was when reading logs.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for object-storage errors."""


class StorageBucketError(StorageError):
    """Bucket is missing or cannot be created / inspected."""


class StorageUploadError(StorageError):
    """Putting an object into the bucket failed."""


class StorageDownloadError(StorageError):
    """Reading an object back from the bucket failed."""


class StorageDeleteError(StorageError):
    """Removing an object from the bucket failed."""
