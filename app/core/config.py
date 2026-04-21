"""Application configuration.

Centralised settings driven entirely by environment variables (with a
``.env`` file fallback for local development).

Database configuration supports two styles so the project works both in a
local dev setup (docker compose exposes Postgres on a non-default port) and
in a production environment where a full ``DATABASE_URL`` is handed to the
container:

* If ``DATABASE_URL`` is set, it wins verbatim.
* Otherwise a URL is assembled from the individual ``POSTGRES_*`` vars.

All SQLAlchemy/Alembic code should import :data:`settings` and read
``settings.sqlalchemy_url`` instead of touching environment variables
directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Answer Key Ingestion API"

    # ------------------------------------------------------------------ vector store
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "answer_keys_bge_m3"

    # ------------------------------------------------------------------ file storage
    # Optional: when set, uploaded files are persisted here and the filesystem
    # path is recorded on ``AnswerKeyFile.storage_path``. When left empty the
    # file body is not kept on disk.
    storage_dir: str = Field(default="", alias="STORAGE_DIR")

    # ------------------------------------------------------------------ database
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="mfa_db", alias="POSTGRES_DB")
    postgres_user: str = Field(default="mfa_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="mfa_password", alias="POSTGRES_PASSWORD")

    db_echo: bool = Field(default=False, alias="DB_ECHO")
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")

    # ------------------------------------------------------------------ object storage (MinIO)
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="mfa-bucket-grading", alias="MINIO_BUCKET_NAME")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")
    minio_region: str | None = Field(default=None, alias="MINIO_REGION")
    # Optional public-facing endpoint used for presigned URLs. Useful when
    # the API server reaches MinIO on an internal Docker hostname but the
    # URL has to be handed back to a browser.
    minio_public_endpoint: str | None = Field(default=None, alias="MINIO_PUBLIC_ENDPOINT")
    # The object-name prefix used for answer-key uploads. Keeping it
    # configurable lets multiple environments share a bucket.
    minio_object_prefix: str = Field(default="answer-keys", alias="MINIO_OBJECT_PREFIX")

    # ------------------------------------------------------------------ Keycloak / OIDC
    # The API acts as an OAuth2 resource server: it does not issue tokens,
    # it only validates bearer JWTs minted by Keycloak. All values below
    # must be sourced from env vars / ``.env``; never hardcode secrets.
    keycloak_issuer: str = Field(alias="KEYCLOAK_ISSUER")
    keycloak_jwks_uri: str = Field(alias="KEYCLOAK_JWKS_URI")
    keycloak_client_id: str = Field(alias="KEYCLOAK_CLIENT_ID")
    keycloak_client_secret: str = Field(alias="KEYCLOAK_CLIENT_SECRET")
    keycloak_token_endpoint: str = Field(alias="KEYCLOAK_TOKEN_ENDPOINT")

    @computed_field  # type: ignore[misc]
    @property
    def sqlalchemy_url(self) -> str:
        """Resolve the SQLAlchemy connection URL.

        Prefers ``DATABASE_URL`` if provided; otherwise builds a psycopg2
        URL from the ``POSTGRES_*`` vars. The ``postgresql+psycopg2://``
        scheme is used explicitly so SQLAlchemy picks the right driver
        regardless of what happens to be installed alongside.
        """
        if self.database_url:
            return self.database_url
        return (
            "postgresql+psycopg2://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()


settings = get_settings()
