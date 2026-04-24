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

from dotenv import load_dotenv
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Push ``.env`` into ``os.environ`` so BOTH pydantic-settings (which reads
# from its own env_file path) AND every ``os.getenv(...)`` call scattered
# across the app (``grading_service``, ``ocr_service``, ...) see the same
# values. ``pydantic-settings`` alone does NOT populate ``os.environ``,
# which is a common footgun: env-var toggles like
# ``GRADING_FINAL_ANSWER_SAFETY_NET=0`` or ``GLM_OCR_BASE_URL=...`` silently
# fall back to their code-level defaults unless we do this explicitly.
#
# ``override=False`` respects any var already set in the shell / container
# environment (docker-compose, CI, direct ``$env:FOO=...`` in PowerShell)
# so operators can override individual vars without editing the file.
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


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

    # ------------------------------------------------------------------ assessment API
    # Default base URL used by ``app.services.assessment_client``.
    # Individual call sites may still pass their own base URL when they
    # target a different upstream.
    assessment_api_base_url: str = Field(
        default="https://phase-one-api.dara-it.site/api/v1/",
        alias="ASSESSMENT_API_BASE_URL",
    )
    # Optional static API key injected as ``Authorization: Bearer <key>``
    # (or ``X-API-Key`` - see :class:`AssessmentAPIClient`). Leave empty
    # to disable auth. Never hardcode - always source from env.
    assessment_api_key: str = Field(default="", alias="ASSESSMENT_API_KEY")
    # Header scheme used when ``assessment_api_key`` is set. ``bearer``
    # sends ``Authorization: Bearer <key>``; ``x-api-key`` sends
    # ``X-API-Key: <key>``; ``none`` disables auth even if a key is set.
    assessment_api_auth_scheme: str = Field(
        default="bearer", alias="ASSESSMENT_API_AUTH_SCHEME"
    )
    assessment_api_timeout_seconds: float = Field(
        default=10.0, alias="ASSESSMENT_API_TIMEOUT_SECONDS"
    )
    assessment_api_max_retries: int = Field(
        default=3, alias="ASSESSMENT_API_MAX_RETRIES"
    )
    assessment_api_backoff_seconds: float = Field(
        default=0.5, alias="ASSESSMENT_API_BACKOFF_SECONDS"
    )

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
