import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api.routes.assessment import router as assessment_router
from app.api.routes.auth import router as auth_router
from app.api.routes.delete import router as delete_router
from app.api.routes.detail import router as detail_router
from app.api.routes.grading import router as grading_router
from app.api.routes.grading_pipeline import router as grading_pipeline_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.security import get_current_user, init_jwks_cache
from app.db.session import init_db
from app.services.assessment_client import get_assessment_client
from app.services.ocr_service import get_ocr_status, log_ocr_status
from app.services.storage import StorageError, get_storage_service

# ``force=True`` makes sure our handlers/level win even if uvicorn (or any
# imported library like PaddleOCR) installed handlers on the root logger
# before this point. Without it, third-party libs can silently suppress our
# INFO logs from ``app.*`` loggers on process startup or after a reload.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
# Belt-and-suspenders: ensure our own application loggers are at INFO even if
# something later bumps the root logger up to WARNING.
logging.getLogger("app").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: warm up external dependencies on boot.

    Startup order:
      1. Optionally create SQL tables (dev-only, gated on AUTO_CREATE_DB=1).
      2. Probe MinIO and ensure the answer-key bucket exists.
      3. Log the PaddleOCR fallback status (package importability only).
      4. Pre-fetch the Keycloak JWKS so the first authenticated request
         does not pay the JWKS round-trip.

    Any failure in steps 1-3 is logged but non-fatal; the JWKS fetch in
    step 4 is also best-effort - a miss will just trigger the lazy
    refresh path on the first request.
    """
    if os.getenv("AUTO_CREATE_DB") == "1":
        logger.info("startup: AUTO_CREATE_DB=1, calling init_db()")
        try:
            init_db()
        except Exception:  # pragma: no cover
            logger.exception("startup: init_db failed")

    try:
        get_storage_service().ensure_bucket_exists()
        logger.info("startup: MinIO bucket %s ready", settings.minio_bucket_name)
    except StorageError as exc:
        logger.warning(
            "startup: MinIO warm-up failed (%s). Uploads will retry on demand.",
            exc,
        )

    # We deliberately pass ``probe_engine=False`` here: we only want to
    # confirm the package is importable. Building the engine during startup
    # would make ``--reload`` painfully slow and let PaddleOCR hijack the
    # root logger before any request runs.
    log_ocr_status(probe_engine=False)

    try:
        await init_jwks_cache()
    except Exception:  # pragma: no cover - network-dependent
        logger.exception(
            "startup: JWKS warm-up failed; will retry lazily on first request"
        )

    yield

    # ------------------------------------------------------------------ shutdown
    # Close the shared assessment-API client so httpx can release its
    # connection pool cleanly on --reload and graceful shutdown.
    try:
        await get_assessment_client().aclose()
    except Exception:  # pragma: no cover - best-effort cleanup
        logger.exception("shutdown: assessment-client close failed")


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# Every non-auth router is locked behind a valid bearer token. The auth
# router is mounted without a global guard so ``/login`` stays
# reachable; its own protected endpoint (``/me``) declares the
# dependency per-route.
_auth_required = [Depends(get_current_user)]

app.include_router(auth_router)
app.include_router(upload_router, dependencies=_auth_required)
app.include_router(detail_router, dependencies=_auth_required)
app.include_router(delete_router, dependencies=_auth_required)
# Grading endpoints further restrict access to ROLE_INSTRUCTOR / ROLE_ADMIN
# via ``Depends(require_role(...))`` declared on each route handler.
app.include_router(grading_router, dependencies=_auth_required)
# Assessment-API passthroughs reuse the caller's Keycloak token when
# calling the upstream service.
app.include_router(assessment_router, dependencies=_auth_required)
# End-to-end auto-grading pipeline: chains the upstream passthroughs
# together with the local VLM grading engine. Role check
# (ROLE_INSTRUCTOR / ROLE_ADMIN) lives on the route handler.
app.include_router(grading_pipeline_router, dependencies=_auth_required)


@app.get("/", dependencies=_auth_required)
def root() -> dict[str, str]:
    return {"message": "API is running"}
