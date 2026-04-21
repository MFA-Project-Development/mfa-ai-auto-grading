#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Container entrypoint for the MFA auto-grading API.
#
# Responsibilities:
#   1. Wait briefly for Postgres to be reachable (compose `depends_on`
#      already gates on the healthcheck, but this protects direct
#      `docker run` invocations too).
#   2. Apply Alembic migrations so the schema matches the code.
#   3. exec into the main command (uvicorn by default).
#
# Toggle behaviour via env:
#   RUN_MIGRATIONS=0   - skip `alembic upgrade head`
#   WAIT_FOR_DB=0      - skip the Postgres readiness probe
# ---------------------------------------------------------------------------
set -euo pipefail

wait_for_db() {
    if [[ "${WAIT_FOR_DB:-1}" != "1" ]]; then
        return 0
    fi

    python - <<'PY'
import os
import sys
import time

from sqlalchemy import create_engine, text

from app.core.config import settings

url = settings.sqlalchemy_url
deadline = time.monotonic() + float(os.getenv("DB_WAIT_TIMEOUT", "60"))
last_err = None
while time.monotonic() < deadline:
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[entrypoint] database reachable at {url.split('@')[-1]}", flush=True)
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_err = exc
        time.sleep(1.5)

print(f"[entrypoint] database not reachable within timeout: {last_err}", file=sys.stderr, flush=True)
sys.exit(1)
PY
}

run_migrations() {
    if [[ "${RUN_MIGRATIONS:-1}" != "1" ]]; then
        echo "[entrypoint] RUN_MIGRATIONS=0, skipping alembic upgrade"
        return 0
    fi
    echo "[entrypoint] applying alembic migrations"
    alembic upgrade head
}

wait_for_db
run_migrations

echo "[entrypoint] exec: $*"
exec "$@"
