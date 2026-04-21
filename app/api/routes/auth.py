"""Authentication routes.

Public endpoints (no bearer required):

* ``POST /login`` - exchanges email + password for a Keycloak access token
  via the OIDC password grant. Credentials never touch the database; we
  only proxy them to Keycloak's token endpoint.

Protected endpoints (require a valid bearer token):

* ``GET /me`` - any authenticated user (returns :class:`CurrentUser`).
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.core.config import settings
from app.core.models import CurrentUser
from app.core.security import get_current_user


logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# --- schemas ----------------------------------------------------------------


class LoginRequest(BaseModel):
    """Credentials accepted by :func:`login`.

    ``email`` is forwarded to Keycloak as the ``username`` form field of
    the password grant; the realm must be configured to accept email as
    a login identifier (the default when "Login with email" is enabled).
    """

    email: EmailStr
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    """Subset of Keycloak's token endpoint response that we pass through.

    ``model_config = {"extra": "allow"}`` keeps any additional fields
    Keycloak may add (e.g. ``not-before-policy``, ``session_state``) so
    clients can use them without another server change.
    """

    access_token: str
    expires_in: int
    refresh_expires_in: int | None = None
    refresh_token: str | None = None
    token_type: str
    scope: str | None = None

    model_config = {"extra": "allow"}


# --- routes -----------------------------------------------------------------


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    """Exchange email + password for a Keycloak access token.

    Performs the OAuth2 Resource Owner Password Credentials grant against
    the configured Keycloak realm. Invalid credentials surface as HTTP
    401; transport / upstream failures surface as HTTP 502.
    """
    form = {
        "grant_type": "password",
        "client_id": settings.keycloak_client_id,
        "client_secret": settings.keycloak_client_secret,
        "username": payload.email,
        "password": payload.password,
        "scope": "openid",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                settings.keycloak_token_endpoint,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.HTTPError as exc:
        logger.exception("Keycloak token endpoint unreachable")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Auth provider is unreachable.",
        ) from exc

    if resp.status_code == 200:
        return TokenResponse.model_validate(resp.json())

    # Keycloak returns 400/401 for bad credentials with an
    # ``error``/``error_description`` body; normalise to 401 for the
    # caller but keep the upstream description for debuggability.
    if resp.status_code in (400, 401):
        try:
            body = resp.json()
        except ValueError:
            body = {}
        detail = body.get("error_description") or body.get("error") or "Invalid credentials."
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": 'Bearer realm="api"'},
        )

    logger.warning(
        "Unexpected status %s from Keycloak token endpoint: %s",
        resp.status_code,
        resp.text[:500],
    )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Unexpected response from auth provider.",
    )


@router.get("/me", response_model=CurrentUser)
def me(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Return the caller's identity as reconstructed from their JWT."""
    return user
