"""Keycloak-backed OAuth2 resource server helpers.

This module wires the FastAPI application up as a pure resource server:

* It fetches the realm's JWKS on startup and caches the keys in memory.
* It validates incoming bearer JWTs locally using ``python-jose``
  (signature, ``exp``, ``iss``, ``azp``). No calls to Keycloak's
  introspection endpoint are made on the request path.
* It exposes :func:`get_current_user` and :func:`require_role` as
  FastAPI dependencies so route handlers can opt into authentication
  and authorisation.

Key rotation is handled transparently: if a token references a ``kid``
that is not in the cache we refresh the JWKS once and retry the lookup.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

from app.core.config import settings
from app.core.models import CurrentUser


logger = logging.getLogger(__name__)

# The Keycloak realm signs access tokens with RS256; restrict the accepted
# algorithms to avoid ``alg: none`` downgrade attacks.
_ALLOWED_ALGORITHMS: tuple[str, ...] = ("RS256",)

# ``WWW-Authenticate`` must be returned on 401 responses per RFC 6750 so
# compliant clients know this is a bearer-token challenge.
_BEARER_CHALLENGE = {"WWW-Authenticate": 'Bearer realm="api"'}


class _JWKSCache:
    """Thread-safe in-memory cache of Keycloak's JSON Web Key Set.

    Keys are indexed by ``kid`` for O(1) lookup. The cache is populated
    lazily on first access and can be force-refreshed when a token
    references an unknown ``kid`` (typical during key rotation).
    """

    def __init__(self, jwks_uri: str, timeout: float = 5.0) -> None:
        self._jwks_uri = jwks_uri
        self._timeout = timeout
        self._keys: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    async def refresh(self) -> None:
        """Fetch the JWKS from Keycloak and replace the cached keys."""
        logger.info("JWKS: fetching from %s", self._jwks_uri)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._jwks_uri)
            resp.raise_for_status()
            payload = resp.json()

        new_keys = {k["kid"]: k for k in payload.get("keys", []) if "kid" in k}
        if not new_keys:
            raise RuntimeError(
                f"JWKS at {self._jwks_uri} returned no keys with a 'kid'."
            )

        with self._lock:
            self._keys = new_keys
        logger.info("JWKS: cached %d key(s): %s", len(new_keys), list(new_keys))

    async def get_key(self, kid: str) -> dict[str, Any] | None:
        """Return the JWK for ``kid``, refreshing the cache once on miss."""
        with self._lock:
            key = self._keys.get(kid)
        if key is not None:
            return key

        # Unknown kid -> refresh once (Keycloak may have rotated keys).
        logger.info("JWKS: kid %s not in cache, refreshing", kid)
        await self.refresh()
        with self._lock:
            return self._keys.get(kid)


_jwks_cache = _JWKSCache(jwks_uri=settings.keycloak_jwks_uri)

# ``auto_error=False`` lets us raise our own 401 with the WWW-Authenticate
# header that RFC 6750 mandates instead of FastAPI's default 403.
_bearer_scheme = HTTPBearer(auto_error=False, bearerFormat="JWT")


async def init_jwks_cache() -> None:
    """Populate the in-memory JWKS cache.

    Intended to be called from the FastAPI ``lifespan`` handler so the
    first authenticated request does not pay the JWKS round-trip.
    """
    await _jwks_cache.refresh()


def _unauthorized(detail: str) -> HTTPException:
    """Build a 401 with the bearer challenge header pre-populated."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_BEARER_CHALLENGE,
    )


async def validate_token(token: str) -> dict[str, Any]:
    """Validate a Keycloak-issued JWT and return its decoded claims.

    Performs the following checks:

    * Header ``kid`` resolves to a cached JWK (with one retry after a
      cache refresh to handle key rotation).
    * Signature is valid under RS256 using the resolved JWK.
    * ``exp`` is in the future.
    * ``iss`` equals :data:`settings.keycloak_issuer`.
    * ``azp`` equals :data:`settings.keycloak_client_id`.

    Any failure raises :class:`fastapi.HTTPException` with status 401
    and a ``WWW-Authenticate: Bearer`` header.
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise _unauthorized(f"Malformed token header: {exc}") from exc

    kid = unverified_header.get("kid")
    if not kid:
        raise _unauthorized("Token header missing 'kid'.")

    try:
        key = await _jwks_cache.get_key(kid)
    except httpx.HTTPError as exc:
        logger.exception("JWKS refresh failed")
        raise _unauthorized("Unable to refresh signing keys.") from exc

    if key is None:
        raise _unauthorized("Signing key not found for token 'kid'.")

    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=list(_ALLOWED_ALGORITHMS),
            issuer=settings.keycloak_issuer,
            # We validate ``azp`` explicitly below. Keycloak tokens for
            # direct-grant clients typically omit the ``aud`` claim, so
            # disable the aud check here rather than forcing a value.
            options={"verify_aud": False},
        )
    except ExpiredSignatureError as exc:
        raise _unauthorized("Token has expired.") from exc
    except JWTError as exc:
        raise _unauthorized(f"Invalid token: {exc}") from exc

    azp = claims.get("azp")
    if azp != settings.keycloak_client_id:
        raise _unauthorized(
            f"Token 'azp' mismatch: expected {settings.keycloak_client_id!r}, "
            f"got {azp!r}."
        )

    return claims


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> CurrentUser:
    """FastAPI dependency that returns the authenticated :class:`CurrentUser`.

    Extracts the bearer token from the ``Authorization`` header, validates
    it locally and maps the claims onto :class:`CurrentUser`. Raises 401
    when the header is missing, malformed, or the token fails validation.
    """
    if credentials is None or (credentials.scheme or "").lower() != "bearer":
        raise _unauthorized("Missing or invalid Authorization header.")

    claims = await validate_token(credentials.credentials)

    try:
        return CurrentUser(
            sub=claims["sub"],
            email=claims.get("email", ""),
            preferred_username=claims.get("preferred_username", ""),
            name=claims.get("name", ""),
            given_name=claims.get("given_name", ""),
            family_name=claims.get("family_name", ""),
            gender=claims.get("gender", "N/A"),
            email_verified=bool(claims.get("email_verified", False)),
            # Roles MUST come from the top-level "roles" claim (ROLE_*),
            # never from "realm_access.roles" which holds unprefixed names.
            roles=list(claims.get("roles", []) or []),
            sid=claims.get("sid", ""),
        )
    except KeyError as exc:
        raise _unauthorized(f"Token missing required claim: {exc.args[0]}") from exc


def require_role(*roles: str):
    """Dependency factory returning a guard that requires ANY of ``roles``.

    The returned dependency re-uses :func:`get_current_user` so the token
    is validated exactly once per request. If the caller holds none of
    the requested roles an HTTP 403 is raised.

    Usage::

        @router.get("/admin", dependencies=[Depends(require_role("ROLE_ADMIN"))])
        def admin_only() -> ...:
            ...

        @router.get("/staff")
        def staff(
            user: CurrentUser = Depends(
                require_role("ROLE_INSTRUCTOR", "ROLE_ADMIN")
            ),
        ) -> ...:
            ...
    """
    required = set(roles)

    async def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if required.isdisjoint(user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Insufficient role. Required one of: "
                    f"{sorted(required)}; token has: {sorted(user.roles)}."
                ),
            )
        return user

    return _checker
