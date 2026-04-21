"""Shared Pydantic models used by the API layer.

At the moment this module only exposes :class:`CurrentUser`, the
representation of the authenticated principal extracted from a Keycloak
access token. Keeping it in ``app.core`` avoids a circular import between
the security layer and the various route modules that consume it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CurrentUser(BaseModel):
    """Authenticated principal derived from a Keycloak access token.

    The field set mirrors the claims produced by our Keycloak realm. In
    particular ``roles`` is populated from the **top-level** ``roles``
    claim (which carries the ``ROLE_`` prefix), *not* from
    ``realm_access.roles``.
    """

    sub: str
    email: str
    preferred_username: str
    name: str
    given_name: str
    family_name: str
    gender: str = "N/A"
    email_verified: bool = False
    roles: list[str] = Field(default_factory=list)
    sid: str
