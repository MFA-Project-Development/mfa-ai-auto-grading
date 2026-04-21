"""Reusable HTTP client for the Phase-One Assessment API.

This module is the only place in the app that talks to ``httpx`` for
outbound integrations (the Keycloak JWKS fetch in
:mod:`app.core.security` is intentionally kept separate because it runs
inside the auth hot-path and has its own caching semantics).

Design notes
------------
* Both an async (:class:`AssessmentAPIClient`) and a thin sync wrapper
  (:class:`AssessmentAPIClientSync`) are provided. Prefer the async
  variant from FastAPI route handlers and background tasks; use the
  sync variant only from scripts / management commands.
* Defaults are sourced from :data:`app.core.config.settings` but every
  knob can be overridden per instance, which makes the client trivial
  to reuse against multiple upstreams.
* Transient failures (network errors, ``429``, and ``5xx``) are retried
  with exponential backoff. ``4xx`` (other than ``429``) fail fast so
  callers see programming errors immediately.
* All low-level ``httpx`` errors are wrapped in the local exception
  hierarchy from :mod:`.exceptions` so callers never import ``httpx``
  to handle failures.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any, Mapping

import httpx

from app.core.config import Settings, settings as _default_settings
from app.services.assessment_client.exceptions import (
    AssessmentAPIDecodeError,
    AssessmentAPIRequestError,
    AssessmentAPIStatusError,
    AssessmentAPITimeoutError,
)

logger = logging.getLogger(__name__)

# Status codes we consider transient and therefore retry. ``429`` is
# included so rate-limited callers get a best-effort retry without
# surfacing the failure; ``5xx`` covers the usual "try again" upstream
# errors.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None


class AssessmentAPIClient:
    """Async HTTP client with retries, timeouts, and typed errors.

    The client can be used either as a short-lived per-call helper or
    as a long-lived instance shared across requests::

        client = AssessmentAPIClient()
        data = await client.get_json("/users/42")

        # Or as an async context manager to guarantee connection reuse
        # within a scope:
        async with AssessmentAPIClient() as client:
            data = await client.get_json("/users/42")
            await client.post_json("/users", json={"name": "Ada"})

    Parameters override :data:`app.core.config.settings` defaults. Pass
    ``base_url=""`` when you want to call absolute URLs exclusively.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        auth_scheme: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        default_headers: Mapping[str, str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or _default_settings
        self._base_url = (
            base_url if base_url is not None else cfg.assessment_api_base_url
        )
        self._api_key = api_key if api_key is not None else cfg.assessment_api_key
        self._auth_scheme = (
            auth_scheme
            if auth_scheme is not None
            else cfg.assessment_api_auth_scheme
        ).lower()
        self._timeout = (
            timeout if timeout is not None else cfg.assessment_api_timeout_seconds
        )
        self._max_retries = (
            max_retries
            if max_retries is not None
            else cfg.assessment_api_max_retries
        )
        self._backoff = (
            backoff_seconds
            if backoff_seconds is not None
            else cfg.assessment_api_backoff_seconds
        )

        # Default headers are merged with per-call headers; per-call
        # values win on conflict.
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "mfa-auto-grading/0.1",
        }
        if default_headers:
            headers.update(default_headers)
        headers.update(self._auth_header())
        self._default_headers = headers

        # The underlying ``httpx.AsyncClient`` is created lazily so the
        # constructor stays cheap and safe to call outside an event
        # loop (e.g. at module import time).
        self._client: httpx.AsyncClient | None = None
        self._owns_client = True

    # ------------------------------------------------------------------ lifecycle

    async def __aenter__(self) -> "AssessmentAPIClient":
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying connection pool."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ verbs

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JSONType:
        """GET ``path`` and return the decoded JSON body."""
        return await self._request_json("GET", path, params=params, headers=headers)

    async def post_json(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JSONType:
        """POST a JSON body and return the decoded JSON response."""
        return await self._request_json(
            "POST", path, json=json, params=params, headers=headers
        )

    async def put_json(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JSONType:
        """PUT a JSON body and return the decoded JSON response."""
        return await self._request_json(
            "PUT", path, json=json, params=params, headers=headers
        )

    async def patch_json(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JSONType:
        """PATCH a JSON body and return the decoded JSON response."""
        return await self._request_json(
            "PATCH", path, json=json, params=params, headers=headers
        )

    async def delete(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JSONType | None:
        """DELETE ``path``. Returns the decoded body or ``None`` on 204."""
        response = await self._request("DELETE", path, params=params, headers=headers)
        if response.status_code == 204 or not response.content:
            return None
        return self._decode_json(response)

    async def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Escape hatch for callers that need the raw ``httpx.Response``.

        Retries and exception wrapping still apply. Use this when you
        need non-JSON responses (binary downloads, streaming, etc.).
        """
        return await self._request(method, path, **kwargs)

    async def request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> JSONType:
        """Generic JSON-in / JSON-out escape hatch.

        Prefer the verb-specific helpers (``get_json`` / ``post_json``
        / ``put_json`` / ``patch_json``) when the method is known up
        front; this one exists for routing layers that forward an
        arbitrary verb to the upstream.
        """
        return await self._request_json(method, path, **kwargs)

    # ------------------------------------------------------------------ internals

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._default_headers,
            )
        return self._client

    def _auth_header(self) -> dict[str, str]:
        if not self._api_key or self._auth_scheme == "none":
            return {}
        if self._auth_scheme == "bearer":
            return {"Authorization": f"Bearer {self._api_key}"}
        if self._auth_scheme in {"x-api-key", "apikey"}:
            return {"X-API-Key": self._api_key}
        logger.warning(
            "assessment_client: unknown auth scheme %r, falling back to Bearer",
            self._auth_scheme,
        )
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> httpx.Response:
        """Issue the HTTP call with retry/backoff and error wrapping."""
        client = self._ensure_client()
        merged_headers = dict(headers) if headers else None

        attempt = 0
        while True:
            attempt += 1
            try:
                response = await client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers=merged_headers,
                    **extra,
                )
            except httpx.TimeoutException as exc:
                if self._should_retry(attempt):
                    await self._sleep_backoff(attempt)
                    continue
                raise AssessmentAPITimeoutError(
                    f"{method} {path} timed out after {self._timeout}s"
                ) from exc
            except httpx.RequestError as exc:
                if self._should_retry(attempt):
                    await self._sleep_backoff(attempt)
                    continue
                raise AssessmentAPIRequestError(
                    f"{method} {path} failed: {exc}"
                ) from exc

            if response.status_code in _RETRYABLE_STATUS and self._should_retry(
                attempt
            ):
                logger.info(
                    "assessment_client: %s %s returned %s, retrying (attempt %d/%d)",
                    method,
                    path,
                    response.status_code,
                    attempt,
                    self._max_retries,
                )
                await self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise self._status_error(response)

            return response

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> JSONType:
        response = await self._request(method, path, **kwargs)
        return self._decode_json(response)

    def _should_retry(self, attempt: int) -> bool:
        return attempt <= self._max_retries

    async def _sleep_backoff(self, attempt: int) -> None:
        # Exponential backoff: base * 2^(attempt-1). Capped at 30s so a
        # mis-configured upstream cannot park a worker forever.
        delay = min(self._backoff * (2 ** (attempt - 1)), 30.0)
        await asyncio.sleep(delay)

    @staticmethod
    def _decode_json(response: httpx.Response) -> JSONType:
        try:
            return response.json()
        except ValueError as exc:
            raise AssessmentAPIDecodeError(
                f"Response from {response.request.url} was not valid JSON: {exc}"
            ) from exc

    @staticmethod
    def _status_error(response: httpx.Response) -> AssessmentAPIStatusError:
        try:
            body: object = response.json()
        except ValueError:
            body = response.text
        return AssessmentAPIStatusError(
            f"{response.request.method} {response.request.url} "
            f"returned HTTP {response.status_code}",
            status_code=response.status_code,
            url=str(response.request.url),
            body=body,
        )


class AssessmentAPIClientSync:
    """Blocking counterpart of :class:`AssessmentAPIClient`.

    Provided for CLI scripts and Alembic hooks that cannot easily host
    an event loop. Request semantics, retries, and error wrapping are
    identical to the async client.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._async = AssessmentAPIClient(**kwargs)

    def __enter__(self) -> "AssessmentAPIClientSync":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        asyncio.run(self._async.aclose())

    def get_json(self, path: str, **kwargs: Any) -> JSONType:
        return asyncio.run(self._async.get_json(path, **kwargs))

    def post_json(self, path: str, **kwargs: Any) -> JSONType:
        return asyncio.run(self._async.post_json(path, **kwargs))

    def put_json(self, path: str, **kwargs: Any) -> JSONType:
        return asyncio.run(self._async.put_json(path, **kwargs))

    def patch_json(self, path: str, **kwargs: Any) -> JSONType:
        return asyncio.run(self._async.patch_json(path, **kwargs))

    def delete(self, path: str, **kwargs: Any) -> JSONType | None:
        return asyncio.run(self._async.delete(path, **kwargs))


@lru_cache(maxsize=1)
def get_assessment_client() -> AssessmentAPIClient:
    """Return a process-wide cached :class:`AssessmentAPIClient`.

    Use this from FastAPI dependencies when you want connection pooling
    across requests. The FastAPI ``lifespan`` should call
    ``await client.aclose()`` on shutdown if you take this route.
    """
    return AssessmentAPIClient()
