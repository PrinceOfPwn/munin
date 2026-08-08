# tags: [core, corvus, transport, redis, upstash, red-contract, tdd, token-hygiene]
"""Corvus Redis REST transport — the external operative store.

GREEN implementation for ``tests/test_corvus_transport.py``. A Corvus
transport is an *external operative store* in the DeepAgents sense: durable
operational state lives OUTSIDE the ephemeral LangGraph graph state (a
``BaseStore``-backed ``StoreBackend``, contrasted with the in-graph
``StateBackend``). Corvus therefore owns no graph state and keeps its
credential off every observable surface (``repr``/``str``, error messages,
exceptions).

Wire contract (Upstash Redis REST API, verified via Context7):

* Single commands ``POST`` a JSON array to the normalized base URL.
* ``/pipeline`` and ``/multi-exec`` accept a two-dimensional JSON array body.
* Batch responses are ordered lists of ``{"result": ...}`` or
  ``{"error": ...}`` items.
* Requests authenticate with ``Authorization: Bearer <token>``.

Only ``requests`` is used; no SQLite, no ``SharedStateStore``, no graph state.
"""

from __future__ import annotations

import abc
import math
import os
from collections.abc import Mapping
from typing import Any

import requests

ENV_URL = "UPSTASH_REDIS_REST_URL"
ENV_TOKEN = "UPSTASH_REDIS_REST_TOKEN"
_REDACTED = "[REDACTED]"
_MAX_ERROR_LEN = 300

__all__ = [
    "CorvusConfigurationError",
    "CorvusTransportError",
    "RedisTransport",
    "UpstashRedisRestTransport",
]


class CorvusTransportError(Exception):
    """Sanitized Corvus transport failure; never carries credentials."""


class CorvusConfigurationError(CorvusTransportError):
    """Invalid or missing Corvus transport configuration."""


class RedisTransport(abc.ABC):
    """Contract for a Corvus Redis transport (external operative store)."""

    @abc.abstractmethod
    def command(self, *parts: Any) -> Any:
        """Execute a single Redis command and return the unwrapped result."""

    @abc.abstractmethod
    def pipeline(self, commands: list[list[Any]], atomic: bool = False) -> list[Any]:
        """Execute ordered commands in one batch (optionally atomic)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying session; idempotent and non-destructive."""


class UpstashRedisRestTransport(RedisTransport):
    """Upstash Redis REST transport backed by a plain ``requests.Session``.

    The session is injected for testability; the credential is held only in a
    private attribute and never serialized into ``repr``/``str`` or into any
    raised ``CorvusTransportError`` message.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        session: requests.Session | None = None,
        timeout: float = 5.0,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized:
            raise CorvusConfigurationError("base_url must not be empty")
        if not token:
            raise CorvusConfigurationError("token must not be empty")
        try:
            finite_timeout = float(timeout)
        except (TypeError, ValueError):
            raise CorvusConfigurationError("timeout must be a finite, positive number") from None
        if not math.isfinite(finite_timeout) or finite_timeout <= 0:
            raise CorvusConfigurationError("timeout must be a finite, positive number")

        self._base_url = normalized
        self._token = token
        self._timeout = finite_timeout
        self._session = session if session is not None else requests.Session()
        self._closed = False

    # ------------------------------------------------------------------
    # Factory — strict from-env with no SQLite fallback.
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        session: requests.Session | None = None,
        timeout: float = 5.0,
        environ: Mapping[str, str] | None = None,
    ) -> UpstashRedisRestTransport:
        """Build a transport from ``UPSTASH_REDIS_REST_URL`` and ``_TOKEN``.

        Both variables are mandatory and there is **no** SQLite fallback: a
        missing variable raises ``CorvusConfigurationError`` naming it. ``environ``
        is injected for tests and defaults to ``os.environ``.
        """
        env = environ if environ is not None else os.environ
        missing = [name for name in (ENV_URL, ENV_TOKEN) if not env.get(name)]
        if missing:
            raise CorvusConfigurationError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )
        return cls(
            base_url=env[ENV_URL],
            token=env[ENV_TOKEN],
            session=session if session is not None else requests.Session(),
            timeout=timeout,
        )

    # -- single command ----------------------------------------------------

    def command(self, *parts: Any) -> Any:
        payload = self._post("", list(parts))
        return self._unwrap_single(payload, "command")

    # -- ordered pipeline -----------------------------------------------------

    def pipeline(self, commands: list[list[Any]], atomic: bool = False) -> list[Any]:
        if self._closed:
            raise CorvusTransportError("Corvus transport is closed")
        if not commands:
            return []
        endpoint = "/multi-exec" if atomic else "/pipeline"
        payload = self._post(endpoint, commands)
        if not isinstance(payload, list):
            raise CorvusTransportError("Corvus pipeline response is not a JSON array")
        if len(payload) != len(commands):
            raise CorvusTransportError(
                f"Corvus pipeline response cardinality mismatch: "
                f"expected {len(commands)} items, got {len(payload)}"
            )
        results: list[Any] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise CorvusTransportError(f"Corvus pipeline item {index} is not a JSON object")
            if "error" in item:
                raise CorvusTransportError(self._redact(f"Corvus pipeline item {index} error: {item['error']}"))
            if "result" in item:
                results.append(item["result"])
            else:
                raise CorvusTransportError(
                    f"Corvus pipeline item {index} has neither 'result' nor 'error'"
                )
        return results

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close()

    # -- representation ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(configured=True, "
            f"timeout={self._timeout!r}, closed={self._closed})"
        )

    __str__ = __repr__

    # -- internals --------------------------------------------------------------

    def _post(self, endpoint: str, body: Any) -> Any:
        if self._closed:
            raise CorvusTransportError("Corvus transport is closed")
        url = self._base_url if not endpoint else f"{self._base_url}{endpoint}"
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            response = self._session.post(url, json=body, headers=headers, timeout=self._timeout)
        except requests.exceptions.RequestException:
            raise CorvusTransportError(
                self._redact("Corvus session failure")
            ) from None

        try:
            response.raise_for_status()
        except requests.exceptions.RequestException:
            raise CorvusTransportError(self._redact("Corvus HTTP error")) from None

        try:
            return response.json()
        except (requests.exceptions.RequestException, ValueError):
            raise CorvusTransportError(self._redact("Corvus invalid JSON response")) from None

    def _unwrap_single(self, payload: Any, verb: str) -> Any:
        if not isinstance(payload, dict):
            raise CorvusTransportError(f"Corvus {verb} response is not a JSON object")
        if "error" in payload:
            raise CorvusTransportError(self._redact(f"Corvus {verb} error: {payload['error']}"))
        if "result" in payload:
            return payload["result"]
        raise CorvusTransportError(f"Corvus {verb} response has neither 'result' nor 'error'")

    def _redact(self, message: str) -> str:
        """Bounded, token-free message for a ``CorvusTransportError``."""
        if self._token and self._token in message:
            message = message.replace(self._token, _REDACTED)
        if len(message) > _MAX_ERROR_LEN:
            message = message[:_MAX_ERROR_LEN] + "...[truncated]"
        return message or _REDACTED