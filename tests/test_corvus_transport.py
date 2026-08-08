# tags: [tests, core, corvus, transport, redis, upstash, red-contract, tdd, token-hygiene]
"""RED contract tests for ``munin.corvus.transport`` — the Corvus transport.

RED phase: exactly one test file, no production code. The target module
``munin.corvus.transport`` is imported *normally* at module top, so GitHub
Actions captures RED (a non-zero pytest exit) while ``RedisTransport``,
``UpstashRedisRestTransport``, ``CorvusTransportError`` and
``CorvusConfigurationError`` are still missing.

Decision-complete contract (from the ticket):

1. ``from_env()`` requires BOTH ``UPSTASH_REDIS_REST_URL`` and
   ``UPSTASH_REDIS_REST_TOKEN`` and has **no SQLite fallback**: a missing
   variable raises ``CorvusConfigurationError`` naming the missing variable.
2. The base URL's trailing slash is normalized away; the token never appears
   in ``repr``/``str`` or in any ``CorvusTransportError`` message.
3. ``command(*parts)`` POSTs a JSON array to the normalized base URL with
   ``Authorization: Bearer <token>`` and a configurable finite timeout, and
   returns the unwrapped value from ``{result: ...}``.
4. ``pipeline(commands, atomic=False)`` POSTs a 2D command array to
   ``/pipeline`` and unwraps the ordered results.
5. ``pipeline(commands, atomic=True)`` POSTs the same 2D array to
   ``/multi-exec`` and unwraps the ordered results.
6. Session failures, HTTP error statuses, invalid JSON, a top-level Redis
   ``{error: ...}`` envelope and a per-item pipeline ``{error: ...}`` all
   become a sanitized ``CorvusTransportError`` with no token leakage.
7. An empty pipeline returns ``[]`` without issuing any HTTP request.
8. ``close()`` is idempotent, and a closed transport rejects further work
   with ``CorvusTransportError``.

Research evidence:

* DeepWiki on ``langchain-ai/deepagents``: durable operational stores live
  OUTSIDE the LangGraph graph state — DeepAgents separates the ephemeral
  in-graph ``StateBackend`` from the persistent external ``StoreBackend``
  (backed by an external ``BaseStore``, e.g. Redis). Corvus transport is
  exactly such an external operative store, so it owns no graph state and
  keeps credentials off every observable surface.
* Context7 on ``/websites/upstash_redis`` (Upstash Redis REST API): single
  commands POST a JSON array to the base URL; ``/pipeline`` and
  ``/multi-exec`` both accept a two-dimensional JSON array body; both batch
  responses return an ordered list of ``{"result": ...}`` or
  ``{"error": ...}`` items; requests authenticate with
  ``Authorization: Bearer <token>``.

The tests use a tiny fake ``requests.Session``/``Response`` defined inside
this file (``requests`` is already a Munin dependency). No dependency or
production change is made.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import pytest
import requests

from munin.corvus.transport import (
    CorvusConfigurationError,
    CorvusTransportError,
    RedisTransport,
    UpstashRedisRestTransport,
)

ENV_URL = "UPSTASH_REDIS_REST_URL"
ENV_TOKEN = "UPSTASH_REDIS_REST_TOKEN"
BASE_URL = "https://redis.example.com"
BASE_URL_TRAILING = "https://redis.example.com/"
TOKEN = "corvus-test-token-that-must-never-leak"  # noqa: S105 - inert test credential


# ---------------------------------------------------------------------------
# Tiny fake requests.Session / requests.Response — no new dependencies.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Duck-typed ``requests.Response``: scripted payload/status/noise."""

    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.invalid_json = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTPError: {self.status_code} for {self.text!r}"
            )

    def json(self) -> Any:
        if self.invalid_json:
            raise requests.exceptions.JSONDecodeError("Expecting value", self.text, 0)
        return self.payload


class _BrokenJsonResponse(_FakeResponse):
    """Response whose body cannot be decoded as JSON."""

    def json(self) -> Any:
        raise requests.exceptions.JSONDecodeError("Expecting value", "not-json", 0)


class _FakeSession:
    """Records POST calls and serves ``Callable[[dict], _FakeResponse]``."""

    def __init__(self, responder: Callable[[dict[str, Any]], _FakeResponse] | None = None) -> None:
        self.posts: list[dict[str, Any]] = []
        self.close_calls = 0
        self._responder = responder or (
            lambda _: _FakeResponse(_default_result())
        )

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        call = {"url": url}
        call.update(kwargs)
        self.posts.append(call)
        return self._responder(call)

    def close(self) -> None:
        self.close_calls += 1


def _default_result() -> dict[str, str]:
    return {"result": "OK"}


def _transport(
    *,
    session: _FakeSession,
    base_url: str = BASE_URL,
    token: str = TOKEN,
    timeout: float = 5.0,
) -> UpstashRedisRestTransport:
    """Build the concrete transport with the injected fake session."""
    return UpstashRedisRestTransport(
        base_url=base_url, token=token, session=session, timeout=timeout
    )


# ---------------------------------------------------------------------------
# 1. from_env: both vars mandatory, no SQLite fallback.
# ---------------------------------------------------------------------------


def test_from_env_requires_both_upstash_vars_and_has_no_sqlite_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_URL, raising=False)
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    with pytest.raises(CorvusConfigurationError) as excinfo:
        UpstashRedisRestTransport.from_env(session=_FakeSession())
    message = str(excinfo.value)
    # No SQLite fallback: a missing variable must fail loudly and name it.
    assert ENV_URL in message
    assert ENV_TOKEN in message


def test_from_env_reports_the_single_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_URL, BASE_URL)
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    with pytest.raises(CorvusConfigurationError) as excinfo:
        _ = UpstashRedisRestTransport.from_env(session=_FakeSession())
    assert ENV_TOKEN in str(excinfo.value)

    monkeypatch.delenv(ENV_URL, raising=False)
    monkeypatch.setenv(ENV_TOKEN, TOKEN)
    with pytest.raises(CorvusConfigurationError) as excinfo:
        _ = UpstashRedisRestTransport.from_env(session=_FakeSession())
    assert ENV_URL in str(excinfo.value)


def test_from_env_builds_a_working_transport_when_both_vars_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()
    monkeypatch.setenv(ENV_URL, BASE_URL_TRAILING)
    monkeypatch.setenv(ENV_TOKEN, TOKEN)
    transport = UpstashRedisRestTransport.from_env(session=session, timeout=3.0)
    assert isinstance(transport, UpstashRedisRestTransport)
    transport.command("PING")
    assert session.posts[0]["url"] == BASE_URL
    assert session.posts[0]["timeout"] == 3.0


# ---------------------------------------------------------------------------
# 2. URL normalization + cred-protected repr/str.
# ---------------------------------------------------------------------------


def test_command_uses_normalized_base_url_without_trailing_slash() -> None:
    session = _FakeSession()
    transport = _transport(session=session, base_url=BASE_URL_TRAILING)
    transport.command("PING")
    assert session.posts[0]["url"] == BASE_URL


def test_repr_and_str_never_expose_the_token() -> None:
    session = _FakeSession()
    transport = _transport(session=session)
    assert TOKEN not in repr(transport)
    assert TOKEN not in str(transport)


# ---------------------------------------------------------------------------
# 3. command(): JSON array, Bearer auth, finite timeout, {result: ...}.
# ---------------------------------------------------------------------------


def test_command_posts_json_array_with_bearer_auth_and_finite_timeout() -> None:
    session = _FakeSession(responder=lambda _: _FakeResponse({"result": "ok-from-redis"}))
    transport = _transport(session=session)
    result = transport.command("SET", "topic/corvus", "hello")
    assert result == "ok-from-redis"

    call = session.posts[0]
    assert call["url"] == BASE_URL
    assert call["json"] == ["SET", "topic/corvus", "hello"]
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    timeout = call["timeout"]
    assert timeout is not None
    assert 0 < float(timeout) < math.inf


def test_command_timeout_is_configurable_and_still_finite() -> None:
    session = _FakeSession()
    transport = _transport(session=session, timeout=12.5)
    transport.command("GET", "k")
    assert session.posts[0]["timeout"] == 12.5
    assert math.isfinite(float(session.posts[0]["timeout"]))


# ---------------------------------------------------------------------------
# 4 + 5. pipeline(): 2D array to /pipeline (non-atomic) or /multi-exec.
# ---------------------------------------------------------------------------


def test_pipeline_non_atomic_posts_to_pipeline_and_unwraps_ordered_results() -> None:
    session = _FakeSession(
        responder=lambda _: _FakeResponse([{"result": "OK"}, {"result": 7}])
    )
    transport = _transport(session=session)
    results = transport.pipeline([["SET", "k", "v"], ["INCR", "k"]], atomic=False)
    assert results == ["OK", 7]
    call = session.posts[0]
    assert call["url"] == f"{BASE_URL}/pipeline"
    assert call["json"] == [["SET", "k", "v"], ["INCR", "k"]]
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_pipeline_atomic_posts_to_multi_exec_and_unwraps_ordered_results() -> None:
    session = _FakeSession(
        responder=lambda _: _FakeResponse([{"result": "OK"}, {"result": "OK"}])
    )
    transport = _transport(session=session, base_url=BASE_URL_TRAILING)
    results = transport.pipeline([["SET", "a", "1"], ["SET", "b", "2"]], atomic=True)
    assert results == ["OK", "OK"]
    call = session.posts[0]
    # No double slash between the normalized base and the /multi-exec suffix.
    assert call["url"] == f"{BASE_URL}/multi-exec"
    assert call["json"] == [["SET", "a", "1"], ["SET", "b", "2"]]


# ---------------------------------------------------------------------------
# 6. All failure modes become sanitized CorvusTransportError, no token.
# ---------------------------------------------------------------------------


def test_session_failure_becomes_sanitized_error() -> None:
    session = _FakeSession(
        responder=lambda _: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("backend down")
        )
    )
    transport = _transport(session=session)
    with pytest.raises(CorvusTransportError) as excinfo:
        transport.command("GET", "k")
    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in repr(excinfo.value)


def test_http_error_status_becomes_sanitized_error() -> None:
    # A 503 with a success-looking body forces the transport to honor the
    # session's raise_for_status() contract instead of trusting the payload.
    session = _FakeSession(
        responder=lambda _: _FakeResponse({"result": "surprise-success"}, status_code=503)
    )
    transport = _transport(session=session)
    with pytest.raises(CorvusTransportError) as excinfo:
        transport.command("GET", "k")
    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in repr(excinfo.value)


def test_invalid_json_becomes_sanitized_error() -> None:
    session = _FakeSession(responder=lambda _: _BrokenJsonResponse(None))
    transport = _transport(session=session)
    with pytest.raises(CorvusTransportError) as excinfo:
        transport.command("GET", "k")
    assert TOKEN not in str(excinfo.value)


def test_top_level_redis_error_becomes_sanitized_error() -> None:
    session = _FakeSession(responder=lambda _: _FakeResponse({"error": "ERR invalid command"}))
    transport = _transport(session=session)
    with pytest.raises(CorvusTransportError) as excinfo:
        transport.command("GET", "k")
    assert "ERR invalid command" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in repr(excinfo.value)


def test_per_item_pipeline_error_becomes_sanitized_error() -> None:
    session = _FakeSession(
        responder=lambda _: _FakeResponse(
            [{"result": "OK"}, {"error": "ERR value is not an int or out of range"}]
        )
    )
    transport = _transport(session=session)
    with pytest.raises(CorvusTransportError) as excinfo:
        transport.pipeline([["SET", "k", "v"], ["INCR", "k"]], atomic=False)
    assert "ERR value is not an int or out of range" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in repr(excinfo.value)


# ---------------------------------------------------------------------------
# 7. Empty pipeline: [] without any HTTP request.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atomic", [False, True])
def test_empty_pipeline_returns_empty_list_without_http(atomic: bool) -> None:
    session = _FakeSession()
    transport = _transport(session=session)
    assert transport.pipeline([], atomic=atomic) == []
    assert session.posts == []


# ---------------------------------------------------------------------------
# 8. close() idempotent; closed transport rejects further work.
# ---------------------------------------------------------------------------


def test_close_is_idempotent_and_closed_transport_rejects_work() -> None:
    session = _FakeSession()
    transport = _transport(session=session)
    transport.command("PING")
    assert session.posts, "transport must work before close"

    transport.close()
    transport.close()
    assert session.close_calls == 1

    with pytest.raises(CorvusTransportError):
        transport.command("GET", "k")
    with pytest.raises(CorvusTransportError):
        transport.pipeline([["GET", "k"]], atomic=False)


# ---------------------------------------------------------------------------
# Minimal structural sanity — RED imports resolve to the right hierarchy.
# ---------------------------------------------------------------------------


def test_transport_symbols_form_the_minimal_public_hierarchy() -> None:
    assert issubclass(UpstashRedisRestTransport, RedisTransport)
    assert issubclass(CorvusConfigurationError, CorvusTransportError)
    assert issubclass(CorvusTransportError, Exception)