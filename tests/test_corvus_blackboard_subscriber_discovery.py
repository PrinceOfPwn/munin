# tags: [tests, core, corvus, blackboard, subscribers, discovery, red-contract, tdd]
"""RED contract for Corvus blackboard subscriber discovery.

``munin.corvus.blackboard`` already publishes ``CorvusBlackboard`` with the
actor registry (``register_actor``/``get_actor``) and the subscription surface
(``subscribe`` writes reverse ``subscribers:topic:<t>`` / ``subscribers:scope:<s>``
sets), plus the durable ``actors:created`` created-order index from
``register_actor``. The new read surface
``CorvusBlackboard.discover_subscribers(*, topics, scopes, limit)`` is not
implemented yet — importing the module is fine but calling the method is a RED
failure (``AttributeError``) until the subscriber-discovery Green pass lands.

Research evidence (the successful mandatory research calls):

* DeepWiki on ``langchain-ai/deepagents``: durable agent identity and
  capability metadata is a record owned outside the ephemeral LangGraph
  checkpoint. ``StoreBackend`` (an adapter over LangGraph's ``BaseStore``) holds
  memory/skills/subagent identity across threads, while the in-graph
  ``StateBackend`` is conversation-scoped and rehydrated from a stable thread
  id. Sub-agents and skills are discovered asynchronously through middleware
  (``AsyncSubAgentMiddleware`` for ``AsyncSubAgent`` entries described by
  ``name``/``description``/``graph_id``; ``SkillsMiddleware.abefore_agent``
  loads ``SkillMetadata`` from backend sources at runtime). Corvus subscriber
  discovery therefore reads a durable identity record (``actor:<id>``) and the
  durable reverse subscription sets over the injected transport, never graph
  state.
* Context7 on ``/websites/upstash_redis`` (Upstash Redis REST API): ``ZREVRANGE
  key start stop`` returns sorted-set members highest-score-first (newest-first
  for epoch-millisecond scores — the ``actors:created`` ordering); ``SMEMBERS
  key`` returns all members of a set (the reverse subscriber sets written by
  ``subscribe``); ``POST /pipeline`` executes a batched, ordered but explicitly
  **non-atomic** command sequence where each ``POST /pipeline`` response is a
  JSON array with one entry per submitted command in submission order — so the
  candidate fan-out (``ZREVRANGE`` + per-topic/per-scope ``SMEMBERS``) and the
  follow-on ``GET`` resolution are both safe as ordered non-atomic pipelines,
  and the OR union is computed in memory preserving ``actors:created``
  newest-first order. No Redis-side ``ZUNIONSTORE``, temp keys, or writes are
  required for the read.

Reused verbatim from ``tests.test_corvus_blackboard_threads``:
``PREFIX``, ``FIXED_UTC``, ``RecordingTransport``, ``make_board``. Reused from
``tests.test_corvus_blackboard_actors``: ``actor_key``, ``actor_json``,
``make_actor``. The candidate fan-out and the resolution are both non-atomic
pipelines served FIFO by the inherited ``RecordingTransport``. No live Redis
and no production change is required for this RED contract.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from munin.corvus.blackboard import CorvusError, CorvusNotFoundError
from munin.corvus.contracts import ActorIdentity
from munin.corvus.transport import CorvusTransportError
from tests.test_corvus_blackboard_threads import (
    FIXED_UTC,
    PREFIX,
    RecordingTransport,
    make_board,
)
from tests.test_corvus_blackboard_actors import (
    actor_key,
    actor_json,
    make_actor,
)

_MUTATION_COMMANDS = frozenset(
    {"SET", "SADD", "ZADD", "ZREM", "XADD", "DEL", "EVAL"}
)


def _assert_read_only(transport: RecordingTransport) -> None:
    """Every step is read-only: no writes, no atomic pipelines, no direct
    candidate/resolution commands, no single-element (bare) commands, and no
    Redis-side ``ZUNIONSTORE`` aggregation."""
    assert not (_MUTATION_COMMANDS & {call[0] for call in transport.command_calls})
    assert all(atomic is False for _, atomic in transport.pipeline_calls)
    for commands, _a in transport.pipeline_calls:
        assert not (_MUTATION_COMMANDS & {cmd[0] for cmd in commands})
        assert "ZUNIONSTORE" not in {cmd[0] for cmd in commands}
        assert not any(len(cmd) == 1 for cmd in commands)


def _topic_set(topic: str) -> str:
    return f"{PREFIX}:subscribers:topic:{topic}"


def _scope_set(scope: str) -> str:
    return f"{PREFIX}:subscribers:scope:{scope}"


def test_discover_subscribers_union_or_and_exact_non_atomic_pipelines() -> None:
    """Happy path: OR union over topics+scopes, newest-first filter on
    ``actors:created``, dedupe, cap to ``limit``, exactly two ordered
    non-atomic pipelines with no direct command calls."""
    newest = make_actor(actor_id="agent:gamma", created_at=FIXED_UTC)
    middle = make_actor(actor_id="agent:beta", created_at=FIXED_UTC - timedelta(hours=1))

    transport = RecordingTransport(
        pipeline_results=[
            [
                # ZREVRANGE actors:created, newest-first base list.
                ["agent:gamma", "agent:beta", "agent:alpha"],
                # SMEMBERS subscribers:topic:browser
                ["agent:alpha", "agent:beta"],
                # SMEMBERS subscribers:topic:cve
                ["agent:beta", "agent:ghost", "agent:alpha"],
                # SMEMBERS subscribers:scope:global
                ["agent:beta", "agent:gamma"],
                # SMEMBERS subscribers:scope:capability:web
                ["agent:alpha"],
            ],
            [actor_json(newest), actor_json(middle)],
        ],
    )
    board = make_board(transport)

    actors = board.discover_subscribers(
        topics=(" Browser ", "CVE", "browser"),
        scopes=("global", "capability:web", "global"),
        limit=2,
    )

    assert isinstance(actors, tuple)
    assert [actor.id for actor in actors] == ["agent:gamma", "agent:beta"]
    assert all(isinstance(actor, ActorIdentity) for actor in actors)

    # No direct command() calls — both candidate fan-out and resolution are
    # ordered non-atomic pipelines.
    assert transport.command_calls == []

    assert len(transport.pipeline_calls) == 2
    candidate_commands, candidate_atomic = transport.pipeline_calls[0]
    assert candidate_atomic is False
    assert candidate_commands == [
        ["ZREVRANGE", f"{PREFIX}:actors:created", 0, 499],
        ["SMEMBERS", _topic_set("browser")],
        ["SMEMBERS", _topic_set("cve")],
        ["SMEMBERS", _scope_set("global")],
        ["SMEMBERS", _scope_set("capability:web")],
    ]
    get_commands, get_atomic = transport.pipeline_calls[1]
    assert get_atomic is False
    assert get_commands == [
        ["GET", actor_key("agent:gamma")],
        ["GET", actor_key("agent:beta")],
    ]

    _assert_read_only(transport)


def test_discover_subscribers_empty_filters_and_empty_candidates() -> None:
    """Both normalized filters empty → ``()`` with zero transport. Empty base
    or empty OR selection → ``()`` with candidate pipeline only and no GET
    pipeline."""
    empty_filters = RecordingTransport()
    assert make_board(empty_filters).discover_subscribers() == ()
    assert empty_filters.command_calls == []
    assert empty_filters.pipeline_calls == []

    default_names = RecordingTransport()
    assert (
        make_board(default_names).discover_subscribers(topics=(), scopes=())
        == ()
    )
    assert default_names.command_calls == []
    assert default_names.pipeline_calls == []

    explicit_none = RecordingTransport()
    assert (
        make_board(explicit_none).discover_subscribers(topics=None, scopes=None)
        == ()
    )
    assert explicit_none.command_calls == []
    assert explicit_none.pipeline_calls == []

    # Empty base (no actors registered) → () no GET pipeline.
    empty_base = RecordingTransport(
        pipeline_results=[
            [
                [],  # ZREVRANGE actors:created
                ["agent:beta"],  # SMEMBERS subscribers:topic:web
            ],
        ],
    )
    assert make_board(empty_base).discover_subscribers(topics=("web",)) == ()
    assert empty_base.command_calls == []
    assert len(empty_base.pipeline_calls) == 1
    cand, atomic = empty_base.pipeline_calls[0]
    assert atomic is False
    assert cand == [
        ["ZREVRANGE", f"{PREFIX}:actors:created", 0, 499],
        ["SMEMBERS", _topic_set("web")],
    ]
    _assert_read_only(empty_base)

    # Non-empty base but no subscriber matches the OR union → () no GET
    # pipeline.
    mismatch = RecordingTransport(
        pipeline_results=[
            [
                ["agent:gamma", "agent:alpha"],
                ["agent:beta"],
                ["agent:beta"],
            ],
        ],
    )
    assert (
        make_board(mismatch).discover_subscribers(
            topics=("web",), scopes=("capability:web",)
        )
        == ()
    )
    assert mismatch.command_calls == []
    assert len(mismatch.pipeline_calls) == 1
    cand2, atomic2 = mismatch.pipeline_calls[0]
    assert atomic2 is False
    assert cand2 == [
        ["ZREVRANGE", f"{PREFIX}:actors:created", 0, 499],
        ["SMEMBERS", _topic_set("web")],
        ["SMEMBERS", _scope_set("capability:web")],
    ]
    _assert_read_only(mismatch)


class _OneShotPipelineErrorTransport(RecordingTransport):
    """Serve the first (candidate) pipeline from queued results, then raise the
    configured transport error on the second (GET resolution) pipeline.

    The parent ``RecordingTransport.pipeline`` raises ``pipeline_error`` before
    consuming any queued result, so to fail only on the second pipeline call we
    do not forward ``pipeline_error`` to the parent — the subclass raises it
    itself once the first (candidate) response has been served.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._get_error = "get transport exploded"
        self._served = False

    def pipeline(self, commands: list[list[Any]], atomic: bool = False) -> list[Any]:
        self.pipeline_calls.append((list(commands), atomic))
        if not self._served:
            self._served = True
            if self.pipeline_results:
                return self.pipeline_results.pop(0)
            return [None] * len(commands)
        raise CorvusTransportError(self._get_error)


def test_discover_subscribers_validation_and_failure_read_only_guarantees() -> None:
    """``limit`` validated first (non-bool int 1..500) and topic/scope
    normalization before any transport → zero transport on bad input. Missing
    actor → ``CorvusNotFoundError``; corrupt actor → sanitized ``CorvusError``
    with no raw payload echo; candidate/GET response cardinality mismatch →
    ``CorvusError``; transport errors → sanitized ``CorvusError``. Every path
    stays read-only."""
    validation = RecordingTransport()
    board = make_board(validation)
    for bad_limit in (True, False, 0, 501, "100", 1.0):
        with pytest.raises(CorvusError):
            board.discover_subscribers(topics=("web",), limit=bad_limit)
    for bad_topics in ("bare-string", ("",), ("   ",), (42,)):
        with pytest.raises(CorvusError):
            board.discover_subscribers(topics=bad_topics, scopes=("global",))
    for bad_scope in ("", "bare", "nope:prefix", "global:x"):
        with pytest.raises(CorvusError):
            board.discover_subscribers(topics=("web",), scopes=(bad_scope,))
    with pytest.raises(CorvusError):
        board.discover_subscribers(scopes="global")
    assert validation.command_calls == []
    assert validation.pipeline_calls == []

    # Candidate pipeline transport failure → sanitized CorvusError, no raw
    # transport text leaked, read-only.
    candidate_msg = "boom: subscriber read failed"
    candidate_fail = RecordingTransport(
        pipeline_error=CorvusTransportError(candidate_msg),
    )
    with pytest.raises(CorvusError) as excinfo:
        make_board(candidate_fail).discover_subscribers(
            topics=("web",), scopes=("global",)
        )
    assert excinfo.value.__cause__ is None
    assert len(candidate_fail.pipeline_calls) == 1
    _assert_read_only(candidate_fail)

    # Candidate response cardinality mismatch (short response) → CorvusError.
    candidate_mismatch = RecordingTransport(
        pipeline_results=[
            [
                ["agent:gamma", "agent:beta"],
                ["agent:beta"],
            ],
        ],
    )
    with pytest.raises(CorvusError):
        make_board(candidate_mismatch).discover_subscribers(
            topics=("web", "cve"), scopes=("global",)
        )
    assert len(candidate_mismatch.pipeline_calls) == 1
    _assert_read_only(candidate_mismatch)

    # GET pipeline transport failure → sanitized CorvusError, no raw payload
    # echo. The candidate pipeline returns its queued result first; the second
    # (GET) pipeline raises the configured transport error.
    get_fail = _OneShotPipelineErrorTransport(
        pipeline_results=[
            [
                ["agent:web-7", "agent:beta"],
                ["agent:web-7"],
                ["agent:beta"],
            ],
        ],
    )
    with pytest.raises(CorvusError) as excinfo_get:
        make_board(get_fail).discover_subscribers(
            topics=("web",), scopes=("global",)
        )
    assert excinfo_get.value.__cause__ is None
    assert len(get_fail.pipeline_calls) == 2
    _assert_read_only(get_fail)

    # Missing actor (None in GET response) → CorvusNotFoundError.
    missing = RecordingTransport(
        pipeline_results=[
            [
                ["agent:web-7"],
                ["agent:web-7"],
                ["agent:web-7"],
            ],
            [None],
        ],
    )
    with pytest.raises(CorvusNotFoundError):
        make_board(missing).discover_subscribers(
            topics=("web",), scopes=("global",)
        )
    assert len(missing.pipeline_calls) == 2
    _assert_read_only(missing)

    # Corrupt actor payload → sanitized CorvusError, no raw payload echo.
    corrupt_payload = "{not-json"
    corrupt = RecordingTransport(
        pipeline_results=[
            [
                ["agent:web-7"],
                ["agent:web-7"],
                ["agent:web-7"],
            ],
            [corrupt_payload],
        ],
    )
    with pytest.raises(CorvusError) as excinfo_corrupt:
        make_board(corrupt).discover_subscribers(
            topics=("web",), scopes=("global",)
        )
    assert corrupt_payload not in str(excinfo_corrupt.value)
    assert corrupt_payload not in repr(excinfo_corrupt.value)
    assert len(corrupt.pipeline_calls) == 2
    _assert_read_only(corrupt)

    # GET response cardinality mismatch (more results than selected ids) →
    # CorvusError.
    resolved_actor = make_actor(actor_id="agent:mismatch")
    get_mismatch = RecordingTransport(
        pipeline_results=[
            [
                ["agent:mismatch"],
                ["agent:mismatch"],
            ],
            [actor_json(resolved_actor), actor_json(resolved_actor)],
        ],
    )
    with pytest.raises(CorvusError):
        make_board(get_mismatch).discover_subscribers(topics=("only-match",))
    assert len(get_mismatch.pipeline_calls) == 2
    _assert_read_only(get_mismatch)
