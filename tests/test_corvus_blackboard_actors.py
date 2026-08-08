# tags: [tests, core, corvus, blackboard, actors, registry, presence, subscriptions, red-contract, tdd]
"""RED contract for Corvus blackboard actor registry micro-pass.

``munin.corvus.blackboard`` publishes ``CorvusBlackboard`` with the post,
thread, question and search surfaces, but the actor surface
(``register_actor``, ``get_actor``, ``heartbeat``, ``subscribe``) is not
implemented yet. The tests drive the missing surface through the published
``CorvusBlackboard`` instance only; importing the module and calling these
methods is currently a RED failure (``AttributeError``).

Research evidence for the actor-store design (the successful mandatory
research calls):

* DeepWiki on ``langchain-ai/deepagents``: subagent identity is a durable
  record carried on the supervisor's middleware, not ephemeral graph state —
  it must be rehydratable from an external store across threads. Context7 on
  ``/websites/upstash_redis``: ``SET key value EX <seconds>`` writes a value
  with a hard TTL; ``GET`` reads one exact key; ``ZADD key score member``
  indexes a member by an epoch-millisecond score; ``SADD key member ...``
  adds one or more set members — forward per-actor sets plus reverse
  per-topic/per-scope sets make subscription lookups exact; a ``/multi-exec``
  batch commits a whole command list as one atomic transaction while a plain
  ``/pipeline`` batch is ordered but non-atomic. Actor registration is a
  single atomic ``/multi-exec`` write, and presence uses ``SET ... EX``
  so expired actors can never stay visible.

No live Redis is required: every interaction is asserted through the reused
``RecordingTransport`` from ``tests.test_corvus_blackboard_threads`` with the
shared helpers (``FIXED_UTC``, ``PREFIX``, ``make_board``).
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from munin.corvus.blackboard import CorvusError, CorvusNotFoundError
from munin.corvus.contracts import ActorIdentity
from tests.test_corvus_blackboard_threads import (
    FIXED_UTC,
    PREFIX,
    RecordingTransport,
    make_board,
)


def actor_key(actor_id: str) -> str:
    return f"{PREFIX}:actor:{actor_id}"


def actor_json(actor: ActorIdentity) -> str:
    return json.dumps(actor.to_wire(), sort_keys=True, separators=(",", ":"))


def make_actor(*, actor_id: str = "agent:web-7", **overrides: object) -> ActorIdentity:
    """Deterministic identity reused across the actor passes in this file."""
    create_actor = ActorIdentity
    values = {
        "id": actor_id,
        "name": "Web Agent 7",
        "kind": "temporary_agent",
        "runtime": "deep_agent",
        "role": "Web Reconnaissance Operator",
        "capabilities": ("browser", "web"),
        "model_route": "generated_specialist",
        "created_at": FIXED_UTC,
        "terminated_at": None,
    }
    values.update(overrides)
    return create_actor(**values)  # type: ignore[call-arg]


def test_register_actor_writes_identity_and_capability_indices_in_one_atomic_pipeline() -> None:
    transport = RecordingTransport()
    board = make_board(transport)
    actor = make_actor()

    result = board.register_actor(identity=actor)

    assert result == actor
    assert transport.command_calls == []
    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    set_cmd = ["SET", actor_key(actor.id), actor_json(actor)]
    assert set_cmd in commands

    timestamp = int(actor.created_at.timestamp() * 1000)
    zadd_cmds = [cmd for cmd in commands if cmd[0] == "ZADD"]
    assert ["ZADD", f"{PREFIX}:actors:created", timestamp, actor.id] in zadd_cmds
    assert ["ZADD", f"{PREFIX}:index:capability:browser", timestamp, actor.id] in zadd_cmds
    assert ["ZADD", f"{PREFIX}:index:capability:web", timestamp, actor.id] in zadd_cmds
    assert len(zadd_cmds) == 3


def test_get_actor_reads_exact_key_and_missing_corrupt_raise_without_payload_echo() -> None:
    actor = make_actor()
    transport = RecordingTransport(get_results={actor_key(actor.id): actor_json(actor)})
    board = make_board(transport)

    assert board.get_actor(actor.id) == actor
    assert transport.command_calls == [("GET", actor_key(actor.id))]

    missing = RecordingTransport(get_results={actor_key(actor.id): None})
    with pytest.raises(CorvusNotFoundError):
        make_board(missing).get_actor(actor.id)

    corrupt_payload = "{not-json"
    corrupt = RecordingTransport(get_results={actor_key(actor.id): corrupt_payload})
    with pytest.raises(CorvusError) as excinfo:
        make_board(corrupt).get_actor(actor.id)
    # Generic CorvusError: stable message, never an echo of the raw payload.
    assert str(excinfo.value)
    assert corrupt_payload not in str(excinfo.value)
    assert corrupt_payload not in repr(excinfo.value)


def presence_key(actor_id: str) -> str:
    return f"{PREFIX}:presence:{actor_id}"


def presence_json(presence) -> str:
    return json.dumps(presence.to_wire(), sort_keys=True, separators=(",", ":"))


def test_heartbeat_records_presence_with_exact_server_timestamps_after_actor_verify() -> None:
    actor = make_actor()
    transport = RecordingTransport(get_results={actor_key(actor.id): actor_json(actor)})
    board = make_board(transport)

    presence = board.heartbeat(actor_id=actor.id, ttl_seconds=90)

    assert type(presence).__name__ == "ActorPresence"
    assert presence.actor_id == actor.id
    assert presence.heartbeat_at == FIXED_UTC
    assert presence.expires_at == FIXED_UTC + timedelta(seconds=90)

    wire = presence.to_wire()
    assert wire == {
        "actor_id": actor.id,
        "heartbeat_at": "2026-08-08T06:41:22.184Z",
        "expires_at": "2026-08-08T06:42:52.184Z",
    }

    expected_payload = presence_json(presence)
    assert transport.command_calls == [
        ("GET", actor_key(actor.id)),
        ("SET", presence_key(actor.id), expected_payload, "EX", 90),
    ]

    transport = RecordingTransport()
    board = make_board(transport)
    for bad_actor in ("", "   ", 42):
        with pytest.raises(CorvusError):
            board.heartbeat(actor_id=bad_actor, ttl_seconds=90)
    for bad_ttl in (True, 0, 86401):
        with pytest.raises(CorvusError):
            board.heartbeat(actor_id=actor.id, ttl_seconds=bad_ttl)
    assert transport.command_calls == []
    assert transport.pipeline_calls == []


def subscription_key(actor_id: str) -> str:
    return f"{PREFIX}:subscription:{actor_id}"


def forward_topic_set(actor_id: str) -> str:
    return f"{PREFIX}:subscription:actor:{actor_id}:topics"


def forward_scope_set(actor_id: str) -> str:
    return f"{PREFIX}:subscription:actor:{actor_id}:scopes"


def reverse_topic_set(topic: str) -> str:
    return f"{PREFIX}:subscribers:topic:{topic}"


def reverse_scope_set(scope: str) -> str:
    return f"{PREFIX}:subscribers:scope:{scope}"


def subscription_json(subs) -> str:
    return json.dumps(subs.to_wire(), sort_keys=True, separators=(",", ":"))


def test_subscribe_normalizes_dedupes_and_writes_forward_and_reverse_sets_atomically() -> None:
    actor = make_actor()
    transport = RecordingTransport(get_results={actor_key(actor.id): actor_json(actor)})
    board = make_board(transport)

    subs = board.subscribe(
        actor_id=actor.id,
        topics=(" Browser ", "CVE", "browser"),
        scopes=("global", "capability:web", "global"),
    )

    assert type(subs).__name__ == "ActorSubscriptions"
    assert subs.actor_id == actor.id
    assert tuple(subs.topics) == ("browser", "cve")
    assert tuple(subs.scopes) == ("global", "capability:web")
    assert subs.updated_at == FIXED_UTC

    assert subs.to_wire() == {
        "actor_id": actor.id,
        "topics": ["browser", "cve"],
        "scopes": ["global", "capability:web"],
        "updated_at": "2026-08-08T06:41:22.184Z",
    }

    assert transport.command_calls == [("GET", actor_key(actor.id))]
    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True
    assert commands == [
        ["SET", subscription_key(actor.id), subscription_json(subs)],
        ["SADD", forward_topic_set(actor.id), "browser", "cve"],
        ["SADD", forward_scope_set(actor.id), "global", "capability:web"],
        ["SADD", reverse_topic_set("browser"), actor.id],
        ["SADD", reverse_topic_set("cve"), actor.id],
        ["SADD", reverse_scope_set("global"), actor.id],
        ["SADD", reverse_scope_set("capability:web"), actor.id],
    ]

    transport = RecordingTransport()
    board = make_board(transport)
    for bad_topics in ("bare-string", ("",), ("   ",), (42,)):
        with pytest.raises(CorvusError):
            board.subscribe(
                actor_id=actor.id,
                topics=bad_topics,
                scopes=("global",),
            )
    for bad_scope in ("", "bare", "nope:prefix", "global:x"):
        with pytest.raises(CorvusError):
            board.subscribe(
                actor_id=actor.id,
                topics=("browser",),
                scopes=(bad_scope,),
            )
    for bad_actor in ("", "   ", 42):
        with pytest.raises(CorvusError):
            board.subscribe(
                actor_id=bad_actor,
                topics=("browser",),
                scopes=("global",),
            )
    assert transport.command_calls == []
    assert transport.pipeline_calls == []