# tags: [tests, corvus-blackboard, publish, minimal]
"""RED contract for corvus.blackboard publish orchestration.

The munin.corvus.blackboard module is not implemented yet; importing it
here is deliberate and currently fails, proving the contract is
incomplete. All transport behavior is asserted through RecordingTransport
so no live Redis dependency is needed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from munin.corvus.blackboard import CorvusBlackboard, CorvusError
from munin.corvus.contracts import Post, create_post
from munin.corvus.transport import CorvusTransportError, RedisTransport

FIXED_UTC = datetime(2026, 8, 8, 6, 41, 22, 184000, tzinfo=timezone.utc)
BUENOS_AIRES = ZoneInfo("America/Argentina/Buenos_Aires")
TZ_NAME = "America/Argentina/Buenos_Aires"
PREFIX = "munin:bb"


def clock(_tz) -> datetime:
    """Callable clock contract: clock(tz) -> datetime."""
    return FIXED_UTC.astimezone(BUENOS_AIRES)


def fingerprint_key(prefix: str, fingerprint: str) -> str:
    digest = hashlib.sha256(fingerprint.encode("ascii")).hexdigest()
    return f"{prefix}:fingerprint:{digest}"


def post_key(prefix: str, post_id: str) -> str:
    return f"{prefix}:post:{post_id}"


class RecordingTransport(RedisTransport):
    """In-memory RedisTransport recording every command and pipeline.
    Results are scripted through command_results; pipeline_error makes the
    atomic pipeline fail so rollback can be observed."""

    def __init__(
        self,
        command_results: dict | None = None,
        pipeline_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.command_calls: list[tuple] = []
        self.pipeline_calls: list[tuple] = []
        self.command_results: dict = command_results or {}
        self.pipeline_error: Exception | None = pipeline_error

    def command(self, *parts):
        self.command_calls.append(parts)
        name, *rest = parts
        if name == "GET" and rest and rest[0] in self.command_results:
            return self.command_results[rest[0]]
        return self.command_results.get(name)

    def pipeline(self, commands, atomic=False):
        self.pipeline_calls.append((list(commands), atomic))
        if self.pipeline_error is not None:
            raise self.pipeline_error
        return [self.command_results.get(cmd[0]) for cmd in commands]

    def close(self) -> None:
        return None


def make_board(transport: RecordingTransport) -> CorvusBlackboard:
    return CorvusBlackboard(
        transport=transport,
        clock=clock,
        timezone=TZ_NAME,
        key_prefix=PREFIX,
    )


def test_publish_returns_post_with_injected_timestamps_and_normalized_topics() -> None:
    transport = RecordingTransport()
    board = make_board(transport)

    post = board.publish(
        post_type="OBSERVATION", scope="global", actor_id="srv-1",
        content="c2 sighting", topics=("Intel", "intel ", " c2", "C2"),
        observed_at=FIXED_UTC, expires_at=FIXED_UTC + timedelta(hours=1),
    )

    assert isinstance(post, Post)
    assert post.published_at == FIXED_UTC
    assert post.published_at_local == "2026-08-08T03:41:22.184-03:00"
    assert post.timezone_name == TZ_NAME
    assert sorted(post.topics) == ["c2", "intel"]
    assert post.type.value == "observation"
    assert post.status.value == "open"


def test_one_atomic_pipeline_with_single_stream_and_index_indices() -> None:
    transport = RecordingTransport()
    board = make_board(transport)

    post = board.publish(
        actor_id="srv-1", post_type="OBSERVATION", scope="global",
        content="signal", topics=("C2", "OSINT"), observed_at=FIXED_UTC,
    )

    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    xadd_keys = [cmd[1] for cmd in commands if cmd[0] == "XADD"]
    assert xadd_keys == [f"{PREFIX}:stream"]

    set_keys = [cmd[1] for cmd in commands if cmd[0] == "SET"]
    assert f"{PREFIX}:post:{post.id}" in set_keys

    zadd_keys = {cmd[1] for cmd in commands if cmd[0] == "ZADD"}
    assert {
        f"{PREFIX}:index:all",
        f"{PREFIX}:index:scope:{post.scope}",
        f"{PREFIX}:index:state:{post.status.value}",
        f"{PREFIX}:index:actor:{post.actor_id}",
        f"{PREFIX}:thread:{post.id}",
        f"{PREFIX}:index:topic:c2",
        f"{PREFIX}:index:topic:osint",
    } <= zadd_keys


def test_stored_wire_json_round_trips_exactly() -> None:
    transport = RecordingTransport()
    board = make_board(transport)

    post = board.publish(
        actor_id="srv-1", post_type="OBSERVATION", scope="global",
        content="signal", topics=("intel",), observed_at=FIXED_UTC,
    )

    commands, _ = transport.pipeline_calls[0]
    payload = next(
        (
            json.loads(cmd[2])
            for cmd in commands
            if cmd[0] == "SET" and cmd[1] == f"{PREFIX}:post:{post.id}"
        ),
        None,
    )
    assert payload == post.to_wire()
    assert payload["published_at"] == "2026-08-08T06:41:22.184Z"
    assert payload["published_at_local"] == "2026-08-08T03:41:22.184-03:00"
    assert payload["timezone_name"] == TZ_NAME


def test_fingerprint_duplicate_reuses_existing_post_without_pipeline() -> None:
    fingerprint = "evt-1"
    existing_id = "post-existing"
    existing = create_post(
        actor_id="srv-1", post_type="OBSERVATION", scope="global",
        content="signal", topics=("c2",), observed_at=FIXED_UTC,
        clock=clock, timezone=TZ_NAME,
    ).model_copy(update={"id": existing_id})
    fp_key = fingerprint_key(PREFIX, fingerprint)
    transport = RecordingTransport(command_results={
        "SET": None,
        fp_key: existing_id,
        post_key(PREFIX, existing_id): json.dumps(existing.to_wire()),
    })
    board = make_board(transport)

    post = board.publish(
        actor_id="srv-1", post_type="OBSERVATION", scope="global",
        content="signal", topics=("c2",), fingerprint=fingerprint,
        observed_at=FIXED_UTC,
    )

    assert post.id == existing_id
    assert transport.pipeline_calls == []
    set_call = transport.command_calls[0]
    assert set_call[0] == "SET"
    assert set_call[1] == fp_key
    assert set_call[-1] == "NX"
    assert len(transport.command_calls) == 3
    joined = " ".join(str(part) for call in transport.command_calls for part in call)
    assert fingerprint not in joined


def test_reservation_released_via_eval_when_atomic_pipeline_fails() -> None:
    fingerprint = "evt-2"
    fp_key = fingerprint_key(PREFIX, fingerprint)
    transport = RecordingTransport(
        command_results={"SET": "OK"},
        pipeline_error=CorvusTransportError("atomic write failed"),
    )
    board = make_board(transport)

    with pytest.raises(CorvusError):
        board.publish(
            actor_id="srv-1", post_type="OBSERVATION", scope="global",
            content="signal", topics=("c2",), fingerprint=fingerprint,
            observed_at=FIXED_UTC,
        )

    names = [call[0] for call in transport.command_calls]
    assert "DEL" not in names
    assert "EVAL" in names
    reserved_call = transport.command_calls[0]
    assert reserved_call[0] == "SET"
    assert reserved_call[1] == fp_key
    candidate_id = reserved_call[2]
    eval_call = next(call for call in transport.command_calls if call[0] == "EVAL")
    eval_text = " ".join(str(part) for part in eval_call)
    assert fp_key in eval_text
    assert candidate_id in eval_text
    assert len(transport.pipeline_calls) == 1


def test_blank_topic_and_invalid_scope_raise_before_transport_mutation() -> None:
    transport = RecordingTransport()
    board = make_board(transport)

    with pytest.raises(CorvusError):
        board.publish(
            actor_id="srv-1", post_type="OBSERVATION", scope="global",
            content="signal", topics=("",), observed_at=FIXED_UTC,
        )
    assert transport.command_calls == []
    assert transport.pipeline_calls == []

    with pytest.raises(CorvusError):
        board.publish(
            actor_id="srv-1", post_type="OBSERVATION", scope="invalid",
            content="signal", topics=("intel",), observed_at=FIXED_UTC,
        )
    assert transport.command_calls == []
    assert transport.pipeline_calls == []