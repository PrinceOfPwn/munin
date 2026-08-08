# tags: [tests, core, corvus, blackboard, threads, replies, reactions, red-contract, tdd]
"""RED contract for Corvus blackboard threaded messages and state.

``munin.corvus.blackboard`` only partially exists: ``CorvusBlackboard``,
``CorvusError`` and ``CorvusNotFoundError`` are published, but the threaded
message surface (``get_post``, ``reply``, ``get_thread``, ``react``,
``resolve``) is not implemented yet. The tests drive the missing surface
through the published ``CorvusBlackboard`` instance only.

Research evidence for the thread/external-state and command design:

* DeepWiki on ``langchain-ai/deepagents``: durable operational records live
  OUTSIDE the LangGraph checkpoint. Deep Agents separates the ephemeral
  in-graph ``StateBackend`` from the persistent ``StoreBackend`` (backed by an
  external ``BaseStore``, e.g. Redis), and a stable thread id is the
  rehydrate identity while the checkpoint never grows with the full thread
  transcript. Corvus threads are therefore external ordered-set stores, never
  graph state.
* Context7 on ``/websites/upstash_redis`` (Upstash Redis REST API): ``GET``
  returns an exact key's value; ``ZADD`` / ``ZRANGE`` / ``ZREM`` maintain
  ordered-set membership; ``XADD`` appends one stream entry; ``/multi-exec``
  (``pipeline(..., atomic=True)``) commits a whole command batch as one atomic
  transaction.

The tests drive every Redis interaction through a ``RecordingTransport`` (a
tiny ``RedisTransport`` fake defined in this file), so no live Redis and no
production change is required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from munin.corvus.blackboard import (
    CorvusBlackboard,
    CorvusError,
    CorvusNotFoundError,
)
from munin.corvus.contracts import (
    Post,
    PostState,
    PostType,
    Reaction,
    ReactionType,
    create_post,
)
from munin.corvus.transport import CorvusTransportError, RedisTransport

PREFIX = "munin:bb"
FIXED_UTC = datetime(2026, 8, 8, 6, 41, 22, 184000, tzinfo=timezone.utc)
BUENOS_AIRES = ZoneInfo("America/Argentina/Buenos_Aires")
TZ_NAME = "America/Argentina/Buenos_Aires"
ROOT_ID = "post-root"
CHILD_ID = "post-child"
GRANDCHILD_ID = "post-grandchild"


def clock(_tz: Any) -> datetime:
    """Callable clock contract: clock(tz) -> datetime."""
    return FIXED_UTC.astimezone(BUENOS_AIRES)


def post_key(post_id: str) -> str:
    return f"{PREFIX}:post:{post_id}"


def wire_json(post: Post) -> str:
    return json.dumps(post.to_wire(), sort_keys=True, separators=(",", ":"))


def make_post(
    post_id: str,
    *,
    actor_id: str = "srv-1",
    post_type: Any = "OBSERVATION",
    content: str = "signal",
    reply_to: str | None = None,
    thread_root_id: str | None = None,
    **overrides: Any,
) -> Post:
    """Server-fixed post reused as scripted storage: deterministic id."""
    post = create_post(
        actor_id=actor_id,
        post_type=post_type,
        scope="global",
        content=content,
        reply_to=reply_to,
        thread_root_id=thread_root_id,
        clock=clock,
        timezone=TZ_NAME,
    )
    update: dict[str, Any] = {"id": post_id}
    update.update(overrides)
    return post.model_copy(update=update)


class RecordingTransport(RedisTransport):
    """In-memory ``RedisTransport`` fake.

    Scripts exact-key ``GET`` and ``ZRANGE`` results, records every command
    call, records every pipeline call together with its ``atomic`` flag, and
    serves pipeline results FIFO from a queued ``pipeline_results`` list. A
    scripted ``pipeline_error`` is raised before any queued result is consumed.
    """

    def __init__(
        self,
        *,
        get_results: dict[str, Any] | None = None,
        zrange_results: dict[str, Any] | None = None,
        pipeline_results: list[list[Any]] | None = None,
        pipeline_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.get_results: dict[str, Any] = dict(get_results or {})
        self.zrange_results: dict[str, Any] = dict(zrange_results or {})
        self.pipeline_results: list[list[Any]] = list(pipeline_results or [])
        self.pipeline_error: Exception | None = pipeline_error
        self.command_calls: list[tuple] = []
        self.pipeline_calls: list[tuple] = []

    def command(self, *parts: Any) -> Any:
        self.command_calls.append(parts)
        key = parts[1] if len(parts) > 1 else None
        if parts and parts[0] == "GET" and key in self.get_results:
            return self.get_results[key]
        if parts and parts[0] == "ZRANGE" and key in self.zrange_results:
            return self.zrange_results[key]
        return None

    def pipeline(self, commands: list[list[Any]], atomic: bool = False) -> list[Any]:
        self.pipeline_calls.append((list(commands), atomic))
        if self.pipeline_error is not None:
            raise self.pipeline_error
        if self.pipeline_results:
            return self.pipeline_results.pop(0)
        return [None] * len(commands)

    def close(self) -> None:
        return None


def make_board(transport: RecordingTransport) -> CorvusBlackboard:
    return CorvusBlackboard(
        transport=transport,
        clock=clock,
        timezone=TZ_NAME,
        key_prefix=PREFIX,
    )


def test_get_post_reads_exact_key_and_rejects_missing_or_corrupt() -> None:
    root = make_post(ROOT_ID)
    transport = RecordingTransport(get_results={post_key(ROOT_ID): wire_json(root)})
    board = make_board(transport)

    assert board.get_post(ROOT_ID) == root
    assert transport.command_calls == [("GET", post_key(ROOT_ID))]

    missing = RecordingTransport(get_results={post_key(ROOT_ID): None})
    with pytest.raises(CorvusNotFoundError):
        make_board(missing).get_post(ROOT_ID)

    corrupt = RecordingTransport(get_results={post_key(ROOT_ID): "{not-json"})
    with pytest.raises(CorvusError):
        make_board(corrupt).get_post(ROOT_ID)


def test_reply_links_parent_and_indexes_under_root_thread_atomically() -> None:
    parent = make_post(ROOT_ID, content="root signal", post_type="OBSERVATION")
    transport = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(parent)}
    )
    board = make_board(transport)

    reply = board.reply(
        actor_id="actor-2",
        reply_to=ROOT_ID,
        content="observation",
        topics=("intel",),
        confidence=0.9,
        evidence_refs=("evt-1", "evt-2"),
        investigation_refs=("inv-1",),
    )

    assert isinstance(reply, Post)
    assert reply.type == PostType.REPLY
    assert reply.reply_to == ROOT_ID
    assert reply.thread_root_id == ROOT_ID
    assert reply.published_at == FIXED_UTC
    assert reply.published_at_local == "2026-08-08T03:41:22.184-03:00"
    assert reply.confidence == 0.9
    assert tuple(reply.evidence_refs) == ("evt-1", "evt-2")
    assert tuple(reply.investigation_refs) == ("inv-1",)

    assert ("GET", post_key(ROOT_ID)) in transport.command_calls
    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    set_cmd = next(
        cmd for cmd in commands if cmd[0] == "SET" and cmd[1] == post_key(reply.id)
    )
    stored = json.loads(set_cmd[2])
    assert stored["reply_to"] == ROOT_ID
    assert stored["thread_root_id"] == ROOT_ID

    score = int(reply.published_at.timestamp() * 1000)
    thread_zadds = [
        cmd
        for cmd in commands
        if cmd[0] == "ZADD" and cmd[1] == f"{PREFIX}:thread:{ROOT_ID}"
    ]
    assert thread_zadds == [
        ["ZADD", f"{PREFIX}:thread:{ROOT_ID}", score, reply.id]
    ]

    assert not any(
        cmd[0] == "ZADD" and cmd[1] == f"{PREFIX}:thread:{reply.id}"
        for cmd in commands
    )

    xadd_keys = [cmd[1] for cmd in commands if cmd[0] == "XADD"]
    assert xadd_keys == [f"{PREFIX}:stream"]

    child = make_post(
        CHILD_ID,
        post_type="REPLY",
        content="child",
        reply_to=ROOT_ID,
        thread_root_id=ROOT_ID,
    )
    deep_transport = RecordingTransport(
        get_results={post_key(CHILD_ID): wire_json(child)}
    )
    deep = make_board(deep_transport).reply(
        actor_id="actor-3", reply_to=CHILD_ID, content="deep"
    )
    assert deep.reply_to == CHILD_ID
    assert deep.thread_root_id == ROOT_ID


def test_get_thread_resolves_root_reads_index_and_batches_post_gets() -> None:
    thread_root = make_post(ROOT_ID, content="root")
    child = make_post(
        CHILD_ID,
        post_type="REPLY",
        content="a",
        reply_to=ROOT_ID,
        thread_root_id=ROOT_ID,
    )
    grandchild = make_post(
        GRANDCHILD_ID,
        post_type="REPLY",
        content="b",
        reply_to=CHILD_ID,
        thread_root_id=ROOT_ID,
    )

    transport = RecordingTransport(
        get_results={
            post_key(GRANDCHILD_ID): wire_json(grandchild),
            post_key(ROOT_ID): wire_json(thread_root),
            post_key(CHILD_ID): wire_json(child),
        },
        zrange_results={
            f"{PREFIX}:thread:{ROOT_ID}": [ROOT_ID, CHILD_ID, GRANDCHILD_ID]
        },
        pipeline_results=[
            [
                wire_json(thread_root),
                wire_json(child),
                wire_json(grandchild),
            ]
        ],
    )
    board = make_board(transport)

    posts = board.get_thread(GRANDCHILD_ID, limit=100)

    assert isinstance(posts, tuple)
    assert [post.id for post in posts] == [ROOT_ID, CHILD_ID, GRANDCHILD_ID]
    assert posts[0].id == ROOT_ID
    assert posts[0].type == PostType.OBSERVATION
    assert [post.type for post in posts[1:]] == [PostType.REPLY, PostType.REPLY]

    root_lookups = [call for call in transport.command_calls if call[0] == "GET"]
    assert root_lookups == [("GET", post_key(GRANDCHILD_ID))]

    zrange_calls = [call for call in transport.command_calls if call[0] == "ZRANGE"]
    assert zrange_calls == [("ZRANGE", f"{PREFIX}:thread:{ROOT_ID}", 0, 99)]

    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is False
    assert commands == [
        ["GET", post_key(ROOT_ID)],
        ["GET", post_key(CHILD_ID)],
        ["GET", post_key(GRANDCHILD_ID)],
    ]

    limit_transport = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(thread_root)}
    )
    limited = make_board(limit_transport)
    for bad_limit in (0, 501):
        with pytest.raises(CorvusError):
            limited.get_thread(ROOT_ID, limit=bad_limit)
    assert limit_transport.command_calls == []
    assert limit_transport.pipeline_calls == []


def test_react_verifies_post_and_stores_reaction_atomically() -> None:
    target = make_post(ROOT_ID, content="signal")
    transport = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(target)}
    )
    board = make_board(transport)

    reaction = board.react(
        post_id=ROOT_ID, reaction_type="CONFIRMED", actor_id="actor-9"
    )

    assert isinstance(reaction, Reaction)
    assert reaction.post_id == ROOT_ID
    assert reaction.type == ReactionType.CONFIRMED
    assert reaction.actor_id == "actor-9"
    assert reaction.timestamp == FIXED_UTC
    assert reaction.id.startswith("reaction_")

    assert ("GET", post_key(ROOT_ID)) in transport.command_calls
    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    set_cmd = next(
        cmd
        for cmd in commands
        if cmd[0] == "SET" and cmd[1] == f"{PREFIX}:reaction:{reaction.id}"
    )
    stored = json.loads(set_cmd[2])
    assert stored == reaction.to_wire()
    assert stored["timestamp"] == "2026-08-08T06:41:22.184Z"

    score = int(reaction.timestamp.timestamp() * 1000)
    zadd_cmds = [
        cmd
        for cmd in commands
        if cmd[0] == "ZADD" and cmd[1] == f"{PREFIX}:reactions:{ROOT_ID}"
    ]
    assert zadd_cmds == [["ZADD", f"{PREFIX}:reactions:{ROOT_ID}", score, reaction.id]]

    xadd_cmds = [cmd for cmd in commands if cmd[0] == "XADD"]
    assert len(xadd_cmds) == 1
    assert xadd_cmds[0][1] == f"{PREFIX}:stream"

    reject = RecordingTransport(get_results={post_key(ROOT_ID): wire_json(target)})
    with pytest.raises(CorvusError):
        make_board(reject).react(
            post_id=ROOT_ID, reaction_type="LIKE", actor_id="actor-9"
        )
    assert reject.pipeline_calls == []

    failing = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(target)},
        pipeline_error=CorvusTransportError("atomic write failed"),
    )
    with pytest.raises(CorvusError):
        make_board(failing).react(
            post_id=ROOT_ID, reaction_type="CONFIRMED", actor_id="actor-9"
        )


def test_resolve_preserves_identity_and_updates_state_atomically() -> None:
    open_post = make_post(ROOT_ID, content="signal")
    assert open_post.status == PostState.OPEN
    assert open_post.revision == 1

    transport = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(open_post)}
    )
    board = make_board(transport)

    resolved = board.resolve(post_id=ROOT_ID, actor_id="ops-7")

    assert resolved.id == ROOT_ID
    assert resolved.published_at == FIXED_UTC
    assert resolved.published_at_local == open_post.published_at_local
    assert resolved.status == PostState.RESOLVED
    assert resolved.revision == open_post.revision + 1

    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    set_cmd = next(
        cmd for cmd in commands if cmd[0] == "SET" and cmd[1] == post_key(ROOT_ID)
    )
    stored = json.loads(set_cmd[2])
    assert stored["id"] == ROOT_ID
    assert stored["published_at"] == "2026-08-08T06:41:22.184Z"
    assert stored["published_at_local"] == "2026-08-08T03:41:22.184-03:00"
    assert stored["status"] == "resolved"
    assert stored["revision"] == open_post.revision + 1

    zrem_cmds = [cmd for cmd in commands if cmd[0] == "ZREM"]
    assert zrem_cmds == [["ZREM", f"{PREFIX}:index:state:open", ROOT_ID]]

    score = int(FIXED_UTC.timestamp() * 1000)
    zadd_cmds = [
        cmd
        for cmd in commands
        if cmd[0] == "ZADD" and cmd[1] == f"{PREFIX}:index:state:resolved"
    ]
    assert zadd_cmds == [["ZADD", f"{PREFIX}:index:state:resolved", score, ROOT_ID]]

    xadd_cmds = [cmd for cmd in commands if cmd[0] == "XADD"]
    assert len(xadd_cmds) == 1
    assert xadd_cmds[0][1] == f"{PREFIX}:stream"
    flat_fields = [part for part in xadd_cmds[0] if isinstance(part, str)]
    assert "ops-7" in flat_fields

    already = make_post(
        ROOT_ID,
        content="signal",
        status=PostState.RESOLVED,
        revision=4,
    )
    idle_transport = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(already)}
    )
    result = make_board(idle_transport).resolve(post_id=ROOT_ID, actor_id="ops-7")
    assert result.status == PostState.RESOLVED
    assert result.revision == 4
    assert idle_transport.pipeline_calls == []


def test_unknown_parent_or_post_raises_not_found_with_no_pipeline() -> None:
    transport = RecordingTransport()
    board = make_board(transport)

    with pytest.raises(CorvusNotFoundError):
        board.reply(actor_id="actor-a", reply_to="post-missing", content="x")
    with pytest.raises(CorvusNotFoundError):
        board.react(
            post_id="post-missing", reaction_type="CONFIRMED", actor_id="actor-a"
        )
    with pytest.raises(CorvusNotFoundError):
        board.resolve(post_id="post-missing", actor_id="actor-a")

    assert transport.pipeline_calls == []