# tags: [tests, core, corvus, blackboard, questions, ask, feed, red-contract, tdd]
"""RED contract for Corvus blackboard question surface.

``munin.corvus.blackboard`` publishes ``CorvusBlackboard`` with ``publish``,
``get_post``, ``reply``, ``get_thread``, ``react`` and ``resolve``, but the
question surface (``ask``, ``get_open_questions``, ``feed``) is not implemented
yet. The tests drive the missing surface through the published
``CorvusBlackboard`` instance only; importing the module and calling these
methods is currently a RED failure.

Research evidence for the external-store and command design:

* DeepWiki on ``langchain-ai/deepagents``: durable operational records live
  OUTSIDE the ephemeral LangGraph checkpoint. Deep Agents separates the
  in-graph ephemeral ``StateBackend`` from the persistent
  ``StoreBackend``/``BaseStore``; a stable thread id is the rehydrate identity
  and the checkpoint never grows with the full thread transcript. Corvus open
  questions are therefore external Redis sorted-set indices, never graph state.
* Context7 on ``/websites/upstash_redis`` (Upstash Redis): ``ZADD`` /
  ``ZREM`` maintain sorted-set membership; ``ZREVRANGE`` returns members
  highest-score-first (newest-first for epoch-millisecond scores); ``GET``
  reads one exact key; a ``/multi-exec`` atomic batch commits all-or-nothing
  while a plain ``/pipeline`` batch is ordered but non-atomic — the question
  markers are written in one atomic publish pipeline and read via a single
  non-atomic ``GET`` batch.

Shared helpers (``RecordingTransport``, ``make_board``, ``make_post``, etc.)
are reused verbatim from ``tests.test_corvus_blackboard_threads``. Only a
minimal ``QuestionRecordingTransport`` subclass is added here, because the
threaded fake scripts ``ZRANGE`` but the question surface reads with
``ZREVRANGE``. No live Redis is required.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from munin.corvus.blackboard import CorvusError, CorvusNotFoundError
from munin.corvus.contracts import Post, PostState, PostType
from tests.test_corvus_blackboard_threads import (
    FIXED_UTC,
    PREFIX,
    RecordingTransport,
    ROOT_ID,
    make_board,
    make_post,
    post_key,
    wire_json,
)

_MUTATION_COMMANDS = frozenset({"SET", "ZADD", "ZREM", "XADD", "DEL", "EVAL"})


class QuestionRecordingTransport(RecordingTransport):
    """Minimal ``RecordingTransport`` addition: scriptable ``ZREVRANGE``.

    The threaded fake scripts ``GET`` and ``ZRANGE`` only; the question reads
    use ``ZREVRANGE`` (newest-first) so a keyed ``zrevrange_results`` map is
    added here. ``GET`` scripting, pipeline FIFO, cardinality and error
    injection are all inherited unchanged.
    """

    def __init__(
        self,
        *,
        zrevrange_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.zrevrange_results: dict[str, Any] = dict(zrevrange_results or {})

    def command(self, *parts: Any) -> Any:
        key = parts[1] if len(parts) > 1 else None
        if parts and parts[0] == "ZREVRANGE":
            self.command_calls.append(parts)
            if key in self.zrevrange_results:
                return self.zrevrange_results[key]
            return None
        return super().command(*parts)


def test_ask_creates_open_question_with_one_atomic_pipeline() -> None:
    transport = QuestionRecordingTransport()
    board = make_board(transport)

    question = board.ask(
        actor_id="actor-q1",
        content="Which C2 deployment serves this beacon?",
        topics=("c2",),
        scope="global",
        confidence=0.6,
        evidence_refs=("evt-9",),
        investigation_refs=("inv-7",),
    )

    assert isinstance(question, Post)
    assert question.type == PostType.QUESTION
    assert question.status == PostState.OPEN
    assert question.actor_id == "actor-q1"
    assert question.scope == "global"
    assert tuple(question.topics) == ("c2",)
    assert question.published_at == FIXED_UTC
    assert question.published_at_local == "2026-08-08T03:41:22.184-03:00"
    assert question.confidence == 0.6
    assert tuple(question.evidence_refs) == ("evt-9",)
    assert tuple(question.investigation_refs) == ("inv-7",)

    assert transport.command_calls == []
    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    score = int(FIXED_UTC.timestamp() * 1000)
    open_zadds = [
        cmd
        for cmd in commands
        if cmd[0] == "ZADD" and cmd[1] == f"{PREFIX}:questions:open"
    ]
    assert open_zadds == [["ZADD", f"{PREFIX}:questions:open", score, question.id]]

    set_cmd = next(
        cmd
        for cmd in commands
        if cmd[0] == "SET" and cmd[1] == post_key(question.id)
    )
    stored = json.loads(set_cmd[2])
    assert stored["id"] == question.id
    assert stored["type"] == "question"
    assert stored["status"] == "open"
    assert stored["published_at"] == "2026-08-08T06:41:22.184Z"

    xadd_keys = [cmd[1] for cmd in commands if cmd[0] == "XADD"]
    assert xadd_keys == [f"{PREFIX}:stream"]


def test_resolve_question_moves_open_to_resolved_atomically() -> None:
    open_question = make_post(
        ROOT_ID, post_type="QUESTION", content="Which implant?"
    )
    assert open_question.type == PostType.QUESTION
    assert open_question.status == PostState.OPEN
    assert open_question.revision == 1

    transport = RecordingTransport(
        get_results={post_key(ROOT_ID): wire_json(open_question)}
    )
    board = make_board(transport)

    resolved = board.resolve(post_id=ROOT_ID, actor_id="ops-7")

    assert resolved.id == ROOT_ID
    assert resolved.type == PostType.QUESTION
    assert resolved.status == PostState.RESOLVED
    assert resolved.published_at == FIXED_UTC
    assert resolved.published_at_local == open_question.published_at_local
    assert resolved.revision == open_question.revision + 1

    assert transport.command_calls == [("GET", post_key(ROOT_ID))]
    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is True

    score = int(FIXED_UTC.timestamp() * 1000)

    zrem_questions = [
        cmd
        for cmd in commands
        if cmd[0] == "ZREM" and cmd[1] == f"{PREFIX}:questions:open"
    ]
    assert zrem_questions == [["ZREM", f"{PREFIX}:questions:open", ROOT_ID]]

    zadd_questions = [
        cmd
        for cmd in commands
        if cmd[0] == "ZADD" and cmd[1] == f"{PREFIX}:questions:resolved"
    ]
    assert zadd_questions == [
        ["ZADD", f"{PREFIX}:questions:resolved", score, ROOT_ID]
    ]

    set_cmd = next(
        cmd
        for cmd in commands
        if cmd[0] == "SET" and cmd[1] == post_key(ROOT_ID)
    )
    stored = json.loads(set_cmd[2])
    assert stored["type"] == "question"
    assert stored["status"] == "resolved"
    assert stored["revision"] == open_question.revision + 1

    assert ["ZREM", f"{PREFIX}:index:state:open", ROOT_ID] in commands
    assert ["ZADD", f"{PREFIX}:index:state:resolved", score, ROOT_ID] in commands

    xadd_cmds = [cmd for cmd in commands if cmd[0] == "XADD"]
    assert len(xadd_cmds) == 1
    assert xadd_cmds[0][1] == f"{PREFIX}:stream"


def test_get_open_questions_revranges_and_batches_newest_first() -> None:
    older = make_post("post-q-old", post_type="QUESTION", content="old")
    newer = make_post("post-q-new", post_type="QUESTION", content="new")
    transport = QuestionRecordingTransport(
        get_results={
            post_key("post-q-old"): wire_json(older),
            post_key("post-q-new"): wire_json(newer),
        },
        zrevrange_results={
            f"{PREFIX}:questions:open": ["post-q-new", "post-q-old"],
        },
        pipeline_results=[
            [wire_json(newer), wire_json(older)],
        ],
    )
    board = make_board(transport)

    open_questions = board.get_open_questions(limit=50)

    assert isinstance(open_questions, tuple)
    assert [post.id for post in open_questions] == ["post-q-new", "post-q-old"]
    assert all(post.type is PostType.QUESTION for post in open_questions)
    assert all(post.status is PostState.OPEN for post in open_questions)

    rev_calls = [call for call in transport.command_calls if call[0] == "ZREVRANGE"]
    assert rev_calls == [("ZREVRANGE", f"{PREFIX}:questions:open", 0, 49)]
    assert not any(call[0] == "GET" for call in transport.command_calls)

    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is False
    assert commands == [
        ["GET", post_key("post-q-new")],
        ["GET", post_key("post-q-old")],
    ]

    assert not (_MUTATION_COMMANDS & {call[0] for call in transport.command_calls})
    assert all(a is False for _, a in transport.pipeline_calls)

    validation = QuestionRecordingTransport()
    for bad_limit in (True, 0, 501):
        with pytest.raises(CorvusError):
            make_board(validation).get_open_questions(limit=bad_limit)
    assert validation.command_calls == []
    assert validation.pipeline_calls == []


def test_feed_reads_scope_index_newest_first_without_mutations() -> None:
    first = make_post("post-a-one", content="first")
    second = make_post("post-b-two", content="second")
    transport = QuestionRecordingTransport(
        zrevrange_results={
            f"{PREFIX}:index:scope:global": ["post-b-two", "post-a-one"],
        },
        pipeline_results=[
            [wire_json(second), wire_json(first)],
        ],
    )
    board = make_board(transport)

    posts = board.feed(scope="global", limit=50)

    assert isinstance(posts, tuple)
    assert [post.id for post in posts] == ["post-b-two", "post-a-one"]
    assert posts[0].scope == "global"

    rev_calls = [call for call in transport.command_calls if call[0] == "ZREVRANGE"]
    assert rev_calls == [("ZREVRANGE", f"{PREFIX}:index:scope:global", 0, 49)]
    assert not any(call[0] == "GET" for call in transport.command_calls)

    assert len(transport.pipeline_calls) == 1
    commands, atomic = transport.pipeline_calls[0]
    assert atomic is False
    assert commands == [
        ["GET", post_key("post-b-two")],
        ["GET", post_key("post-a-one")],
    ]

    assert not (_MUTATION_COMMANDS & {call[0] for call in transport.command_calls})
    assert all(a is False for _, a in transport.pipeline_calls)

    missing = QuestionRecordingTransport(
        zrevrange_results={f"{PREFIX}:index:scope:global": ["post-ghost"]},
        pipeline_results=[[None]],
    )
    with pytest.raises(CorvusNotFoundError):
        make_board(missing).feed(scope="global", limit=50)

    corrupt = QuestionRecordingTransport(
        zrevrange_results={f"{PREFIX}:index:scope:global": ["post-bad"]},
        pipeline_results=[["{not-json"]],
    )
    with pytest.raises(CorvusError):
        make_board(corrupt).feed(scope="global", limit=50)

    cardinality = QuestionRecordingTransport(
        zrevrange_results={
            f"{PREFIX}:index:scope:global": ["post-x", "post-y"],
        },
        pipeline_results=[["{single"]],
    )
    with pytest.raises(CorvusError):
        make_board(cardinality).feed(scope="global", limit=50)

    for strict in (missing, corrupt, cardinality):
        assert not (_MUTATION_COMMANDS & {call[0] for call in strict.command_calls})
        assert all(a is False for _, a in strict.pipeline_calls)

    validation = QuestionRecordingTransport()
    for bad_scope in ("", "bare", "nope:prefix"):
        with pytest.raises(CorvusError):
            make_board(validation).feed(scope=bad_scope, limit=50)
    for bad_limit in (True, 0, 501):
        with pytest.raises(CorvusError):
            make_board(validation).feed(scope="global", limit=bad_limit)
    assert validation.command_calls == []
    assert validation.pipeline_calls == []