# tags: [tests, core, corvus, blackboard, discovery, capabilities, open-questions, red-contract, tdd]
"""RED contract for Corvus blackboard capability-based open-question discovery.

Micro-pass A: the capability fan-in read surface only. ``CorvusBlackboard``
already publishes ``get_actor`` (exact-key actor GET) and the question indexes
built by ``ask``/``publish``, but ``discover_open_questions`` is not
implemented yet — importing the module and calling the method is currently a
RED failure (``AttributeError``).

Research evidence (the successful mandatory research calls):

* DeepWiki on ``langchain-ai/deepagents``: subagent capability metadata is a
  durable record owned by ``SubAgentMiddleware`` — ``SubAgent``/``CompiledSubAgent``
  carry ``name``, ``description``, and ``tools`` as declarative identity, stored
  outside the ephemeral checkpoint and discovered at runtime by a name →
  graph lookup. Corvus discovery therefore reads a durable actor record via
  ``get_actor`` and fans out over its *normalized capability topic slugs*; the
  capability list is never trusted as graph state.
* Context7 on ``/websites/upstash_redis`` (Upstash Redis REST API):
  ``ZREVRANGE key start stop`` returns sorted-set members highest-score-first
  (newest-first for epoch-millisecond scores); ``POST /pipeline`` executes an
  ordered but **non-atomic** batch (other client requests may interleave) while
  ``POST /multi-exec`` is atomic. The candidate fan-out (``questions:open`` plus
  one ``index:topic:<capability>`` per capability) and the follow-on ``GET``
  resolution are therefore both safe as plain non-atomic pipelines, and the OR
  union is computed in memory preserving ``questions:open`` newest-first order
  — no Redis-side ``ZUNIONSTORE``, temp keys, or writes are required.

Reused verbatim from ``tests.test_corvus_blackboard_threads``:
``RecordingTransport``, ``make_board``, ``make_post``, ``post_key``,
``wire_json``. Reused from ``tests.test_corvus_blackboard_actors``:
``actor_key``, ``actor_json``, ``make_actor``. A minimal
``DiscoveryRecordingTransport`` subclass is added only for typing clarity —
candidate and resolution are both non-atomic pipelines served FIFO. No live
Redis and no production change is required.
"""

from __future__ import annotations

import pytest

from munin.corvus.blackboard import CorvusError, CorvusNotFoundError
from munin.corvus.contracts import Post
from tests.test_corvus_blackboard_threads import (
    PREFIX,
    RecordingTransport,
    make_board,
    make_post,
    post_key,
    wire_json,
)
from tests.test_corvus_blackboard_actors import (
    actor_json,
    actor_key,
    make_actor,
)

_MUTATION_COMMANDS = frozenset({"SET", "ZADD", "ZREM", "XADD", "DEL", "EVAL"})


class DiscoveryRecordingTransport(RecordingTransport):
    """``RecordingTransport`` for the discovery surface.

    ``get_actor`` uses one exact-key command ``GET`` (scripted via
    ``get_results``); the candidate fan-out and the matched-post resolution are
    both non-atomic pipelines, so the inherited ``pipeline_results`` FIFO serves
    them in order — the first dequeued result is the candidate list, a second is
    the ``GET`` results. No command-level scripting beyond ``GET`` is required.
    """


def _assert_read_only(transport: DiscoveryRecordingTransport) -> None:
    assert not (_MUTATION_COMMANDS & {call[0] for call in transport.command_calls})
    assert all(a is False for _, a in transport.pipeline_calls)
    for commands, _a in transport.pipeline_calls:
        assert not (_MUTATION_COMMANDS & {cmd[0] for cmd in commands})
        assert "ZUNIONSTORE" not in {cmd[0] for cmd in commands}
        assert not any(len(cmd) == 1 for cmd in commands)


def test_discover_open_questions_actor_capabilities_union_exact_pipelines() -> None:
    q_newest = make_post("post-q3", post_type="QUESTION", content="cve match")
    q_mid = make_post("post-q2", post_type="QUESTION", content="browser match")
    actor = make_actor(actor_id="agent:web-7", capabilities=("Browser", "CVE"))

    transport = DiscoveryRecordingTransport(
        get_results={actor_key(actor.id): actor_json(actor)},
        pipeline_results=[
            [
                ["post-q3", "post-q2", "post-q1"],
                ["post-q2", "post-q1"],
                ["post-q3"],
            ],
            [wire_json(q_newest), wire_json(q_mid)],
        ],
    )
    board = make_board(transport)

    posts = board.discover_open_questions(actor.id, limit=2)

    assert isinstance(posts, tuple)
    assert [post.id for post in posts] == ["post-q3", "post-q2"]
    assert all(isinstance(post, Post) for post in posts)

    assert transport.command_calls == [("GET", actor_key(actor.id))]

    assert len(transport.pipeline_calls) == 2
    candidate_commands, candidate_atomic = transport.pipeline_calls[0]
    assert candidate_atomic is False
    assert candidate_commands == [
        ["ZREVRANGE", f"{PREFIX}:questions:open", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:topic:browser", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:topic:cve", 0, 499],
    ]
    get_commands, get_atomic = transport.pipeline_calls[1]
    assert get_atomic is False
    assert get_commands == [
        ["GET", post_key("post-q3")],
        ["GET", post_key("post-q2")],
    ]

    _assert_read_only(transport)


def test_discover_open_questions_empty_capabilities_and_empty_questions() -> None:
    no_caps = make_actor(actor_id="agent:quiet", capabilities=())
    first = DiscoveryRecordingTransport(
        get_results={actor_key(no_caps.id): actor_json(no_caps)}
    )
    assert make_board(first).discover_open_questions(no_caps.id) == ()
    assert first.command_calls == [("GET", actor_key(no_caps.id))]
    assert first.pipeline_calls == []

    empty_question = make_actor(actor_id="agent:lone", capabilities=("web",))
    second = DiscoveryRecordingTransport(
        get_results={actor_key(empty_question.id): actor_json(empty_question)},
        pipeline_results=[[[], []]],
    )
    assert make_board(second).discover_open_questions(empty_question.id) == ()
    assert second.command_calls == [("GET", actor_key(empty_question.id))]
    assert len(second.pipeline_calls) == 1
    candidate_commands, candidate_atomic = second.pipeline_calls[0]
    assert candidate_atomic is False
    assert candidate_commands == [
        ["ZREVRANGE", f"{PREFIX}:questions:open", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:topic:web", 0, 499],
    ]

    _assert_read_only(second)


def test_discover_open_questions_validation_and_failure_read_only_guarantees() -> None:
    validation = DiscoveryRecordingTransport()
    board = make_board(validation)
    for bad_actor in ("", "   ", 42):
        with pytest.raises(CorvusError):
            board.discover_open_questions(bad_actor)
    for bad_limit in (True, 0, 501):
        with pytest.raises(CorvusError):
            board.discover_open_questions("agent:ok", limit=bad_limit)
    assert validation.command_calls == []
    assert validation.pipeline_calls == []

    missing_actor = make_actor(actor_id="agent:ghost")
    missing = DiscoveryRecordingTransport(
        get_results={actor_key(missing_actor.id): None}
    )
    with pytest.raises(CorvusNotFoundError):
        make_board(missing).discover_open_questions(missing_actor.id)
    assert missing.command_calls == [("GET", actor_key(missing_actor.id))]
    assert missing.pipeline_calls == []

    web_actor = make_actor(actor_id="agent:web-7", capabilities=("web",))
    missing_post = DiscoveryRecordingTransport(
        get_results={actor_key(web_actor.id): actor_json(web_actor)},
        pipeline_results=[[["post-ghost"], ["post-ghost"]], [None]],
    )
    with pytest.raises(CorvusNotFoundError):
        make_board(missing_post).discover_open_questions(web_actor.id)
    assert len(missing_post.pipeline_calls) == 2
    _assert_read_only(missing_post)

    corrupt = DiscoveryRecordingTransport(
        get_results={actor_key(web_actor.id): actor_json(web_actor)},
        pipeline_results=[[["post-bad"], ["post-bad"]], ["{not-json"]],
    )
    with pytest.raises(CorvusError) as excinfo:
        make_board(corrupt).discover_open_questions(web_actor.id)
    assert "{not-json" not in str(excinfo.value)
    assert "{not-json" not in repr(excinfo.value)
    assert len(corrupt.pipeline_calls) == 2
    _assert_read_only(corrupt)

    candidate_mismatch = DiscoveryRecordingTransport(
        get_results={actor_key(web_actor.id): actor_json(web_actor)},
        pipeline_results=[[["post-q1"]]],
    )
    with pytest.raises(CorvusError):
        make_board(candidate_mismatch).discover_open_questions(web_actor.id)
    assert len(candidate_mismatch.pipeline_calls) == 1
    _assert_read_only(candidate_mismatch)

    q1 = make_post("post-q1", post_type="QUESTION", content="web match")
    get_mismatch = DiscoveryRecordingTransport(
        get_results={actor_key(web_actor.id): actor_json(web_actor)},
        pipeline_results=[
            [["post-q1"], ["post-q1"]],
            [wire_json(q1), wire_json(q1)],
        ],
    )
    with pytest.raises(CorvusError):
        make_board(get_mismatch).discover_open_questions(web_actor.id)
    assert len(get_mismatch.pipeline_calls) == 2
    _assert_read_only(get_mismatch)