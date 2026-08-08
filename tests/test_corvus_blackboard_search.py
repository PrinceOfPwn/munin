# tags: [tests, core, corvus, blackboard, search, multi-filter, intersection, red-contract, tdd]
"""RED contract for Corvus blackboard multi-filter search.

``munin.corvus.blackboard`` publishes ``CorvusBlackboard`` with ``publish``,
``get_post``, ``reply``, ``get_thread``, ``react``, ``resolve``, ``ask``,
``get_open_questions`` and ``feed``, but the search surface (``search``) is not
implemented yet. The tests drive the missing surface through the published
``CorvusBlackboard`` instance only; importing the module and calling the method
is currently a RED failure (``AttributeError``).

Research evidence for the external-store and command design:

* DeepWiki on ``langchain-ai/deepagents``: durable operational records live
  OUTSIDE the ephemeral LangGraph checkpoint. Deep Agents separates the
  in-graph ephemeral ``StateBackend`` (files persisted in a per-thread
  ``files`` state channel, checkpointed after each agent step) from the
  persistent ``StoreBackend`` (an external LangGraph ``BaseStore`` organized by
  namespaces, available cross-thread). A stable thread id is the rehydrate
  identity, and the checkpoint never grows with the full thread transcript —
  the ``DeltaChannel`` keeps checkpoint blobs small while durable records are
  managed separately. Corvus search indices are therefore external Redis
  sorted-set stores, never graph state; the search intersects candidate sets
  outside the checkpoint and resolves only matched ids.
* Context7 on ``/websites/upstash_redis`` (Upstash Redis REST API): ``ZREVRANGE
  key start stop`` returns sorted-set members highest-score-first (newest-first
  for epoch-millisecond scores) over ``[start, stop]`` inclusive offsets;
  ``GET`` reads one exact key. A ``/pipeline`` batch is ordered but
  **non-atomic** — commands are processed in order yet other client requests may
  interleave, so a multi-``ZREVRANGE`` candidate fan-out and a follow-on
  ``GET``-only resolution batch are both safe as non-atomic pipelines. No
  Redis-side ``ZINTERSTORE`` or temp-key writes are required: the intersection
  is computed in memory, preserving ``index:all`` newest-first order.

The tests drive every Redis interaction through a ``SearchRecordingTransport``
(a ``RecordingTransport`` subclass defined in this file that scripts
non-atomic pipeline results FIFO), so no live Redis and no production change is
required.
"""

from __future__ import annotations

import pytest

from munin.corvus.blackboard import CorvusError, CorvusNotFoundError
from tests.test_corvus_blackboard_threads import (
    PREFIX,
    RecordingTransport,
    make_board,
    make_post,
    post_key,
    wire_json,
)

_MUTATION_COMMANDS = frozenset({"SET", "ZADD", "ZREM", "XADD", "DEL", "EVAL"})


class SearchRecordingTransport(RecordingTransport):
    """``RecordingTransport`` for the search surface.

    The candidate ``ZREVRANGE`` fan-out and the matched post ``GET`` resolution
    are both non-atomic pipelines, so the inherited ``pipeline_results`` FIFO
    is sufficient: the first dequeued result serves the candidate pipeline,
    a second dequeued result serves the ``GET``-only resolution pipeline. No
    command-level scripting is required.
    """


def _assert_read_only(transport: SearchRecordingTransport) -> None:
    assert transport.command_calls == []
    assert all(a is False for _, a in transport.pipeline_calls)
    for commands, _a in transport.pipeline_calls:
        assert not (_MUTATION_COMMANDS & {cmd[0] for cmd in commands})


def test_search_no_filters_candidate_pipeline_then_get_pipeline_newest_first() -> None:
    older = make_post("post-a")
    newer = make_post("post-b")
    transport = SearchRecordingTransport(
        pipeline_results=[
            [["post-b", "post-a"]],
            [wire_json(newer), wire_json(older)],
        ],
    )
    board = make_board(transport)

    posts = board.search(limit=50)

    assert isinstance(posts, tuple)
    assert [post.id for post in posts] == ["post-b", "post-a"]

    assert len(transport.pipeline_calls) == 2
    candidate_commands, candidate_atomic = transport.pipeline_calls[0]
    assert candidate_atomic is False
    assert candidate_commands == [
        ["ZREVRANGE", f"{PREFIX}:index:all", 0, 499],
    ]
    get_commands, get_atomic = transport.pipeline_calls[1]
    assert get_atomic is False
    assert get_commands == [
        ["GET", post_key("post-b")],
        ["GET", post_key("post-a")],
    ]

    _assert_read_only(transport)


def test_search_multi_filter_intersects_in_index_all_order_and_caps_to_limit() -> None:
    p_newest = make_post("post-n1")
    p_mid = make_post("post-n2")
    newest_first = ["post-n1", "post-n2", "post-n3", "post-n4"]
    topic_matches = ["post-n4", "post-n2", "post-n1"]
    scope_matches = ["post-n3", "post-n2", "post-n1"]
    actor_matches = ["post-n2", "post-n1"]
    state_matches = ["post-n4", "post-n2", "post-n1", "post-n3"]
    transport = SearchRecordingTransport(
        pipeline_results=[
            [
                newest_first,
                topic_matches,
                scope_matches,
                actor_matches,
                state_matches,
            ],
            [wire_json(p_newest), wire_json(p_mid)],
        ],
    )
    board = make_board(transport)

    posts = board.search(
        topics=("C2!",),
        scope="global",
        actor_id="actor-x",
        status="OPEN",
        limit=2,
    )

    assert [post.id for post in posts] == ["post-n1", "post-n2"]

    assert len(transport.pipeline_calls) == 2
    candidate_commands, candidate_atomic = transport.pipeline_calls[0]
    assert candidate_atomic is False
    assert candidate_commands == [
        ["ZREVRANGE", f"{PREFIX}:index:all", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:topic:c2", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:scope:global", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:actor:actor-x", 0, 499],
        ["ZREVRANGE", f"{PREFIX}:index:state:open", 0, 499],
    ]
    get_commands, get_atomic = transport.pipeline_calls[1]
    assert get_atomic is False
    assert get_commands == [
        ["GET", post_key("post-n1")],
        ["GET", post_key("post-n2")],
    ]

    _assert_read_only(transport)


def test_search_normalizes_topics_and_rejects_invalid_arguments_before_transport() -> None:
    normalization = SearchRecordingTransport(
        pipeline_results=[
            [["post-a"], ["post-a"]],
            [wire_json(make_post("post-a"))],
        ],
    )
    board = make_board(normalization)
    posts = board.search(topics=("  C_2!! ",), limit=1)
    assert [post.id for post in posts] == ["post-a"]

    candidate_commands, _a = normalization.pipeline_calls[0]
    rev_keys = [cmd[1] for cmd in candidate_commands if cmd[0] == "ZREVRANGE"]
    assert f"{PREFIX}:index:topic:c_2" in rev_keys

    bad_topics = SearchRecordingTransport()
    with pytest.raises(CorvusError):
        make_board(bad_topics).search(topics="bare-string", limit=50)
    blank_topics = SearchRecordingTransport()
    with pytest.raises(CorvusError):
        make_board(blank_topics).search(topics=("   ",), limit=50)
    nonstring_topics = SearchRecordingTransport()
    with pytest.raises(CorvusError):
        make_board(nonstring_topics).search(topics=(42,), limit=50)

    validation = SearchRecordingTransport()
    for bad_scope in ("", "bare", "nope:prefix"):
        with pytest.raises(CorvusError):
            make_board(validation).search(scope=bad_scope, limit=50)
    for bad_status in ("", " bogus ", "LIKE"):
        with pytest.raises(CorvusError):
            make_board(validation).search(status=bad_status, limit=50)
    for bad_actor in ("", "   "):
        with pytest.raises(CorvusError):
            make_board(validation).search(actor_id=bad_actor, limit=50)
    for bad_limit in (True, 0, 501):
        with pytest.raises(CorvusError):
            make_board(validation).search(limit=bad_limit)

    assert bad_topics.command_calls == []
    assert bad_topics.pipeline_calls == []
    assert blank_topics.command_calls == []
    assert blank_topics.pipeline_calls == []
    assert nonstring_topics.command_calls == []
    assert nonstring_topics.pipeline_calls == []
    assert validation.command_calls == []
    assert validation.pipeline_calls == []


def test_search_empty_missing_corrupt_and_read_only_guarantees() -> None:
    empty = SearchRecordingTransport(
        pipeline_results=[[[]]],
    )
    assert make_board(empty).search(limit=50) == ()
    assert len(empty.pipeline_calls) == 1
    _assert_read_only(empty)

    missing = SearchRecordingTransport(
        pipeline_results=[
            [["post-ghost"]],
            [None],
        ],
    )
    with pytest.raises(CorvusNotFoundError):
        make_board(missing).search(limit=50)
    assert len(missing.pipeline_calls) == 2
    _assert_read_only(missing)

    corrupt = SearchRecordingTransport(
        pipeline_results=[
            [["post-bad"]],
            ['{"not-json'],
        ],
    )
    with pytest.raises(CorvusError):
        make_board(corrupt).search(limit=50)
    assert len(corrupt.pipeline_calls) == 2
    _assert_read_only(corrupt)

    cardinality = SearchRecordingTransport(
        pipeline_results=[
            [["post-x", "post-y"]],
            ["{only-one"],
        ],
    )
    with pytest.raises(CorvusError):
        make_board(cardinality).search(limit=50)
    assert len(cardinality.pipeline_calls) == 2
    _assert_read_only(cardinality)
