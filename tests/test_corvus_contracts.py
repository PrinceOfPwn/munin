# tags: [tests, core, corvus, contracts, red-contract, tdd]
"""RED contract tests for ``munin.corvus.contracts`` — Munin v2 Corvus posts.

RED phase: tests only, no production code. Local execution is forbidden by
the ticket; the target module is imported lazily inside helpers so a missing
implementation surfaces as a clear pytest assertion failure rather than a
collection error.

Decision-complete contract (from the second architecture review):

Enums: member names remain uppercase exactly as approved; wire values are
lowercase strings:
    PostType.OPINION.value == "opinion"
    PostState.RESOLVED.value == "resolved"
    ReactionType.WORKED_FOR_ME.value == "worked_for_me"

Post types (exact): POST, QUESTION, REPLY, LESSON, WARNING, DIRECTIVE,
OPINION, OBSERVATION, HYPOTHESIS, REQUEST, ANSWER, INCIDENT.
States (exact): OPEN, ANSWERED, RESOLVED, STALE.
Reactions (exact): HELPFUL, CONFIRMED, WORKED_FOR_ME, CONTRADICTED,
NEEDS_EVIDENCE, STALE. No likes or vanity metrics.

Actor IDs are namespaced and durable: the fixture uses ``agent:raven-mind``,
and Post/Reaction ``actor_id`` values preserve it exactly.

Approved scopes: literal ``global``, or one of six prefixes (``actor``,
``conversation``, ``run``, ``investigation``, ``capability``, ``agent``)
followed by ``:`` and a nonempty remainder. Parsing splits only once at the
first colon, so the remainder may itself contain colons —
``actor:agent:raven-mind`` is valid. Empty remainder and unknown prefixes
are invalid. ``global:<id>`` is invalid.

Public error type: ``CorvusContractError`` — the exact one type any contract
violation raises. No adaptive discovery.

One explicit JSON wire API:
    Post.to_wire()
    Reaction.to_wire()
No ``model_dump``/``dict`` probing and no adaptive serialization helpers.

``Post.to_wire()`` must contain the exact canonical key set (17 keys):
    id, type, actor_id, content, scope, topics, published_at,
    published_at_local, timezone_name, edited_at, revision, confidence,
    evidence_refs, investigation_refs, reply_to, thread_root_id, status
Optional lifecycle keys are present only when non-null: observed_at, expires_at.

Every wire value is JSON-compatible. ``type`` and ``status`` are lowercase
strings. UTC timestamps are ISO-8601 with exactly milliseconds and ``Z``;
local timestamps use exactly milliseconds and a numeric offset.

``create_post(...)`` stamps server time from an injected clock/timezone and
rejects a forged ``published_at`` (caller-supplied server timestamps are not
accepted-and-ignored). ``edit_post(...)`` preserves
``published_at``, sets a new ``edited_at``, and increments ``revision``.
Replies have their own server timestamp and preserve ``reply_to`` and
``thread_root_id``.

``create_reaction(post_id, reaction_type, actor_id, clock)`` is the
server-side factory for reactions; the timestamp is server-generated UTC and
a forged ``timestamp`` argument is rejected.
``Reaction.to_wire()`` keys are exactly:
    id, post_id, type, actor_id, timestamp
``type`` is lowercase; ``timestamp`` is UTC ISO-8601 with exactly
milliseconds and ``Z``.

Canonical values/defaults for ``topics``, ``confidence``, ``evidence_refs``,
and ``investigation_refs`` are asserted explicitly.

OPINION and HYPOTHESIS stay distinct — no invented epistemic field.
Likes, vanity metrics, and unknown reaction names stay prohibited.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest


ACTOR_ID = "agent:raven-mind"


# ---------------------------------------------------------------------------
# Lazy import helpers — absence becomes a RED assertion failure.
# ---------------------------------------------------------------------------


def _contracts_module():
    try:
        from munin.corvus import contracts  # noqa: WPS433 — lazy import is required
    except ImportError as exc:  # pragma: no cover
        pytest.fail(
            "RED contract expects munin.corvus.contracts; ImportError: " + str(exc)
        )
    return contracts


def _require(symbol: str):
    obj = getattr(_contracts_module(), symbol, None)
    if obj is None:  # pragma: no cover
        pytest.fail(f"RED contract expects munin.corvus.contracts.{symbol}")
    return obj


def _utc(year, month, day, hour, minute, second, ms):
    return datetime(year, month, day, hour, minute, second, ms, tzinfo=timezone.utc)


def _create_post(**kwargs):
    """Build a Post with the approved defaults; override via kwargs."""
    create_post = _require("create_post")
    defaults = {
        "actor_id": ACTOR_ID,
        "post_type": "POST",
        "scope": "global",
        "content": "note",
        "clock": lambda tz: _utc(2026, 8, 8, 12, 0, 0, 0),
    }
    defaults.update(kwargs)
    return create_post(**defaults)  # type: ignore[call-arg]


def _edit_post(**kwargs):
    edit_post = _require("edit_post")
    return edit_post(**kwargs)  # type: ignore[call-arg]


def _create_reaction(**kwargs):
    create_reaction = _require("create_reaction")
    defaults = {
        "post_id": "post_munin-0001",
        "reaction_type": "HELPFUL",
        "actor_id": ACTOR_ID,
        "clock": lambda tz: _utc(2026, 8, 8, 14, 0, 0, 999_000),
    }
    defaults.update(kwargs)
    return create_reaction(**defaults)  # type: ignore[call-arg]


def _create_actor(**kwargs):
    """Build a concrete ActorIdentity with the approved defaults; override via kwargs."""
    ActorIdentity = _require("ActorIdentity")
    defaults = {
        "id": "agent:raven-mind",
        "name": "Raven Mind",
        "kind": "temporary_agent",
        "runtime": "deep_agent",
        "role": "Web Research Operator",
        "capabilities": ("browser", "web"),
        "model_route": "generated_specialist",
        "created_at": _utc(2026, 8, 8, 12, 0, 0, 0),
        "terminated_at": None,
    }
    defaults.update(kwargs)
    return ActorIdentity(**defaults)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Enums — exact sets + lowercase wire values.
# ---------------------------------------------------------------------------


def test_post_type_enum_is_exactly_the_documented_set_with_lowercase_values() -> None:
    PostType = _require("PostType")
    assert {member.name for member in PostType} == {  # type: ignore[attr-defined]
        "POST", "QUESTION", "REPLY", "LESSON", "WARNING", "DIRECTIVE",
        "OPINION", "OBSERVATION", "HYPOTHESIS", "REQUEST", "ANSWER", "INCIDENT",
    }
    # Wire values are lowercase strings; verify the representative examples
    # the review calls out plus the general rule for every member.
    for member in PostType:  # type: ignore[attr-defined]
        assert member.value == member.name.lower()
    assert PostType.OPINION.value == "opinion"  # type: ignore[attr-defined]


def test_post_state_enum_is_exactly_the_documented_set_with_lowercase_values() -> None:
    PostState = _require("PostState")
    assert {member.name for member in PostState} == {  # type: ignore[attr-defined]
        "OPEN", "ANSWERED", "RESOLVED", "STALE",
    }
    for member in PostState:  # type: ignore[attr-defined]
        assert member.value == member.name.lower()
    assert PostState.RESOLVED.value == "resolved"  # type: ignore[attr-defined]


def test_reaction_type_enum_is_exactly_the_documented_set_with_lowercase_values() -> None:
    ReactionType = _require("ReactionType")
    assert {member.name for member in ReactionType} == {  # type: ignore[attr-defined]
        "HELPFUL", "CONFIRMED", "WORKED_FOR_ME",
        "CONTRADICTED", "NEEDS_EVIDENCE", "STALE",
    }
    for member in ReactionType:  # type: ignore[attr-defined]
        assert member.value == member.name.lower()
    assert ReactionType.WORKED_FOR_ME.value == "worked_for_me"  # type: ignore[attr-defined]


def test_no_like_or_vanity_metric_is_public() -> None:
    contracts = _contracts_module()
    for enum_name in ("PostType", "PostState", "ReactionType"):
        enum = getattr(contracts, enum_name)
        assert "LIKE" not in {member.name for member in enum}  # type: ignore[attr-defined]
    for bad in ("likes", "like", "stars", "upvotes"):
        assert not hasattr(contracts, bad)


# ---------------------------------------------------------------------------
# ActorIdentity: exact public fields, immutable capabilities, durable attribution.
# ---------------------------------------------------------------------------


def test_actor_identity_exposes_exactly_the_public_contract_fields() -> None:
    actor = _create_actor()
    # Every public field exists with the exact approved value.
    assert actor.id == "agent:raven-mind"
    assert actor.name == "Raven Mind"
    assert actor.kind == "temporary_agent"
    assert actor.runtime == "deep_agent"
    assert actor.role == "Web Research Operator"
    assert actor.capabilities == ("browser", "web")
    assert actor.model_route == "generated_specialist"
    assert actor.created_at == _utc(2026, 8, 8, 12, 0, 0, 0)
    assert actor.terminated_at is None
    # The identity wire carries exactly the nine public fields, with the
    # created_at timestamp serialized as UTC ISO-8601 (ms + Z).
    wire = actor.to_wire()
    assert set(wire) == {
        "id", "name", "kind", "runtime", "role", "capabilities",
        "model_route", "created_at", "terminated_at",
    }
    assert wire["id"] == "agent:raven-mind"
    assert wire["created_at"] == _utc_ms_z(actor.created_at)


def test_actor_identity_capabilities_are_an_immutable_tuple() -> None:
    actor = _create_actor()
    assert isinstance(actor.capabilities, tuple), "capabilities must be an immutable tuple"
    with pytest.raises(TypeError):
        actor.capabilities[0] = "exfil"  # type: ignore[index]
    # Whole-field reassignment stays rejected on the frozen identity.
    with pytest.raises((TypeError, ValueError)):
        actor.capabilities = ("chat",)  # type: ignore[misc]


def test_terminated_temporary_actor_preserves_identity_and_attribution() -> None:
    terminated_at = _utc(2026, 8, 8, 18, 0, 0, 0)
    actor = _create_actor(kind="temporary_agent", terminated_at=terminated_at)
    assert actor.terminated_at == terminated_at
    wire = actor.to_wire()
    assert wire["id"] == "agent:raven-mind"
    assert wire["terminated_at"] == _utc_ms_z(terminated_at)
    # Termination never rewrites author attribution: a post produced by the
    # terminated actor still carries the same namespaced actor_id.
    post = _create_post(actor_id=actor.id)
    assert post.to_wire()["actor_id"] == "agent:raven-mind"


# ---------------------------------------------------------------------------
# create_post: injected clock, forge-resistance, actor_id preserved, local stamp.
# ---------------------------------------------------------------------------


def test_create_post_stamps_published_at_from_an_injected_clock() -> None:
    fixed = _utc(2026, 8, 8, 12, 34, 56, 789_000)
    post = _create_post(clock=lambda tz: fixed)
    assert post.published_at == fixed


def test_create_post_preserves_the_namespaced_actor_id_exactly() -> None:
    post = _create_post()
    wire = post.to_wire()
    assert wire["actor_id"] == ACTOR_ID


def test_create_post_rejects_a_forged_published_at() -> None:
    fixed = _utc(2026, 8, 8, 12, 0, 0, 0)
    forged = _utc(1999, 1, 1, 0, 0, 0, 0)
    # Caller-supplied server timestamps are forbidden, not accepted-and-ignored.
    with pytest.raises(_require("CorvusContractError")):
        _create_post(clock=lambda tz: fixed, published_at=forged)
    # Without the forged argument the injected clock still stamps server time.
    post = _create_post(clock=lambda tz: fixed)
    assert post.published_at == fixed


def test_create_post_emits_local_timestamp_with_milliseconds_and_numeric_offset() -> None:
    fixed_utc = _utc(2026, 8, 8, 10, 0, 0, 123_000)
    post = _create_post(
        post_type="OBSERVATION",
        clock=lambda tz: fixed_utc.astimezone(tz),
        timezone=ZoneInfo("Europe/Berlin"),
    )
    assert _iso_local_ms_with_offset(post.published_at_local)


# ---------------------------------------------------------------------------
# Canonical values / defaults for topics, confidence, evidence_refs,
# investigation_refs.
# ---------------------------------------------------------------------------


def test_default_post_topics_confidence_and_refs_canonical_values() -> None:
    post = _create_post()
    wire = post.to_wire()
    # Empty topics (not None, not missing) — canonical default.
    assert wire["topics"] == []
    # Confidence default is documented as None; GREEN may pick 0.0, but the
    # wire shape must be JSON-compatible and present.
    assert wire["confidence"] is None
    assert wire["evidence_refs"] == []
    assert wire["investigation_refs"] == []


def test_supplied_topics_confidence_and_refs_round_trip_to_wire() -> None:
    post = _create_post(
        topics=["ad", "kerberos"],
        confidence=0.8,
        evidence_refs=["ev-1", "ev-2"],
        investigation_refs=["inv-7"],
    )
    wire = post.to_wire()
    assert wire["topics"] == ["ad", "kerberos"]
    assert wire["confidence"] == 0.8
    assert wire["evidence_refs"] == ["ev-1", "ev-2"]
    assert wire["investigation_refs"] == ["inv-7"]


# ---------------------------------------------------------------------------
# Scope validation — split once at first colon; approved prefixes only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scope",
    [
        "global",
        f"actor:{ACTOR_ID}",                       # actor:agent:raven-mind — remainder has a colon
        "conversation:conv-1",
        "run:run-1",
        "investigation:inv-1",
        "capability:recon",
        "agent:raven-mind",
    ],
)
def test_scope_validation_accepts_each_approved_scope(scope: str) -> None:
    post = _create_post(scope=scope)
    assert post.scope == scope


@pytest.mark.parametrize(
    "bad_scope",
    [
        "",                       # empty
        "global:WEB01",           # global must not carry an id
        "asset:WEB01",            # unknown prefix
        "asset:agent:raven-mind", # unknown prefix even with a colon remainder
        "WEB01",                  # bare id, no prefix
        "GLOBAL",                 # wrong case for 'global'
        "actor:",                 # approved prefix, empty remainder
        "actor: ",                # whitespace remainder is blank
        "conversation",           # prefix without colon/remainder
    ],
)
def test_scope_validation_rejects_disallowed_scopes(bad_scope: str) -> None:
    with pytest.raises(_require("CorvusContractError")):
        _create_post(scope=bad_scope)


# ---------------------------------------------------------------------------
# OPINION and HYPOTHESIS stay distinct via type preservation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("post_type", ["OPINION", "HYPOTHESIS"])
def test_opinion_and_hypothesis_preserve_their_type_through_to_wire(post_type: str) -> None:
    post = _create_post(post_type=post_type, content="non-fact claim")
    wire = post.to_wire()
    assert wire["type"] == post_type.lower()
    assert wire["type"] != "observation"


# ---------------------------------------------------------------------------
# edit_post: published_at preserved, edited_at set, revision incremented.
# ---------------------------------------------------------------------------


def test_edit_post_preserves_published_at_sets_edited_at_and_increments_revision() -> None:
    first_at = _utc(2026, 8, 8, 12, 0, 0, 0)
    edit_at = _utc(2026, 8, 8, 13, 30, 0, 500_000)
    post = _create_post(post_type="LESSON", clock=lambda tz: first_at)
    original_published = post.published_at
    original_revision = post.revision

    edited = _edit_post(post=post, content="revised lesson", clock=lambda tz: edit_at)
    assert edited.published_at == original_published
    assert edited.edited_at == edit_at
    assert edited.revision == original_revision + 1

    wire = edited.to_wire()
    assert _iso_utc_z_with_ms(wire["published_at"])
    assert _iso_utc_z_with_ms(wire["edited_at"])


def test_edit_post_rejects_a_rewrite_of_published_at() -> None:
    first_at = _utc(2026, 8, 8, 12, 0, 0, 0)
    forged = _utc(1999, 1, 1, 0, 0, 0, 0)
    post = _create_post(post_type="WARNING", clock=lambda tz: first_at)
    # Edits must not be able to rewrite the server-owned published_at either.
    with pytest.raises(_require("CorvusContractError")):
        _edit_post(
            post=post,
            content="revised",
            clock=lambda tz: first_at,
            published_at=forged,
        )


# ---------------------------------------------------------------------------
# Replies: own server timestamp + preserve reply_to and thread_root_id.
# ---------------------------------------------------------------------------


def test_reply_has_own_timestamp_and_preserves_thread_links() -> None:
    root_at = _utc(2026, 8, 8, 9, 0, 0, 0)
    reply_at = _utc(2026, 8, 8, 9, 5, 30, 250_000)
    question = _create_post(post_type="QUESTION", content="What is the exposure of WEB01?", clock=lambda tz: root_at)
    root_id = question.id
    reply = _create_post(
        post_type="REPLY",
        content="Port 443 only.",
        clock=lambda tz: reply_at,
        reply_to=root_id,
        thread_root_id=root_id,
    )
    assert reply.published_at == reply_at
    assert reply.published_at != question.published_at
    wire = reply.to_wire()
    assert wire["reply_to"] == root_id
    assert wire["thread_root_id"] == root_id
    assert wire["type"] == "reply"


# ---------------------------------------------------------------------------
# Reactions: server-side factory, exact wire keys, forge-resistant timestamp.
# ---------------------------------------------------------------------------


def test_create_reaction_records_actor_and_exact_server_timestamp() -> None:
    fixed = _utc(2026, 8, 8, 14, 0, 0, 999_000)
    reaction = _create_reaction(clock=lambda tz: fixed)
    assert reaction.actor_id == ACTOR_ID
    assert reaction.timestamp == fixed


def test_create_reaction_rejects_a_forged_timestamp() -> None:
    fixed = _utc(2026, 8, 8, 14, 0, 0, 999_000)
    forged = _utc(1999, 1, 1, 0, 0, 0, 0)
    # Caller-supplied reaction timestamps are forbidden, not accepted-and-ignored.
    with pytest.raises(_require("CorvusContractError")):
        _create_reaction(clock=lambda tz: fixed, timestamp=forged)


def test_create_reaction_rejects_unsupported_like() -> None:
    with pytest.raises(_require("CorvusContractError")):
        _create_reaction(reaction_type="LIKE")


@pytest.mark.parametrize(
    "reaction_type",
    ["HELPFUL", "CONFIRMED", "WORKED_FOR_ME", "CONTRADICTED", "NEEDS_EVIDENCE", "STALE"],
)
def test_each_documented_reaction_kind_is_accepted(reaction_type: str) -> None:
    reaction = _create_reaction(reaction_type=reaction_type)
    ReactionType = _require("ReactionType")
    # Exact enum identity — not a string-suffix or adaptive comparison.
    assert reaction.type is ReactionType[reaction_type]
    assert reaction.type == ReactionType[reaction_type]


# ---------------------------------------------------------------------------
# Serialization: Post.to_wire() + Reaction.to_wire() — one exact API.
# ---------------------------------------------------------------------------


CANONICAL_POST_KEYS = {
    "id", "type", "actor_id", "content", "scope", "topics",
    "published_at", "published_at_local", "timezone_name", "edited_at",
    "revision", "confidence", "evidence_refs", "investigation_refs",
    "reply_to", "thread_root_id", "status",
}

OPTIONAL_LIFECYCLE_KEYS = {"observed_at", "expires_at"}


def test_post_to_wire_contains_exactly_the_canonical_keys_when_lifecycle_absent() -> None:
    post = _create_post()
    wire = post.to_wire()
    assert set(wire) == CANONICAL_POST_KEYS


def test_post_to_wire_observed_at_is_exact_utc_ms_and_expires_at_is_absent() -> None:
    post = _create_post(observed_at=_utc(2026, 8, 8, 8, 0, 0, 0))
    wire = post.to_wire()
    # The supplied observed_at must survive to the wire exactly, and a null
    # expires_at must stay absent — the implementation cannot discard either.
    extra = set(wire) - CANONICAL_POST_KEYS
    assert extra == {"observed_at"}
    assert extra <= OPTIONAL_LIFECYCLE_KEYS
    assert wire["observed_at"] == _utc_ms_z(_utc(2026, 8, 8, 8, 0, 0, 0))
    assert "expires_at" not in wire


def test_post_to_wire_expires_at_is_exact_utc_ms_when_observed_at_absent() -> None:
    post = _create_post(expires_at=_utc(2026, 8, 20, 23, 59, 59, 999_000))
    wire = post.to_wire()
    extra = set(wire) - CANONICAL_POST_KEYS
    assert extra == {"expires_at"}
    assert extra <= OPTIONAL_LIFECYCLE_KEYS
    assert wire["expires_at"] == _utc_ms_z(_utc(2026, 8, 20, 23, 59, 59, 999_000))
    assert "observed_at" not in wire


def test_post_to_wire_preserves_both_lifecycle_fields_when_both_non_null() -> None:
    post = _create_post(
        observed_at=_utc(2026, 8, 8, 8, 0, 0, 0),
        expires_at=_utc(2026, 8, 20, 23, 59, 59, 999_000),
    )
    wire = post.to_wire()
    extra = set(wire) - CANONICAL_POST_KEYS
    assert extra == {"observed_at", "expires_at"}
    assert extra <= OPTIONAL_LIFECYCLE_KEYS
    assert wire["observed_at"] == _utc_ms_z(_utc(2026, 8, 8, 8, 0, 0, 0))
    assert wire["expires_at"] == _utc_ms_z(_utc(2026, 8, 20, 23, 59, 59, 999_000))


def test_post_to_wire_status_and_type_are_lowercase_strings() -> None:
    post = _create_post(post_type="WARNING")
    wire = post.to_wire()
    assert wire["type"] == "warning"
    assert wire["status"] == "open"


def test_post_to_wire_published_at_is_utc_z_with_milliseconds() -> None:
    fixed = _utc(2026, 8, 8, 12, 34, 56, 789_000)
    post = _create_post(clock=lambda tz: fixed)
    wire = post.to_wire()
    assert _iso_utc_z_with_ms(wire["published_at"])


def test_post_to_wire_local_timestamp_carries_milliseconds_and_numeric_offset() -> None:
    fixed_utc = _utc(2026, 8, 8, 10, 0, 0, 123_000)
    post = _create_post(
        clock=lambda tz: fixed_utc.astimezone(tz),
        timezone=ZoneInfo("America/Sao_Paulo"),
    )
    wire = post.to_wire()
    assert _iso_local_ms_with_offset(wire["published_at_local"])
    # The configured timezone name is itself part of the wire (17-key set),
    # recorded in addition to published_at_local.
    assert wire["timezone_name"] == "America/Sao_Paulo"


def test_reaction_to_wire_uses_exact_canonical_keys() -> None:
    fixed = _utc(2026, 8, 8, 14, 0, 0, 999_000)
    reaction = _create_reaction(clock=lambda tz: fixed)
    wire = reaction.to_wire()
    assert set(wire) == {"id", "post_id", "type", "actor_id", "timestamp"}


def test_reaction_to_wire_type_is_lowercase_and_timestamp_is_utc_z_with_ms() -> None:
    fixed = _utc(2026, 8, 8, 14, 0, 0, 999_000)
    reaction = _create_reaction(clock=lambda tz: fixed)
    wire = reaction.to_wire()
    assert wire["type"] == "helpful"
    assert wire["actor_id"] == ACTOR_ID
    assert _iso_utc_z_with_ms(wire["timestamp"])


def test_post_and_reaction_wires_are_json_compatible_with_plain_dumps() -> None:
    post = _create_post(
        post_type="OBSERVATION",
        confidence=0.5,
        observed_at=_utc(2026, 8, 8, 8, 0, 0, 0),
        expires_at=_utc(2026, 8, 20, 23, 59, 59, 999_000),
    )
    reaction = _create_reaction(reaction_type="CONFIRMED")
    # No fallback serializer: plain json.dumps must not raise, and the wire
    # must round-trip through json.loads unchanged.
    post_wire = post.to_wire()
    reaction_wire = reaction.to_wire()
    assert json.dumps(post_wire)
    assert json.dumps(reaction_wire)
    assert json.loads(json.dumps(post_wire)) == post_wire
    assert json.loads(json.dumps(reaction_wire)) == reaction_wire


def test_generated_ids_use_stable_namespaced_prefixes() -> None:
    post = _create_post()
    reaction = _create_reaction()
    assert post.id.startswith("post_")
    assert reaction.id.startswith("reaction_")


# ---------------------------------------------------------------------------
# ISO millisecond literal matchers.
# ---------------------------------------------------------------------------


def _utc_ms_z(value: datetime) -> str:
    """Wire serialization of a UTC timestamp: ISO-8601, exactly ms, trailing Z."""
    utc = value.astimezone(timezone.utc)
    ms = utc.microsecond // 1000
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{ms:03d}Z"


def _iso_utc_z_with_ms(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value))


def _iso_local_ms_with_offset(value: str) -> bool:
    return bool(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}", value)
    )
