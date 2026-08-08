# tags: [core, corvus, contracts, blackboard, identity, posts, reactions, domain-events, pydantic, enum, frozen, json-serialization, scope-validation, forge-resistance]
"""Corvus domain contracts — actors, posts, and reactions.

Pure domain value objects plus the server-side factories that stamp server
time from an injected clock. No storage, transport, retries, or background
tasks live here; those arrive in later tickets.

Design notes
------------

* **Frozen, closed Pydantic v2 models.** ``model_config = ConfigDict(frozen=True,
  extra="forbid")`` makes whole-field reassignment raise ``ValidationError`` (a
  ``ValueError`` subclass) and rejects unknown constructor fields, satisfying
  the RED contract's ``pytest.raises((TypeError, ValueError))``. Tuple item
  assignment raises ``TypeError`` natively. Collection fields
  (``capabilities``, ``topics``, ``evidence_refs``, ``investigation_refs``)
  are immutable tuples on the model; ``to_wire()`` projects them to JSON lists
  at the boundary.
* **Server-owned timestamps and IDs.** Factories inject a clock callable
  ``Callable[[tzinfo], datetime]`` and use a private sentinel default for
  ``published_at`` / reaction ``timestamp`` — *any* explicit supply, even
  ``None``, is rejected (forge-resistance, not accept-and-ignore). IDs are
  ``post_`` / ``reaction_`` prefixed with collision-resistant ``uuid4`` hex.
* **Strict enums.** ``str, Enum`` with lowercase values. The factories accept
  enum instances or member *names* (unambiguous), and reject unknown names
  (including ``LIKE``) by shrinking Pydantic's broad ``ValidationError`` into
  the contract's single ``CorvusContractError``.
* **Exact wire API.** ``to_wire()`` emits a JSON-compatible ``dict`` with the
  exact canonical key set. ``@model_serializer`` is intentionally not used:
  the wire shape differs from Pydantic's ``model_dump`` (selective lifecycle
  keys, lowercase enum strings, custom timestamp formatting), so a hand-written
  serializer is the precise contract.
* **Scope validation.** ``global`` literal, or ``prefix:remainder`` split
  *once* at the first colon — remainder may itself contain colons
  (``actor:agent:raven-mind``).
* **Error normalization.** Every factory boundary converts Pydantic
  ``ValidationError``, ``ZoneInfoNotFoundError``, and raw ``TypeError`` from
  invalid caller input into ``CorvusContractError`` with sanitized messages
  (error locations and types only — never raw input values).

Architectural boundaries (from DeepWiki on ``langchain-ai/deepagents``):
``deepagents`` models ``AgentState`` as a ``TypedDict`` with a ``DeltaChannel``
messages reducer; subagent identity lives on the supervisor's
``SubAgentMiddleware``, and transient domain events are excluded from the
LangGraph checkpoint via ``_EXCLUDED_STATE_KEYS``. Posts, reactions, and actor
identities are therefore domain records here — never packed into graph state —
so checkpoint growth stays bounded (the DeltaChannel is the only growth
channel) and transport replay restores them independently of the executor.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------------------------------------------------------------------------
# Private sentinel — distinguishes "not supplied" from an explicit ``None``.
# ---------------------------------------------------------------------------


_UNSET = object()
"""Private sentinel marking a server-owned factory argument as "not supplied".

Factory parameters for server-owned values (``published_at``, reaction
``timestamp``) default to this sentinel so that *any* explicit caller supply —
including ``None`` — is detected and rejected as a server-ownership violation.
"""


# ---------------------------------------------------------------------------
# Public error — the exact one type for every contract violation.
# ---------------------------------------------------------------------------


class CorvusContractError(Exception):
    """The single public error for any Corvus contract violation.

    Every validation failure in this module — bad enums, forged timestamps,
    invalid scopes, unknown reactions — collapses to this type so callers
    catch one thing. Pydantic's ``ValidationError`` is converted into it at
    the factory boundary so adaptive discovery cannot leak through.
    """


# ---------------------------------------------------------------------------
# Enums — exact sets, names uppercase, lowercase wire values.
# ---------------------------------------------------------------------------


class PostType(str, enum.Enum):
    POST = "post"
    QUESTION = "question"
    REPLY = "reply"
    LESSON = "lesson"
    WARNING = "warning"
    DIRECTIVE = "directive"
    OPINION = "opinion"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    REQUEST = "request"
    ANSWER = "answer"
    INCIDENT = "incident"


class PostState(str, enum.Enum):
    OPEN = "open"
    ANSWERED = "answered"
    RESOLVED = "resolved"
    STALE = "stale"


class ReactionType(str, enum.Enum):
    HELPFUL = "helpful"
    CONFIRMED = "confirmed"
    WORKED_FOR_ME = "worked_for_me"
    CONTRADICTED = "contradicted"
    NEEDS_EVIDENCE = "needs_evidence"
    STALE = "stale"


_APPROVED_SCOPE_PREFIXES: frozenset[str] = frozenset(
    {"actor", "conversation", "run", "investigation", "capability", "agent"}
)


# ---------------------------------------------------------------------------
# Timestamp rendering helpers.
# ---------------------------------------------------------------------------


def _utc_ms_z(value: datetime) -> str:
    """ISO-8601 UTC with exactly milliseconds and a trailing ``Z``.

    Normalizes any aware datetime to UTC, truncates microseconds to a
    millisecond field, and renders the canonical wire form
    ``YYYY-MM-DDTHH:MM:SS.mmmZ``.
    """
    utc = value.astimezone(timezone.utc)
    ms = utc.microsecond // 1000
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{ms:03d}Z"


def _local_ms_with_offset(value: datetime) -> str:
    """ISO-8601 local with exactly milliseconds and a numeric offset.

    Renders the canonical local wire form
    ``YYYY-MM-DDTHH:MM:SS.mmm+HH:MM``. Naive datetimes are rejected upstream;
    only aware datetimes reach this helper.
    """
    local = value  # already aware and in the target tz
    ms = local.microsecond // 1000
    offset = local.utcoffset()
    if offset is None:  # pragma: no cover — defensive; factories inject aware
        raise CorvusContractError("published_at_local must be timezone-aware")
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    return f"{local:%Y-%m-%dT%H:%M:%S}.{ms:03d}{sign}{hours:02d}:{minutes:02d}"


def _new_id(prefix: str) -> str:
    """Server-owned, collision-resistant id: ``<prefix>_<uuid4 hex>``."""
    return f"{prefix}_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Coercion and normalization helpers — all violations raise
# ``CorvusContractError``, never a raw Pydantic/zoneinfo/TypeError leak.
# ---------------------------------------------------------------------------


def _resolve_timezone(timezone: Any) -> ZoneInfo:
    """Resolve a caller timezone into an IANA ``ZoneInfo``.

    Accepts ``None`` (defaults to ``UTC``), an existing ``ZoneInfo``, or an
    IANA key string (e.g. ``"America/Sao_Paulo"``). Unknown IANA names and
    unsupported types raise ``CorvusContractError`` — never
    ``ZoneInfoNotFoundError`` — so both ``create_post`` and ``edit_post``
    normalize timezone input through this single helper.
    """
    if timezone is None:
        return ZoneInfo("UTC")
    if isinstance(timezone, ZoneInfo):
        return timezone
    if isinstance(timezone, str):
        try:
            return ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise CorvusContractError(f"unknown timezone: {timezone!r}") from exc
    raise CorvusContractError(
        f"timezone must be a str IANA key or ZoneInfo, got {type(timezone).__name__}"
    )


def _coerce_post_type(value: Any) -> PostType:
    """Accept a ``PostType`` instance or an unambiguous member name/value.

    Rejects unknown names (including any future ``LIKE``) and ambiguous inputs
    with ``CorvusContractError``.
    """
    if isinstance(value, PostType):
        return value
    if isinstance(value, str):
        name = value.strip()
        # Member names are uppercase; values are lowercase. A bare uppercase
        # name is the canonical caller form. Accept the lowercase value too
        # only when it maps to exactly one member (always true here, since
        # values are unique), but keep it unambiguous by resolving via name
        # first and falling back to value lookup.
        if name in PostType.__members__:
            return PostType.__members__[name]
        for member in PostType:
            if member.value == name:
                return member
    raise CorvusContractError(f"unknown post type: {value!r}")


def _coerce_post_state(value: Any) -> PostState:
    """Accept a ``PostState`` instance or an unambiguous member name/value.

    Rejects unknown states with ``CorvusContractError``. An invalid state is
    never silently coerced to ``OPEN``.
    """
    if isinstance(value, PostState):
        return value
    if isinstance(value, str):
        name = value.strip()
        if name in PostState.__members__:
            return PostState.__members__[name]
        for member in PostState:
            if member.value == name:
                return member
    raise CorvusContractError(f"unknown post state: {value!r}")


def _coerce_reaction_type(value: Any) -> ReactionType:
    """Accept a ``ReactionType`` instance or an unambiguous member name/value.

    Rejects unknown names (including ``LIKE``) with ``CorvusContractError``.
    """
    if isinstance(value, ReactionType):
        return value
    if isinstance(value, str):
        name = value.strip()
        if name in ReactionType.__members__:
            return ReactionType.__members__[name]
        for member in ReactionType:
            if member.value == name:
                return member
    raise CorvusContractError(f"unknown reaction type: {value!r}")


def _coerce_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Coerce an optional iterable of strings into an immutable tuple.

    ``None`` yields an empty tuple. A bare string or a non-iterable is
    rejected with ``CorvusContractError``, and non-string entries are rejected
    too — invalid caller input can never reach Pydantic as a raw ``TypeError``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        raise CorvusContractError(f"{field_name} must be an iterable of strings")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise CorvusContractError(
            f"{field_name} must be an iterable of strings"
        ) from exc
    for item in items:
        if not isinstance(item, str):
            raise CorvusContractError(
                f"{field_name} entries must be strings, got {type(item).__name__}"
            )
    return items


def _validate_scope(scope: Any) -> str:
    """Validate an approved Corvus scope.

    Accepted forms:
      * the literal ``global`` (case-sensitive), or
      * ``<prefix>:<nonempty remainder>`` where ``prefix`` is one of the six
        approved prefixes and ``remainder`` is nonempty after stripping.
        Splitting happens *once* at the first colon, so the remainder may
        itself contain colons (``actor:agent:raven-mind``).
    """
    if not isinstance(scope, str):
        raise CorvusContractError(f"scope must be a string: {scope!r}")
    if scope == "global":
        return scope
    if not scope:
        raise CorvusContractError("scope must not be empty")
    prefix, sep, remainder = scope.partition(":")
    if not sep:
        # No colon and not 'global' — bare id or prefix without remainder.
        raise CorvusContractError(f"unknown scope: {scope!r}")
    if prefix not in _APPROVED_SCOPE_PREFIXES:
        raise CorvusContractError(f"unknown scope prefix: {prefix!r}")
    if not remainder.strip():
        raise CorvusContractError(
            f"scope remainder must be nonempty: {scope!r}"
        )
    return scope


def validate_scope(scope: Any) -> str:
    """Public thin wrapper over the existing ``_validate_scope`` check.

    Delegates entirely to ``_validate_scope`` so there is exactly one set of
    validation rules; no rules are duplicated here and ``create_post`` keeps
    calling ``_validate_scope`` directly, so its behavior is unchanged.
    """
    return _validate_scope(scope)


def coerce_post_state(value: Any) -> PostState:
    """Public thin wrapper over the existing ``_coerce_post_state`` check.

    Delegates entirely to ``_coerce_post_state`` so there is exactly one set of
    coercion rules; no rules are duplicated here and ``create_post`` keeps
    calling ``_coerce_post_state`` directly, so its behavior is unchanged.
    """
    return _coerce_post_state(value)


def _normalize_aware_utc(value: datetime) -> datetime:
    """Normalize a datetime to aware UTC.

    Non-datetime input is rejected with ``CorvusContractError``. Naive
    datetimes are assumed UTC (factories inject aware clocks, so this is
    defensive only). Aware datetimes are converted to UTC.
    """
    if not isinstance(value, datetime):
        raise CorvusContractError(
            f"expected a datetime, got {type(value).__name__}"
        )
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _sanitize_validation_error(exc: ValidationError) -> str:
    """Build a sanitized contract error message from a Pydantic failure.

    Reports only error locations and types — never raw input values — so a
    converted ``CorvusContractError`` cannot leak caller data to adaptive
    discovery.
    """
    details = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in err.get("loc", ()))
        err_type = err.get("type", "validation_error")
        details.append(f"{loc}: {err_type}")
    if not details:
        return "invalid contract value"
    return "invalid contract value: " + "; ".join(details)


# ---------------------------------------------------------------------------
# ActorIdentity — frozen value object with the nine public fields.
# ---------------------------------------------------------------------------


class ActorIdentity(BaseModel):
    """An immutable, namespaced actor identity with durable attribution.

    Six approved ``kind`` values: ``permanent_agent``, ``temporary_agent``,
    ``operator``, ``capability``, ``system``, ``external``. Capabilities are
    stored as an immutable tuple so item assignment raises ``TypeError`` and
    whole-field reassignment raises ``ValidationError`` (a ``ValueError``
    subclass) on the frozen model. ``extra="forbid"`` rejects unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    kind: str = "temporary_agent"
    runtime: str = "deep_agent"
    role: str
    capabilities: tuple[str, ...] = ()
    model_route: str = "generated_specialist"
    created_at: datetime
    terminated_at: datetime | None = None

    def to_wire(self) -> dict[str, Any]:
        """Emit exactly the nine public fields with JSON-compatible values."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "runtime": self.runtime,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "model_route": self.model_route,
            "created_at": _utc_ms_z(self.created_at),
            "terminated_at": (
                _utc_ms_z(self.terminated_at)
                if self.terminated_at is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# ActorPresence — frozen value object with the three public fields.
# ---------------------------------------------------------------------------


class ActorPresence(BaseModel):
    """An immutable actor presence snapshot stamped by a server heartbeat.

    ``heartbeat_at`` is the server-owned UTC instant; ``expires_at`` is the
    hard TTL horizon (``heartbeat_at + ttl_seconds``). Both render as
    ISO-8601 UTC with exactly milliseconds and ``Z`` on the wire.
    ``extra="forbid"`` rejects unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str
    heartbeat_at: datetime
    expires_at: datetime

    def to_wire(self) -> dict[str, Any]:
        """Emit exactly the three canonical keys with JSON-compatible values."""
        return {
            "actor_id": self.actor_id,
            "heartbeat_at": _utc_ms_z(self.heartbeat_at),
            "expires_at": _utc_ms_z(self.expires_at),
        }


# ---------------------------------------------------------------------------
# ActorSubscriptions — frozen value object with the four public fields.
# ---------------------------------------------------------------------------


class ActorSubscriptions(BaseModel):
    """An immutable actor subscription snapshot stamped by the server.

    ``topics`` and ``scopes`` are immutable tuples on the model; ``to_wire()``
    projects them to JSON lists. ``updated_at`` is the server-owned UTC instant
    rendered as ISO-8601 UTC with exactly milliseconds and ``Z``.
    ``extra="forbid"`` rejects unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str
    topics: tuple[str, ...] = Field(default_factory=tuple)
    scopes: tuple[str, ...] = Field(default_factory=tuple)
    updated_at: datetime

    def to_wire(self) -> dict[str, Any]:
        """Emit exactly the four canonical keys with JSON-compatible values."""
        return {
            "actor_id": self.actor_id,
            "topics": list(self.topics),
            "scopes": list(self.scopes),
            "updated_at": _utc_ms_z(self.updated_at),
        }


# ---------------------------------------------------------------------------
# Post — frozen value object.
# ---------------------------------------------------------------------------


class Post(BaseModel):
    """An immutable Corvus post value object.

    Server-owned fields (``id``, ``published_at``, ``published_at_local``,
    ``timezone_name``, ``edited_at``, ``revision``) are set only by the
    factories. ``published_at_local`` is the canonical ISO-8601 string
    (exactly milliseconds and a numeric offset) rendered server-side from the
    timezone-aware instant; ``to_wire()`` emits it unchanged. Collection
    fields are immutable tuples internally; ``to_wire()`` projects them to
    JSON lists. The canonical wire set is 17 keys; optional lifecycle keys
    (``observed_at``, ``expires_at``) are emitted only when non-null.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: PostType
    actor_id: str
    content: str
    scope: str
    topics: tuple[str, ...] = Field(default_factory=tuple)
    published_at: datetime
    published_at_local: str
    timezone_name: str = "UTC"
    edited_at: datetime | None = None
    revision: int = 1
    confidence: float | None = None
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    investigation_refs: tuple[str, ...] = Field(default_factory=tuple)
    reply_to: str | None = None
    thread_root_id: str | None = None
    status: PostState = PostState.OPEN
    observed_at: datetime | None = None
    expires_at: datetime | None = None

    def to_wire(self) -> dict[str, Any]:
        """Emit the exact canonical key set with JSON-compatible values.

        ``type`` and ``status`` are lowercase strings. UTC timestamps
        (``published_at``, ``edited_at``, ``observed_at``, ``expires_at``) are
        ISO-8601 with exactly milliseconds and ``Z``. The local timestamp
        carries exactly milliseconds and a numeric offset. Optional lifecycle
        keys are present only when non-null. Immutable tuple collections are
        projected to JSON lists.
        """
        wire: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "actor_id": self.actor_id,
            "content": self.content,
            "scope": self.scope,
            "topics": list(self.topics),
            "published_at": _utc_ms_z(self.published_at),
            "published_at_local": self.published_at_local,
            "timezone_name": self.timezone_name,
            "edited_at": (
                _utc_ms_z(self.edited_at) if self.edited_at is not None else None
            ),
            "revision": self.revision,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "investigation_refs": list(self.investigation_refs),
            "reply_to": self.reply_to,
            "thread_root_id": self.thread_root_id,
            "status": self.status.value,
        }
        if self.observed_at is not None:
            wire["observed_at"] = _utc_ms_z(self.observed_at)
        if self.expires_at is not None:
            wire["expires_at"] = _utc_ms_z(self.expires_at)
        return wire


# ---------------------------------------------------------------------------
# Reaction — frozen value object.
# ---------------------------------------------------------------------------


class Reaction(BaseModel):
    """An immutable Corvus reaction value object.

    Five public fields; ``timestamp`` and ``id`` are server-owned.
    ``extra="forbid"`` rejects unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    post_id: str
    type: ReactionType
    actor_id: str
    timestamp: datetime

    def to_wire(self) -> dict[str, Any]:
        """Emit exactly the five canonical keys with JSON-compatible values."""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "type": self.type.value,
            "actor_id": self.actor_id,
            "timestamp": _utc_ms_z(self.timestamp),
        }


# ---------------------------------------------------------------------------
# Server-side factories.
# ---------------------------------------------------------------------------


def create_post(
    *,
    actor_id: str,
    post_type: Any,
    scope: Any,
    content: str,
    clock: Callable[[Any], datetime],
    timezone: Any = None,
    topics: Iterable[str] | None = None,
    confidence: float | None = None,
    evidence_refs: Iterable[str] | None = None,
    investigation_refs: Iterable[str] | None = None,
    reply_to: str | None = None,
    thread_root_id: str | None = None,
    status: Any = PostState.OPEN,
    observed_at: datetime | None = None,
    expires_at: datetime | None = None,
    published_at: Any = _UNSET,
    **unexpected: Any,
) -> Post:
    """Server-side factory for a Corvus post.

    Stamps server time from an injected ``clock(tz)`` callable, normalizes
    ``published_at`` to aware UTC, renders ``published_at_local`` as the
    canonical ISO-8601 string (exactly milliseconds and a numeric offset) in
    the configured timezone, and rejects any caller-supplied ``published_at``
    (forge-resistance via a private sentinel — even an explicit ``None`` is
    rejected, not accepted-and-ignored). Unknown keyword arguments are
    rejected by name. Validates scope, post type, and post state with
    ``CorvusContractError`` on violation, and converts Pydantic
    ``ValidationError`` / raw ``TypeError`` from invalid caller input into
    ``CorvusContractError`` with sanitized messages.
    """
    if unexpected:
        raise CorvusContractError(
            "unexpected keyword arguments: " + ", ".join(sorted(unexpected))
        )
    if published_at is not _UNSET:
        raise CorvusContractError(
            "published_at is server-owned; caller-supplied values are forbidden"
        )
    if not callable(clock):
        raise CorvusContractError(
            f"clock must be callable, got {type(clock).__name__}"
        )

    resolved_tz = _resolve_timezone(timezone)
    resolved_scope = _validate_scope(scope)
    resolved_type = _coerce_post_type(post_type)
    resolved_status = _coerce_post_state(status)

    clock_value = clock(resolved_tz)
    if not isinstance(clock_value, datetime):
        raise CorvusContractError(
            f"clock must return a datetime, got {type(clock_value).__name__}"
        )

    published_utc = _normalize_aware_utc(clock_value)
    if clock_value.tzinfo is None:
        local_value = published_utc
    else:
        local_value = clock_value.astimezone(resolved_tz)
    published_local_str = _local_ms_with_offset(local_value)

    topics_tuple = _coerce_string_tuple(topics, "topics")
    evidence_tuple = _coerce_string_tuple(evidence_refs, "evidence_refs")
    investigation_tuple = _coerce_string_tuple(
        investigation_refs, "investigation_refs"
    )
    observed_utc = (
        _normalize_aware_utc(observed_at) if observed_at is not None else None
    )
    expires_utc = (
        _normalize_aware_utc(expires_at) if expires_at is not None else None
    )

    # Store the IANA zone key (e.g. "America/Sao_Paulo", "UTC"), not the
    # timezone abbreviation, so the wire carries a stable, round-trippable
    # identifier alongside published_at_local.
    tz_name = resolved_tz.key

    try:
        return Post(
            id=_new_id("post"),
            type=resolved_type,
            actor_id=actor_id,
            content=content,
            scope=resolved_scope,
            topics=topics_tuple,
            published_at=published_utc,
            published_at_local=published_local_str,
            timezone_name=tz_name,
            status=resolved_status,
            confidence=confidence,
            evidence_refs=evidence_tuple,
            investigation_refs=investigation_tuple,
            reply_to=reply_to,
            thread_root_id=thread_root_id,
            observed_at=observed_utc,
            expires_at=expires_utc,
        )
    except ValidationError as exc:
        raise CorvusContractError(_sanitize_validation_error(exc)) from exc
    except TypeError as exc:
        raise CorvusContractError("invalid contract value: internal type error") from exc


def edit_post(
    *,
    post: Post,
    content: str,
    clock: Callable[[Any], datetime],
    timezone: Any = None,
    published_at: Any = _UNSET,
    **unexpected: Any,
) -> Post:
    """Server-side factory for an edited post.

    Returns a new ``Post`` that preserves ``published_at`` and ``id`` from the
    original, sets a new ``edited_at`` from the clock, and increments
    ``revision``. Rejects any caller-supplied ``published_at`` (private
    sentinel, so even an explicit ``None`` is a violation) to prevent
    rewriting the server-owned publication timestamp, and rejects unknown
    keyword arguments by name. Timezone strings resolve through the same
    helper as ``create_post``. Never mutates the original (frozen model →
    returns a new instance).
    """
    if unexpected:
        raise CorvusContractError(
            "unexpected keyword arguments: " + ", ".join(sorted(unexpected))
        )
    if published_at is not _UNSET:
        raise CorvusContractError(
            "published_at is server-owned; it cannot be rewritten on edit"
        )
    if not isinstance(post, Post):
        raise CorvusContractError(
            f"post must be a Post instance, got {type(post).__name__}"
        )
    if not isinstance(content, str):
        raise CorvusContractError(
            f"content must be a string, got {type(content).__name__}"
        )
    if not callable(clock):
        raise CorvusContractError(
            f"clock must be callable, got {type(clock).__name__}"
        )

    resolved_tz = _resolve_timezone(timezone)
    clock_value = clock(resolved_tz)
    if not isinstance(clock_value, datetime):
        raise CorvusContractError(
            f"clock must return a datetime, got {type(clock_value).__name__}"
        )
    edited_utc = _normalize_aware_utc(clock_value)

    try:
        return post.model_copy(
            update={
                "content": content,
                "edited_at": edited_utc,
                "revision": post.revision + 1,
            }
        )
    except ValidationError as exc:
        raise CorvusContractError(_sanitize_validation_error(exc)) from exc
    except TypeError as exc:
        raise CorvusContractError("invalid contract value: internal type error") from exc


def create_reaction(
    *,
    post_id: str,
    reaction_type: Any,
    actor_id: str,
    clock: Callable[[Any], datetime],
    timestamp: Any = _UNSET,
    **unexpected: Any,
) -> Reaction:
    """Server-side factory for a Corvus reaction.

    Stamps the timestamp from an injected ``clock(tz)`` callable (aware UTC)
    and rejects any caller-supplied ``timestamp`` (forge-resistance via a
    private sentinel — even an explicit ``None`` is rejected). Rejects unknown
    reaction types — including ``LIKE`` — with ``CorvusContractError``, and
    converts Pydantic ``ValidationError`` / raw ``TypeError`` from invalid
    caller input into ``CorvusContractError`` with sanitized messages.
    """
    if unexpected:
        raise CorvusContractError(
            "unexpected keyword arguments: " + ", ".join(sorted(unexpected))
        )
    if timestamp is not _UNSET:
        raise CorvusContractError(
            "timestamp is server-owned; caller-supplied values are forbidden"
        )
    if not callable(clock):
        raise CorvusContractError(
            f"clock must be callable, got {type(clock).__name__}"
        )

    resolved_type = _coerce_reaction_type(reaction_type)

    clock_value = clock(ZoneInfo("UTC"))
    if not isinstance(clock_value, datetime):
        raise CorvusContractError(
            f"clock must return a datetime, got {type(clock_value).__name__}"
        )
    timestamp_utc = _normalize_aware_utc(clock_value)

    try:
        return Reaction(
            id=_new_id("reaction"),
            post_id=post_id,
            type=resolved_type,
            actor_id=actor_id,
            timestamp=timestamp_utc,
        )
    except ValidationError as exc:
        raise CorvusContractError(_sanitize_validation_error(exc)) from exc
    except TypeError as exc:
        raise CorvusContractError("invalid contract value: internal type error") from exc


def create_actor_presence(
    *,
    actor_id: str,
    ttl_seconds: int,
    clock: Callable[[Any], datetime],
    timezone: Any = None,
    **unexpected: Any,
) -> ActorPresence:
    """Server-side factory for an actor presence snapshot.

    Validates that ``actor_id`` is a non-blank string, ``ttl_seconds`` is an
    ``int`` in ``1..86400`` excluding ``bool``, and ``clock`` is a callable
    returning a timezone-aware ``datetime``. Resolves the optional timezone,
    stamps ``heartbeat_at`` from ``clock(tz)`` normalized to aware UTC, and
    sets ``expires_at`` to ``heartbeat_at + timedelta(seconds=ttl_seconds)``.
    Converts Pydantic ``ValidationError`` and raw ``TypeError`` into
    ``CorvusContractError`` with sanitized messages.
    """
    if unexpected:
        raise CorvusContractError(
            "unexpected keyword arguments: " + ", ".join(sorted(unexpected))
        )
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise CorvusContractError(
            f"actor_id must be a non-blank string, got {type(actor_id).__name__}"
        )
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise CorvusContractError(
            f"ttl_seconds must be an int, got {type(ttl_seconds).__name__}"
        )
    if ttl_seconds < 1 or ttl_seconds > 86400:
        raise CorvusContractError("ttl_seconds must be between 1 and 86400")
    if not callable(clock):
        raise CorvusContractError(
            f"clock must be callable, got {type(clock).__name__}"
        )

    resolved_tz = _resolve_timezone(timezone)
    clock_value = clock(resolved_tz)
    if not isinstance(clock_value, datetime):
        raise CorvusContractError(
            f"clock must return a datetime, got {type(clock_value).__name__}"
        )
    heartbeat_utc = _normalize_aware_utc(clock_value)
    expires_utc = heartbeat_utc + timedelta(seconds=ttl_seconds)

    try:
        return ActorPresence(
            actor_id=actor_id,
            heartbeat_at=heartbeat_utc,
            expires_at=expires_utc,
        )
    except ValidationError as exc:
        raise CorvusContractError(_sanitize_validation_error(exc)) from exc
    except TypeError as exc:
        raise CorvusContractError("invalid contract value: internal type error") from exc


def create_actor_subscriptions(
    *,
    actor_id: str,
    topics: Any = None,
    scopes: Any = None,
    clock: Callable[[Any], datetime],
    timezone: Any = None,
    **unexpected: Any,
) -> ActorSubscriptions:
    """Server-side factory for an actor subscription snapshot.

    Validates that ``actor_id`` is a non-blank string and that ``topics`` and
    ``scopes`` are iterables of non-blank strings (coerced to immutable tuples
    via the shared helper; a bare string is rejected). The caller is expected
    to have already normalized/deduped topic and scope values — this factory
    only validates entries. Stamps ``updated_at`` from ``clock(tz)`` normalized
    to aware UTC. Converts Pydantic ``ValidationError`` and raw ``TypeError``
    into ``CorvusContractError`` with sanitized messages.
    """
    if unexpected:
        raise CorvusContractError(
            "unexpected keyword arguments: " + ", ".join(sorted(unexpected))
        )
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise CorvusContractError(
            f"actor_id must be a non-blank string, got {type(actor_id).__name__}"
        )
    if not callable(clock):
        raise CorvusContractError(
            f"clock must be callable, got {type(clock).__name__}"
        )

    resolved_tz = _resolve_timezone(timezone)
    clock_value = clock(resolved_tz)
    if not isinstance(clock_value, datetime):
        raise CorvusContractError(
            f"clock must return a datetime, got {type(clock_value).__name__}"
        )
    updated_utc = _normalize_aware_utc(clock_value)

    topics_tuple = _coerce_string_tuple(topics, "topics")
    for topic in topics_tuple:
        if not topic.strip():
            raise CorvusContractError("topics entries must be non-blank")
    scopes_tuple = _coerce_string_tuple(scopes, "scopes")
    for scope in scopes_tuple:
        if not scope.strip():
            raise CorvusContractError("scopes entries must be non-blank")

    try:
        return ActorSubscriptions(
            actor_id=actor_id,
            topics=topics_tuple,
            scopes=scopes_tuple,
            updated_at=updated_utc,
        )
    except ValidationError as exc:
        raise CorvusContractError(_sanitize_validation_error(exc)) from exc
    except TypeError as exc:
        raise CorvusContractError("invalid contract value: internal type error") from exc
