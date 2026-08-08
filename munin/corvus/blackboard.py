# tags: [core, corvus, blackboard, publish, orchestration, fingerprint, atomic-index, deepagents-boundary, token-hygiene]
"""Corvus blackboard publish orchestration (GREEN for the committed RED contract).

The blackboard is an **external operative store** in the DeepAgents sense:
durable operational records live OUTSIDE the ephemeral LangGraph checkpoint.
DeepWiki (langchain-ai/deepagents) confirms the split between the in-graph
``StateBackend`` and the persistent ``StoreBackend``/``BaseStore``; Corvus
therefore owns no graph state, SQLite, ``SharedStateStore``, checkpoint, or
inbox. Everything here is expressed as Redis commands on the injected
transport.

Redis wire facts, verified via Context7 (Upstash Redis, ``/websites/upstash_redis``):

* ``SET key value NX`` claims exactly when a key is absent — used for the
  fingerprint reservation.
* ``XADD key * field value`` appends one stream entry per publish.
* ``ZADD key score member`` maintains a sorted-set index by epoch-millisecond
  score.
* ``/multi-exec`` (``pipeline(..., atomic=True)``) commits the whole batch as one
  transaction — all-or-nothing for the post record and every index.
* ``EVAL`` runs the Lua compare-and-delete so an owned reservation is released
  only when the stored value is still our own candidate id (never an unguarded
  ``DEL``).

The server remains the sole authority: caller-supplied fingerprints are
hashed into SHA-256 keys and the raw fingerprint never appears in a command,
error, or log surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from munin.corvus.contracts import (
    ActorIdentity,
    ActorPresence,
    ActorSubscriptions,
    CorvusContractError,
    Post,
    PostState,
    PostType,
    Reaction,
    coerce_post_state,
    create_actor_presence,
    create_actor_subscriptions,
    create_post,
    create_reaction,
    validate_scope,
)
from munin.corvus.transport import CorvusTransportError, RedisTransport

__all__ = [
    "CorvusError",
    "CorvusConflictError",
    "CorvusNotFoundError",
    "CorvusBlackboard",
]

# Lua compare-and-delete: touch a reserved key only when the stored value is
# still the exact candidate id this board wrote. The fingerprint key is never
# dropped carelessly, so a lost race does not destroy another writer's claim.
_EVAL_COMPARE_DELETE = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] "
    "then return redis.call('DEL', KEYS[1]) "
    "else return 0 end"
)


class CorvusError(Exception):
    """Base failure for the Corvus blackboard; never echoes credentials."""


class CorvusConflictError(CorvusError):
    """A fingerprint reservation or stored record is missing or corrupt."""


class CorvusNotFoundError(CorvusError):
    """A referenced Corvus record does not exist."""


def _slugify_topic(raw: str) -> str:
    """Safe slug for a stripped, lowercase topic: only ``[a-z0-9_-]`` survives.

    Every run of disallowed characters becomes a single ``-``, repeated dashes
    collapse, leading/trailing dashes are trimmed, and a topic that slugs to
    nothing is rejected by the caller as blank.
    """
    lowered = raw.strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", lowered)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _normalize_topics(topics: Iterable[str] | None) -> tuple[str, ...]:
    """Slug, dedupe (first-seen order), and reject blank/non-string topics.

    A bare ``str`` is treated as a caller error — topics must be an iterable of
    strings. Validation happens before any transport mutation so a bad topic
    can never touch Redis.
    """
    if topics is None:
        return ()
    if isinstance(topics, str):
        raise CorvusError("topics must be an iterable of strings")
    result: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, str):
            raise CorvusError("topics entries must be strings")
        slug = _slugify_topic(topic)
        if not slug:
            raise CorvusError("topics must not be blank")
        if slug not in seen:
            seen.add(slug)
            result.append(slug)
    return tuple(result)


def _normalize_scopes(scopes: Iterable[str] | None) -> tuple[str, ...]:
    """Validate, then dedupe (first-seen order) an iterable of scopes.

    ``None`` yields an empty tuple. A bare ``str`` or a non-iterable is a
    caller error, and non-string entries are rejected. Every member is
    validated through the shared ``validate_scope`` so approved scopes come out
    unchanged (``global`` or ``prefix:remainder``). Validation happens before
    any transport mutation so a bad scope can never touch Redis.
    """
    if scopes is None:
        return ()
    if isinstance(scopes, str):
        raise CorvusError("scopes must be an iterable of strings")
    try:
        items = tuple(scopes)
    except TypeError as exc:
        raise CorvusError("scopes must be an iterable of strings") from exc
    result: list[str] = []
    seen: set[str] = set()
    for scope in items:
        if not isinstance(scope, str):
            raise CorvusError("scopes entries must be strings")
        try:
            validated = validate_scope(scope)
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None
        if validated not in seen:
            seen.add(validated)
            result.append(validated)
    return tuple(result)


def _sanitized(exc: Exception) -> CorvusError:
    """Bounded, credential-free ``CorvusError`` from a transport failure."""
    message = str(exc).strip().replace("\n", " ")
    if not message:
        message = "corvus transport failure"
    if len(message) > 200:
        message = message[:200] + "...[truncated]"
    return CorvusError(message)


def _validate_query_limit(limit: Any) -> None:
    """Reject a non-``int``, ``bool``, or out-of-window query ``limit``.

    ``bool`` is an ``int`` subclass and is rejected explicitly so a truthy
    ``True``/``False`` limit is a caller error, not ``1``/``0``. The ``1..500``
    window runs *before* any transport call so a bad limit can never touch
    Redis.
    """
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise CorvusError("limit must be an int")
    if limit < 1 or limit > 500:
        raise CorvusError("limit must be between 1 and 500")


class CorvusBlackboard:
    """Atomic publish orchestration over a Corvus Redis transport.

    A single ``publish`` claims a fingerprint (when supplied), then commits one
    atomic pipeline that writes the compact post JSON, appends the stream, and
    updates every index. On a fingerprint collision the stored post is fetched
    and validated instead of re-published; on an owned-reservation failure the
    reservation is compare-deleted and a sanitized ``CorvusError`` is raised.
    """

    def __init__(
        self,
        *,
        transport: RedisTransport,
        clock: Callable[[Any], datetime],
        timezone: str | ZoneInfo = "UTC",
        key_prefix: str = "munin:bb",
    ) -> None:
        if not isinstance(transport, RedisTransport):
            raise CorvusError("transport must implement RedisTransport")
        if not callable(clock):
            raise CorvusError("clock must be callable")
        prefix = key_prefix.strip().rstrip(":")
        if not prefix:
            raise CorvusError("key_prefix must not be blank")
        self._transport = transport
        self._clock = clock
        self._timezone = timezone
        self._key_prefix = prefix

    @property
    def key_prefix(self) -> str:
        """The normalized key prefix, stripped of any trailing colons."""
        return self._key_prefix

    # ------------------------------------------------------------------
    # Public — the single atomic publish orchestration.
    # ------------------------------------------------------------------

    def publish(
        self,
        *,
        post_type: Any,
        scope: Any,
        actor_id: str,
        content: str,
        topics: Iterable[str] | None = None,
        confidence: float | None = None,
        evidence_refs: Iterable[str] | None = None,
        investigation_refs: Iterable[str] | None = None,
        observed_at: datetime | None = None,
        expires_at: datetime | None = None,
        fingerprint: str | None = None,
    ) -> Post:
        """Publish one post atomically; returns the stored ``Post``.

        ``topics`` are normalized and deduped before any transport. With a
        ``fingerprint`` the flow first attempts ``SET key NX``; a lost (non-own)
        claim loads the existing post; a won claim proceeds to the atomic write
        and, on failure, best-effort compare-deletes the reservation.
        """
        if fingerprint is not None and (
            not isinstance(fingerprint, str) or not fingerprint.strip()
        ):
            raise CorvusError("fingerprint must be a non-blank string")
        normalized_topics = _normalize_topics(topics)
        try:
            post = create_post(
                actor_id=actor_id,
                post_type=post_type,
                scope=scope,
                content=content,
                topics=normalized_topics,
                confidence=confidence,
                evidence_refs=evidence_refs,
                investigation_refs=investigation_refs,
                observed_at=observed_at,
                expires_at=expires_at,
                clock=self._clock,
                timezone=self._timezone,
            )
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None

        fp_key: str | None = None
        if fingerprint is not None:
            fp_key = self._fingerprint_key(fingerprint)
            try:
                owned = self._transport.command(
                    "SET", fp_key, post.id, "NX"
                )
            except CorvusTransportError as exc:
                raise _sanitized(exc) from None
            if not owned:
                return self._load_existing(fp_key)

        try:
            self._transport.pipeline(
                self._pipeline_commands(post), atomic=True
            )
        except CorvusTransportError as exc:
            if fp_key is not None:
                self._release_reservation(fp_key, post.id)
            raise _sanitized(exc) from None
        return post

    # ------------------------------------------------------------------
    # Public — ask a question.
    # ------------------------------------------------------------------

    def ask(
        self,
        *,
        actor_id: str,
        content: str,
        topics: Iterable[str] | None = None,
        scope: Any = "global",
        confidence: float | None = None,
        evidence_refs: Iterable[str] | None = None,
        investigation_refs: Iterable[str] | None = None,
        observed_at: datetime | None = None,
        expires_at: datetime | None = None,
        fingerprint: str | None = None,
    ) -> Post:
        """Publish an open question atomically; returns the stored ``Post``.

        Thin keyword wrapper over ``publish`` with ``PostType.QUESTION`` fixed
        and every other argument forwarded unchanged, so a ``status`` defaults
        to ``PostState.OPEN`` and the question lands in the ``questions:open``
        index via the same atomic pipeline as any other publish.
        """
        return self.publish(
            post_type=PostType.QUESTION,
            scope=scope,
            actor_id=actor_id,
            content=content,
            topics=topics,
            confidence=confidence,
            evidence_refs=evidence_refs,
            investigation_refs=investigation_refs,
            observed_at=observed_at,
            expires_at=expires_at,
            fingerprint=fingerprint,
        )

    # ------------------------------------------------------------------
    # Public — post retrieval.
    # ------------------------------------------------------------------

    def get_post(self, post_id: str) -> Post:
        """Fetch one post by exact key; never returns a payload or error echo.

        Missing or empty storage raises ``CorvusNotFoundError``. A corrupt
        payload (non-UTF-8 bytes, non-JSON, or invalid ``Post`` wire) raises a
        stable ``CorvusError`` from ``None`` so no raw transport value leaks
        as the exception cause. Transport failures collapse to a sanitized
        ``CorvusError``.
        """
        try:
            raw = self._transport.command("GET", self._post_key(post_id))
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        stored = self._decode_post_raw(raw)
        return self._parse_post_wire(stored)

    # ------------------------------------------------------------------
    # Public — actor registry.
    # ------------------------------------------------------------------

    def register_actor(self, *, identity: ActorIdentity) -> ActorIdentity:
        """Register an actor identity in one atomic pipeline.

        Rejects a non-``ActorIdentity`` or blank-id identity before any
        transport. Normalizes ``identity.capabilities`` with the shared
        ``_normalize_topics`` helper (slug, dedupe, first-seen order) so every
        capability index uses the canonical form. Writes exactly one atomic
        pipeline: ``SET prefix:actor:<id>`` with the compact canonical wire
        JSON, ``ZADD prefix:actors:created`` at the actor's ``created_at``
        epoch-milliseconds, and one ``ZADD prefix:index:capability:<normalized>``
        per capability at the same score/id in capability order. Transport
        failures collapse to a sanitized ``CorvusError``. Returns the original
        (unchanged) identity.
        """
        if not isinstance(identity, ActorIdentity):
            raise CorvusError(
                f"identity must be an ActorIdentity, got {type(identity).__name__}"
            )
        if not isinstance(identity.id, str) or not identity.id.strip():
            raise CorvusError("identity.id must be a non-blank string")

        capabilities = _normalize_topics(identity.capabilities)
        score = int(identity.created_at.timestamp() * 1000)
        commands: list[list[Any]] = [
            ["SET", self._actor_key(identity.id), self._actor_wire(identity)],
            ["ZADD", f"{self._key_prefix}:actors:created", score, identity.id],
        ]
        commands.extend(
            [
                [
                    "ZADD",
                    f"{self._key_prefix}:index:capability:{capability}",
                    score,
                    identity.id,
                ]
                for capability in capabilities
            ]
        )
        try:
            self._transport.pipeline(commands, atomic=True)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        return identity

    def get_actor(self, actor_id: str) -> ActorIdentity:
        """Fetch one actor identity by exact key; never a payload or error echo.

        Rejects a blank/whitespace/non-string ``actor_id`` before transport.
        Reads exactly ``GET prefix:actor:<id>``. Missing or empty storage
        raises ``CorvusNotFoundError``. A corrupt payload (non-UTF-8 bytes,
        non-JSON, or an invalid ``ActorIdentity`` wire) raises a stable
        ``CorvusError`` from ``None`` so no raw transport value, Redis key, or
        actor id leaks. Transport failures collapse to a sanitized
        ``CorvusError``.
        """
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise CorvusError(
                f"actor_id must be a non-blank string, got {type(actor_id).__name__}"
            )
        try:
            raw = self._transport.command("GET", self._actor_key(actor_id))
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        stored = self._decode_actor_raw(raw)
        return self._parse_actor_wire(stored)

    # ------------------------------------------------------------------
    # Public — actor presence heartbeat.
    # ------------------------------------------------------------------

    def heartbeat(
        self, *, actor_id: str, ttl_seconds: int = 60
    ) -> ActorPresence:
        """Record an actor presence heartbeat with a hard TTL.

        Builds the presence through ``create_actor_presence`` first — all
        validation (non-blank actor, intact ``ttl_seconds`` ``1..86400``,
        callable clock, server UTC timestamping) happens before any transport.
        Contract violations collapse to a stable ``CorvusError`` from ``None``.
        The actor is verified via ``get_actor`` (missing raises
        ``CorvusNotFoundError``), then exactly ``SET prefix:presence:<id>``
        with the compact canonical wire JSON and ``EX ttl_seconds``. Transport
        failures collapse to a sanitized ``CorvusError``. Returns the presence.
        """
        try:
            presence = create_actor_presence(
                actor_id=actor_id,
                ttl_seconds=ttl_seconds,
                clock=self._clock,
                timezone=self._timezone,
            )
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None

        actor_resolved = self.get_actor(actor_id)
        if actor_resolved.id != actor_id:
            raise CorvusNotFoundError("missing actor") from None

        payload = json.dumps(
            presence.to_wire(), sort_keys=True, separators=(",", ":")
        )
        try:
            self._transport.command(
                "SET",
                f"{self._key_prefix}:presence:{actor_id}",
                payload,
                "EX",
                ttl_seconds,
            )
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        return presence

    # ------------------------------------------------------------------
    # Public — actor subscriptions.
    # ------------------------------------------------------------------

    def subscribe(
        self,
        *,
        actor_id: str,
        topics: Iterable[str] | None = (),
        scopes: Iterable[str] | None = (),
    ) -> ActorSubscriptions:
        """Persist an actor's topic/scope subscriptions atomically.

        Normalizes ``topics`` with the shared ``_normalize_topics`` helper and
        validates/dedupes ``scopes`` through ``_normalize_scopes`` **before**
        any transport. The actor/timestamp snapshot is built through
        ``create_actor_subscriptions``, which returns the server-stamped
        ``ActorSubscriptions`` (server ``updated_at``). The actor is
        verified via ``get_actor`` (missing raises ``CorvusNotFoundError``).
        One atomic pipeline writes, in exact order: ``SET
        prefix:subscription:<id>`` with the compact canonical JSON; forward
        ``SADD prefix:subscription:actor:<id>:topics`` (when topics are
        nonempty); forward ``SADD ...:scopes`` (when scopes are nonempty); one
        reverse ``SADD prefix:subscribers:topic:<topic>`` per topic; and one
        reverse ``SADD prefix:subscribers:scope:<scope>`` per scope. Transport
        failures collapse to a sanitized ``CorvusError``.
        """
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise CorvusError(
                f"actor_id must be a non-blank string, got {type(actor_id).__name__}"
            )

        normalized_topics = _normalize_topics(topics)
        normalized_scopes = _normalize_scopes(scopes)

        try:
            actor = create_actor_subscriptions(
                actor_id=actor_id,
                topics=normalized_topics,
                scopes=normalized_scopes,
                clock=self._clock,
                timezone=self._timezone,
            )
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None

        actor_resolved = self.get_actor(actor_id)
        if actor_resolved.id != actor_id:
            raise CorvusNotFoundError("missing actor") from None

        payload = json.dumps(
            actor.to_wire(), sort_keys=True, separators=(",", ":")
        )
        commands: list[list[Any]] = [
            ["SET", f"{self._key_prefix}:subscription:{actor_id}", payload]
        ]
        if normalized_topics:
            commands.append(
                [
                    "SADD",
                    f"{self._key_prefix}:subscription:actor:{actor_id}:topics",
                    *normalized_topics,
                ]
            )
        if normalized_scopes:
            commands.append(
                [
                    "SADD",
                    f"{self._key_prefix}:subscription:actor:{actor_id}:scopes",
                    *normalized_scopes,
                ]
            )
        for topic in normalized_topics:
            commands.append(
                [
                    "SADD",
                    f"{self._key_prefix}:subscribers:topic:{topic}",
                    actor_id,
                ]
            )
        for scope in normalized_scopes:
            commands.append(
                [
                    "SADD",
                    f"{self._key_prefix}:subscribers:scope:{scope}",
                    actor_id,
                ]
            )
        try:
            self._transport.pipeline(commands, atomic=True)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        return actor

    # ------------------------------------------------------------------
    # Public — threaded reply.
    # ------------------------------------------------------------------

    def reply(
        self,
        *,
        actor_id: str,
        reply_to: str,
        content: str,
        topics: Iterable[str] | None = None,
        confidence: float | None = None,
        evidence_refs: Iterable[str] | None = None,
        investigation_refs: Iterable[str] | None = None,
    ) -> Post:
        """Reply to a post, inheriting the parent's scope and threading.

        Loads the parent (missing → ``CorvusNotFoundError``), derives the
        thread root as ``parent.thread_root_id or parent.id``, creates a
        ``PostType.REPLY`` post that inherits the parent's scope, normalizes
        topics, then commits one atomic pipeline. Contract violations from
        ``create_post`` collapse to ``CorvusError`` from ``None``; transport
        failures collapse to a sanitized ``CorvusError``.
        """
        parent = self.get_post(reply_to)
        thread_root = parent.thread_root_id or parent.id
        normalized_topics = _normalize_topics(topics)
        try:
            post = create_post(
                actor_id=actor_id,
                post_type=PostType.REPLY,
                scope=parent.scope,
                content=content,
                topics=normalized_topics,
                confidence=confidence,
                evidence_refs=evidence_refs,
                investigation_refs=investigation_refs,
                reply_to=reply_to,
                thread_root_id=thread_root,
                clock=self._clock,
                timezone=self._timezone,
            )
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None
        try:
            self._transport.pipeline(
                self._pipeline_commands(post), atomic=True
            )
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        return post

    # ------------------------------------------------------------------
    # Public — thread retrieval.
    # ------------------------------------------------------------------

    def get_thread(self, post_id: str, *, limit: int = 100) -> tuple[Post, ...]:
        """Resolve the thread for a post and return it in index order.

        Validates ``limit`` as a non-``bool`` ``int`` in ``1..500`` *before* any
        transport call. Loads the referenced post to derive its thread root,
        reads the thread sorted-set with ``ZRANGE root 0 limit-1``, decodes the
        member ids, then issues one non-atomic pipeline of ``GET`` commands —
        one per member — and parses the results to a ``tuple[Post, ...]`` in
        index order. A missing member raises ``CorvusNotFoundError``; a corrupt
        payload raises a stable ``CorvusError`` from ``None``. No payload or
        error echo ever surfaces.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise CorvusError("limit must be an int")
        if limit < 1 or limit > 500:
            raise CorvusError("limit must be between 1 and 500")
        referenced = self.get_post(post_id)
        root = referenced.thread_root_id or referenced.id
        thread_key = f"{self._key_prefix}:thread:{root}"
        try:
            members = self._transport.command("ZRANGE", thread_key, 0, limit - 1)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        member_ids = self._decode_thread_members(members)
        if not member_ids:
            raise CorvusError("missing thread index") from None
        get_commands = [
            ["GET", self._post_key(member_id)] for member_id in member_ids
        ]
        try:
            results = self._transport.pipeline(get_commands, atomic=False)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(results, list) or len(results) != len(member_ids):
            raise CorvusError("thread pipeline response cardinality mismatch")
        posts: list[Post] = []
        for raw in results:
            stored = self._decode_post_raw(raw)
            posts.append(self._parse_post_wire(stored))
        return tuple(posts)

    # ------------------------------------------------------------------
    # Public — feed and open-question queries.
    # ------------------------------------------------------------------

    def get_open_questions(self, *, limit: int = 50) -> tuple[Post, ...]:
        """Return open questions from the ``questions:open`` index newest-first.

        Validates ``limit`` as a non-``bool`` ``int`` in ``1..500`` *before* any
        transport call, then reads the ``questions:open`` sorted-set with
        ``ZREVRANGE 0 limit-1``. An empty index returns ``()``. A missing
        member raises ``CorvusNotFoundError``; a corrupt payload raises a
        stable ``CorvusError`` from ``None``. No payload or error echo ever
        surfaces.
        """
        _validate_query_limit(limit)
        return self._query_index(f"{self._key_prefix}:questions:open", limit)

    def feed(self, *, scope: Any = "global", limit: int = 50) -> tuple[Post, ...]:
        """Return the newest posts for a validated scope from its index.

        Validates ``limit`` as a non-``bool`` ``int`` in ``1..500`` *before* any
        transport call. ``validate_scope`` is applied to ``scope``; a
        ``CorvusContractError`` collapses to a stable ``CorvusError`` from
        ``None``. Reads the ``index:scope:{validated}`` sorted-set with
        ``ZREVRANGE 0 limit-1``; an empty index returns ``()``. A missing
        member raises ``CorvusNotFoundError``; a corrupt payload raises a
        stable ``CorvusError`` from ``None``. No payload or error echo ever
        surfaces.
        """
        _validate_query_limit(limit)
        try:
            validated = validate_scope(scope)
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None
        return self._query_index(
            f"{self._key_prefix}:index:scope:{validated}", limit
        )

    # ------------------------------------------------------------------
    # Public — capability-based open-question discovery.
    # ------------------------------------------------------------------

    def discover_open_questions(
        self, actor_id: str, *, limit: int = 50
    ) -> tuple[Post, ...]:
        """Return open questions matching any of an actor's capabilities.

        Validates ``limit`` (``_validate_query_limit``) and ``actor_id`` as a
        non-blank string *before* any transport call, then reads the durable
        actor record via ``get_actor`` (one exact ``GET``). The actor's
        ``capabilities`` are normalized/deduped with the shared
        ``_normalize_topics`` helper (slug + first-seen dedupe) — discovery fans
        out over the canonical topic slugs, never raw graph state.

        An actor with no normalized capabilities returns ``()`` with no
        candidate pipeline. Otherwise one non-atomic candidate pipeline issues
        ``ZREVRANGE {prefix}:questions:open 0 499`` followed by one
        ``ZREVRANGE {prefix}:index:topic:{capability} 0 499`` per capability,
        in normalized order. The response must be a list with exact command
        cardinality before indexing; each entry decodes via
        ``_decode_index_members``. The OR union of capability member sets is
        filtered through the ``questions:open`` base list, preserving its
        newest-first order and capped to ``limit``. An empty selection returns
        ``()`` with no ``GET`` pipeline.

        Otherwise one non-atomic ``GET`` pipeline resolves the selected post
        ids; the response must match exact cardinality. Member payloads decode
        through ``_decode_post_raw``/``_parse_post_wire`` — missing raises
        ``CorvusNotFoundError``, corrupt raises a stable ``CorvusError`` from
        ``None`` with no payload echo. Transport failures collapse to a
        sanitized ``CorvusError``. No mutation commands, temp keys,
        ``ZUNIONSTORE``, atomic pipelines, or direct candidate ``command``
        calls are used — the candidate fan-out and the resolution are both
        ordered non-atomic pipelines.
        """
        _validate_query_limit(limit)
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise CorvusError(
                f"actor_id must be a non-blank string, got {type(actor_id).__name__}"
            )

        actor = self.get_actor(actor_id)
        capabilities = _normalize_topics(actor.capabilities)
        if not capabilities:
            return ()

        prefix = self._key_prefix
        candidate_commands: list[list[Any]] = [
            ["ZREVRANGE", f"{prefix}:questions:open", 0, 499]
        ]
        for capability in capabilities:
            candidate_commands.append(
                ["ZREVRANGE", f"{prefix}:index:topic:{capability}", 0, 499]
            )

        try:
            candidate_results = self._transport.pipeline(
                candidate_commands, atomic=False
            )
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(candidate_results, list) or len(candidate_results) != len(
            candidate_commands
        ):
            raise CorvusError("candidate pipeline response cardinality mismatch")

        decoded_sets: list[tuple[str, ...]] = []
        for raw in candidate_results:
            decoded_sets.append(self._decode_index_members(raw))
        base_ids = decoded_sets[0]
        if not base_ids:
            return ()
        matched: set[str] = set()
        for extra in decoded_sets[1:]:
            matched.update(extra)
        selected_ids = [
            post_id for post_id in base_ids if post_id in matched
        ][:limit]
        if not selected_ids:
            return ()

        get_commands = [
            ["GET", self._post_key(post_id)] for post_id in selected_ids
        ]
        try:
            results = self._transport.pipeline(get_commands, atomic=False)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(results, list) or len(results) != len(selected_ids):
            raise CorvusError("discovery pipeline response cardinality mismatch")
        posts: list[Post] = []
        for raw in results:
            stored = self._decode_post_raw(raw)
            posts.append(self._parse_post_wire(stored))
        return tuple(posts)

    # ------------------------------------------------------------------
    # Public — subscriber discovery (non-atomic candidate + GET pipelines).
    # ------------------------------------------------------------------

    def discover_subscribers(
        self,
        *,
        topics: Iterable[str] | None = (),
        scopes: Iterable[str] | None = (),
        limit: int = 100,
    ) -> tuple[ActorIdentity, ...]:
        """Return subscribers to any of the given topics or scopes, newest-first.

        Validates ``limit`` (``_validate_query_limit``) as a non-``bool`` ``int``
        in ``1..500`` *before* any normalization or transport call. Normalizes
        and dedupes ``topics`` with the shared ``_normalize_topics`` helper
        (slug + first-seen dedupe) and ``scopes`` with ``_normalize_scopes``
        (``validate_scope`` per entry, first-seen dedupe); their contract
        violations collapse to a stable ``CorvusError`` from ``None``. If both
        normalized filters are empty the call returns ``()`` with no transport.

        Otherwise one non-atomic candidate pipeline issues, in exact order,
        ``ZREVRANGE {prefix}:actors:created 0 499`` (the newest-first base list),
        then one ``SMEMBERS {prefix}:subscribers:topic:{topic}`` per normalized
        topic, then one ``SMEMBERS {prefix}:subscribers:scope:{scope}`` per
        normalized scope. The response must be a ``list`` with exact command
        cardinality before any indexing; each entry decodes via
        ``_decode_index_members``. The OR union of every normalized subscription
        set is filtered through the ``actors:created`` base list preserving its
        newest-first order, explicitly deduped in first-seen order, and capped
        to ``limit``. An empty base or empty selection returns ``()`` with no
        ``GET`` pipeline.

        Otherwise one non-atomic ``GET`` pipeline resolves the selected ids via
        ``GET {prefix}:actor:{id}`` in selected order; the response must match
        exact cardinality. Each payload decodes through
        ``_decode_actor_raw``/``_parse_actor_wire`` — a missing actor raises
        ``CorvusNotFoundError``, a corrupt payload raises a stable
        ``CorvusError`` from ``None`` with no payload echo. Transport failures
        collapse to a sanitized ``CorvusError``. No mutation commands, temp
        keys, ``ZUNIONSTORE``, atomic pipelines, or direct candidate ``command``
        calls are used — the candidate fan-out and the resolution are both
        ordered non-atomic pipelines.
        """
        _validate_query_limit(limit)
        normalized_topics = _normalize_topics(topics)
        normalized_scopes = _normalize_scopes(scopes)
        if not normalized_topics and not normalized_scopes:
            return ()

        prefix = self._key_prefix
        candidate_commands: list[list[Any]] = [
            ["ZREVRANGE", f"{prefix}:actors:created", 0, 499]
        ]
        for topic in normalized_topics:
            candidate_commands.append(
                ["SMEMBERS", f"{prefix}:subscribers:topic:{topic}"]
            )
        for scope in normalized_scopes:
            candidate_commands.append(
                ["SMEMBERS", f"{prefix}:subscribers:scope:{scope}"]
            )

        try:
            candidate_results = self._transport.pipeline(
                candidate_commands, atomic=False
            )
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(candidate_results, list) or len(candidate_results) != len(
            candidate_commands
        ):
            raise CorvusError("candidate pipeline response cardinality mismatch")

        decoded_sets: list[tuple[str, ...]] = []
        for raw in candidate_results:
            decoded_sets.append(self._decode_index_members(raw))
        base_ids = decoded_sets[0]
        if not base_ids:
            return ()
        matched: set[str] = set()
        for extra in decoded_sets[1:]:
            matched.update(extra)

        deduped: list[str] = []
        seen: set[str] = set()
        for actor_id in base_ids:
            if actor_id in matched and actor_id not in seen:
                seen.add(actor_id)
                deduped.append(actor_id)
        selected_ids = deduped[:limit]
        if not selected_ids:
            return ()

        get_commands = [
            ["GET", self._actor_key(actor_id)] for actor_id in selected_ids
        ]
        try:
            results = self._transport.pipeline(get_commands, atomic=False)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(results, list) or len(results) != len(selected_ids):
            raise CorvusError("subscriber discovery pipeline response cardinality mismatch")
        actors: list[ActorIdentity] = []
        for raw in results:
            stored = self._decode_actor_raw(raw)
            actors.append(self._parse_actor_wire(stored))
        return tuple(actors)

    # ------------------------------------------------------------------
    # Public — multi-filter search (non-atomic candidate + GET pipelines).
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        topics: Iterable[str] | None = (),
        scope: Any = None,
        actor_id: Any = None,
        status: Any = None,
        limit: int = 50,
    ) -> tuple[Post, ...]:
        """Search the blackboard across filters, newest-first.

        Validates ``limit`` (``_validate_query_limit``), normalizes ``topics``
        (``_normalize_topics``), validates ``scope`` (``validate_scope``),
        requires ``actor_id`` to be a non-blank string, and coerces ``status``
        via ``coerce_post_state`` — all *before* any transport call; a
        ``CorvusContractError`` collapses to a stable ``CorvusError`` from
        ``None``.

        The first non-atomic candidate pipeline issues ``ZREVRANGE`` commands
        with range ``0 499`` over ``index:all`` followed by one per selected
        filter (normalized topics in source order, then optional scope, actor,
        and state). Candidate member lists are decoded via
        ``_decode_index_members`` and intersected in memory preserving the
        ``index:all`` newest-first order, capped to ``limit``. An empty result
        (either no ``index:all`` members or an empty intersection) returns
        ``()`` with no GET pipeline. Otherwise the second non-atomic pipeline
        issues one ``GET`` per matched id; results are parsed through
        ``_decode_post_raw``/``_parse_post_wire``. A missing member raises
        ``CorvusNotFoundError``; a corrupt payload or pipeline cardinality
        mismatch raises a stable ``CorvusError`` from ``None``. Transport
        failures collapse to a sanitized ``CorvusError``.

        No single ``command`` calls, temp keys, ``ZINTERSTORE``, writes, or
        atomic pipelines are used. The candidate fan-out and the GET
        resolution are both ordered non-atomic pipelines.
        """
        _validate_query_limit(limit)
        normalized_topics = _normalize_topics(topics)
        validated_scope: str | None = None
        if scope is not None:
            try:
                validated_scope = validate_scope(scope)
            except CorvusContractError as exc:
                raise CorvusError(str(exc)) from None
        validated_actor: str | None = None
        if actor_id is not None:
            if not isinstance(actor_id, str) or not actor_id.strip():
                raise CorvusError("actor_id must be a non-blank string")
            validated_actor = actor_id.strip()
        validated_state: PostState | None = None
        if status is not None:
            try:
                validated_state = coerce_post_state(status)
            except CorvusContractError as exc:
                raise CorvusError(str(exc)) from None

        prefix = self._key_prefix
        candidate_commands: list[list[Any]] = [
            ["ZREVRANGE", f"{prefix}:index:all", 0, 499]
        ]
        for topic in normalized_topics:
            candidate_commands.append(
                ["ZREVRANGE", f"{prefix}:index:topic:{topic}", 0, 499]
            )
        if validated_scope is not None:
            candidate_commands.append(
                ["ZREVRANGE", f"{prefix}:index:scope:{validated_scope}", 0, 499]
            )
        if validated_actor is not None:
            candidate_commands.append(
                ["ZREVRANGE", f"{prefix}:index:actor:{validated_actor}", 0, 499]
            )
        if validated_state is not None:
            candidate_commands.append(
                [
                    "ZREVRANGE",
                    f"{prefix}:index:state:{validated_state.value}",
                    0,
                    499,
                ]
            )

        try:
            candidate_results = self._transport.pipeline(
                candidate_commands, atomic=False
            )
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(candidate_results, list) or len(candidate_results) != len(
            candidate_commands
        ):
            raise CorvusError("candidate pipeline response cardinality mismatch")

        decoded_sets: list[tuple[str, ...]] = []
        for raw in candidate_results:
            decoded_sets.append(self._decode_index_members(raw))
        base_ids = decoded_sets[0]
        if not base_ids:
            return ()
        surviving = set(base_ids)
        for extra in decoded_sets[1:]:
            surviving &= set(extra)
        ordered_ids = [
            post_id for post_id in base_ids if post_id in surviving
        ]
        matched_ids = ordered_ids[:limit]
        if not matched_ids:
            return ()

        get_commands = [
            ["GET", self._post_key(post_id)] for post_id in matched_ids
        ]
        try:
            results = self._transport.pipeline(get_commands, atomic=False)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(results, list) or len(results) != len(matched_ids):
            raise CorvusError("search pipeline response cardinality mismatch")
        posts: list[Post] = []
        for raw in results:
            stored = self._decode_post_raw(raw)
            posts.append(self._parse_post_wire(stored))
        return tuple(posts)

    # ------------------------------------------------------------------
    # Public — reaction.
    # ------------------------------------------------------------------

    def react(
        self,
        *,
        post_id: str,
        reaction_type: Any,
        actor_id: str,
    ) -> Reaction:
        """React to a post atomically; returns the stored ``Reaction``.

        Converts ``CorvusContractError`` from ``create_reaction`` to
        ``CorvusError`` from ``None`` — so an invalid reaction type (e.g.
        ``LIKE``) is rejected *before* any transport mutation. The target post
        is verified via ``get_post`` (missing → ``CorvusNotFoundError``) before
        the atomic pipeline writes the compact reaction JSON, the per-post
        reactions index, and one global stream entry. Transport failures
        collapse to a sanitized ``CorvusError``. No other writes.
        """
        try:
            reaction = create_reaction(
                post_id=post_id,
                reaction_type=reaction_type,
                actor_id=actor_id,
                clock=self._clock,
            )
        except CorvusContractError as exc:
            raise CorvusError(str(exc)) from None
        self.get_post(post_id)
        prefix = self._key_prefix
        score = int(reaction.timestamp.timestamp() * 1000)
        commands: list[list[Any]] = [
            [
                "SET",
                f"{prefix}:reaction:{reaction.id}",
                self._reaction_wire(reaction),
            ],
            ["ZADD", f"{prefix}:reactions:{reaction.post_id}", score, reaction.id],
            [
                "XADD",
                f"{prefix}:stream",
                "*",
                "reaction_id",
                reaction.id,
                "post_id",
                reaction.post_id,
                "actor_id",
                reaction.actor_id,
            ],
        ]
        try:
            self._transport.pipeline(commands, atomic=True)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        return reaction

    # ------------------------------------------------------------------
    # Public — resolve.
    # ------------------------------------------------------------------

    def resolve(self, *, post_id: str, actor_id: str) -> Post:
        """Mark a post as resolved atomically; returns the updated ``Post``.

        If the post is already ``RESOLVED``, returns the same object with no
        pipeline. Otherwise builds a ``model_copy`` with ``status`` set to
        ``PostState.RESOLVED`` and ``revision`` incremented by one, preserving
        ``id``, ``published_at``, ``published_at_local``, ``timezone_name`` and
        ``content``. The atomic pipeline rewrites the post key with the
        canonical wire, removes the post from its old state index, adds it to
        the resolved state index using the *original* publication epoch-ms,
        and appends one global stream entry. No ``edited_at`` is fabricated.
        Transport failures collapse to a sanitized ``CorvusError``.
        """
        post = self.get_post(post_id)
        if post.status is PostState.RESOLVED:
            return post
        resolved = post.model_copy(
            update={
                "status": PostState.RESOLVED,
                "revision": post.revision + 1,
            }
        )
        prefix = self._key_prefix
        score = int(post.published_at.timestamp() * 1000)
        commands: list[list[Any]] = [
            [
                "SET",
                self._post_key(post.id),
                self._post_wire(resolved),
            ],
            ["ZREM", f"{prefix}:index:state:{post.status.value}", post.id],
            [
                "ZADD",
                f"{prefix}:index:state:resolved",
                score,
                post.id,
            ],
            [
                "XADD",
                f"{prefix}:stream",
                "*",
                "post_id",
                post.id,
                "event",
                "resolved",
                "actor_id",
                actor_id,
            ],
        ]
        if post.type is PostType.QUESTION:
            commands.append(
                ["ZREM", f"{prefix}:questions:{post.status.value}", post.id]
            )
            commands.append(
                ["ZADD", f"{prefix}:questions:resolved", score, post.id]
            )
        try:
            self._transport.pipeline(commands, atomic=True)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        return resolved

    # ------------------------------------------------------------------
    # Pipeline construction — one atomic multi for stream, indexes, post.
    # ------------------------------------------------------------------

    def _pipeline_commands(self, post: Post) -> list[list[Any]]:
        prefix = self._key_prefix
        score = int(post.published_at.timestamp() * 1000)
        commands: list[list[Any]] = [
            [
                "XADD",
                f"{prefix}:stream",
                "*",
                "post_id",
                post.id,
            ],
            ["ZADD", f"{prefix}:index:all", score, post.id],
            ["ZADD", f"{prefix}:index:state:{post.status.value}", score, post.id],
            ["ZADD", f"{prefix}:index:scope:{post.scope}", score, post.id],
            ["ZADD", f"{prefix}:index:actor:{post.actor_id}", score, post.id],
            [
                "ZADD",
                f"{prefix}:thread:{post.thread_root_id or post.id}",
                score,
                post.id,
            ],
        ]
        for topic in post.topics:
            commands.append(
                ["ZADD", f"{prefix}:index:topic:{topic}", score, post.id]
            )
        if post.type is PostType.QUESTION:
            commands.append(
                ["ZADD", f"{prefix}:questions:{post.status.value}", score, post.id]
            )
        commands.append(
            [
                "SET",
                f"{prefix}:post:{post.id}",
                self._post_wire(post),
            ]
        )
        return commands

    def _post_wire(self, post: Post) -> str:
        return json.dumps(
            post.to_wire(), sort_keys=True, separators=(",", ":")
        )

    def _reaction_wire(self, reaction: Reaction) -> str:
        return json.dumps(
            reaction.to_wire(), sort_keys=True, separators=(",", ":")
        )

    # ------------------------------------------------------------------
    # Fingerprint collision handling.
    # ------------------------------------------------------------------

    def _load_existing(self, fp_key: str) -> Post:
        try:
            existing_id = self._coerce_utf8(
                self._transport.command("GET", fp_key)
            )
            stored = (
                self._transport.command("GET", self._post_key(existing_id))
                if existing_id
                else None
            )
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not existing_id:
            raise CorvusConflictError(
                "fingerprint reserved but stored post is missing"
            ) from None
        if isinstance(stored, bytes):
            try:
                stored = stored.decode("utf-8")
            except UnicodeDecodeError:
                raise CorvusConflictError(
                    "fingerprint reserved but stored post is corrupt"
                ) from None
        if not stored:
            raise CorvusConflictError(
                "fingerprint reserved but stored post is missing"
            ) from None
        try:
            wire = json.loads(stored)
        except ValueError:
            raise CorvusConflictError(
                "fingerprint reserved but stored post is corrupt"
            ) from None
        try:
            return Post.model_validate(wire)
        except (ValueError, TypeError):
            raise CorvusConflictError(
                "fingerprint reserved but stored post is invalid"
            ) from None

    def _coerce_utf8(self, value: Any) -> str:
        """Normalize a transport-read id (possibly bytes) to a UTF-8 string."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                raise CorvusConflictError(
                    "fingerprint reserved but stored id is not valid UTF-8"
                ) from None
        return str(value)

    def _release_reservation(self, fp_key: str, candidate_id: str) -> None:
        try:
            self._transport.command(
                "EVAL", _EVAL_COMPARE_DELETE, 1, fp_key, candidate_id
            )
        except CorvusTransportError:
            # Best-effort only; a failed release must not mask the publish error.
            pass

    # ------------------------------------------------------------------
    # Key helpers.
    # ------------------------------------------------------------------

    def _fingerprint_key(self, fingerprint: str) -> str:
        """SHA-256 of the fingerprint only; the raw value never leaves caller."""
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}:fingerprint:{digest}"

    def _post_key(self, post_id: str) -> str:
        return f"{self._key_prefix}:post:{post_id}"

    def _actor_key(self, actor_id: str) -> str:
        return f"{self._key_prefix}:actor:{actor_id}"

    def _actor_wire(self, actor: ActorIdentity) -> str:
        """Compact canonical wire JSON for an actor identity."""
        return json.dumps(
            actor.to_wire(), sort_keys=True, separators=(",", ":")
        )

    # ------------------------------------------------------------------
    # Actor payload decode + parse helpers — no payload or error echo.
    # ------------------------------------------------------------------

    def _decode_actor_raw(self, raw: Any) -> str:
        """Decode a transport-read actor payload to a UTF-8 string.

        ``None`` or an empty/whitespace value is a missing actor and raises
        ``CorvusNotFoundError``. ``bytes`` are decoded as UTF-8. A non-UTF-8
        ``bytes`` payload raises a stable ``CorvusError`` from ``None``. All
        messages are stable generic text — no actor id or key is ever echoed.
        """
        if raw is None:
            raise CorvusNotFoundError("missing actor") from None
        if isinstance(raw, bytes):
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise CorvusError("corrupt actor payload") from None
            if not decoded.strip():
                raise CorvusNotFoundError("missing actor") from None
            return decoded
        if isinstance(raw, str):
            if not raw.strip():
                raise CorvusNotFoundError("missing actor") from None
            return raw
        raise CorvusError("corrupt actor payload") from None

    def _parse_actor_wire(self, stored: str) -> ActorIdentity:
        """Parse a stored JSON wire string into an ``ActorIdentity``.

        Non-JSON or invalid ``ActorIdentity`` wire raises a stable
        ``CorvusError`` from ``None`` — never a raw ``ValueError``/``TypeError``
        and never an echo of the payload itself, Redis key, or actor id.
        """
        try:
            wire = json.loads(stored)
        except ValueError:
            raise CorvusError("corrupt actor payload") from None
        try:
            return ActorIdentity.model_validate(wire)
        except (ValueError, TypeError):
            raise CorvusError("corrupt actor payload") from None

    # ------------------------------------------------------------------
    # Post/payload decode + parse helpers — no payload or error echo.
    # ------------------------------------------------------------------

    def _decode_post_raw(self, raw: Any) -> str:
        """Decode a transport-read post payload to a UTF-8 string.

        ``None`` or an empty/whitespace value is a missing post and raises
        ``CorvusNotFoundError``. ``bytes`` are decoded as UTF-8. A non-UTF-8
        ``bytes`` payload raises a stable ``CorvusError`` from ``None``. All
        messages are stable generic text — no Redis key or caller-supplied
        post id is ever echoed.
        """
        if raw is None:
            raise CorvusNotFoundError("missing post") from None
        if isinstance(raw, bytes):
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise CorvusError("corrupt post payload") from None
            if not decoded.strip():
                raise CorvusNotFoundError("missing post") from None
            return decoded
        if isinstance(raw, str):
            if not raw.strip():
                raise CorvusNotFoundError("missing post") from None
            return raw
        raise CorvusError("corrupt post payload") from None

    def _parse_post_wire(self, stored: str) -> Post:
        """Parse a stored JSON wire string into a ``Post``.

        Non-JSON or invalid ``Post`` wire raises a stable ``CorvusError`` from
        ``None`` — never a raw ``ValueError``/``TypeError`` and never an echo
        of the payload itself, Redis key, or caller-supplied post id.
        """
        try:
            wire = json.loads(stored)
        except ValueError:
            raise CorvusError("corrupt post payload") from None
        try:
            return Post.model_validate(wire)
        except (ValueError, TypeError):
            raise CorvusError("corrupt post payload") from None

    def _decode_thread_members(self, raw: Any) -> list[str]:
        """Decode a ``ZRANGE`` thread member list to ordered UTF-8 id strings.

        A blank member is corruption, not a silently skipped entry. Messages
        are stable generic text — no Redis key is ever echoed.
        """
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            members: list[str] = []
            for item in raw:
                if isinstance(item, bytes):
                    try:
                        decoded = item.decode("utf-8")
                    except UnicodeDecodeError:
                        raise CorvusError("corrupt thread index") from None
                elif isinstance(item, str):
                    decoded = item
                else:
                    raise CorvusError("corrupt thread index") from None
                if not decoded:
                    raise CorvusError("corrupt thread index") from None
                members.append(decoded)
            return members
        raise CorvusError("corrupt thread index") from None

    def _query_index(self, index_key: str, limit: int) -> tuple[Post, ...]:
        """Read an index sorted-set newest-first and resolve its member posts.

        Issues ``ZREVRANGE index_key 0 limit-1``, decodes the member ids, and
        returns ``()`` for an empty index. Otherwise issues one non-atomic
        pipeline of ``GET`` commands — one per member — and parses the results
        to a ``tuple[Post, ...]`` in index order. A missing member raises
        ``CorvusNotFoundError``; a corrupt payload raises a stable
        ``CorvusError`` from ``None``. No payload or error echo ever surfaces.
        """
        try:
            members = self._transport.command("ZREVRANGE", index_key, 0, limit - 1)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        member_ids = self._decode_index_members(members)
        if not member_ids:
            return ()
        get_commands = [
            ["GET", self._post_key(member_id)] for member_id in member_ids
        ]
        try:
            results = self._transport.pipeline(get_commands, atomic=False)
        except CorvusTransportError as exc:
            raise _sanitized(exc) from None
        if not isinstance(results, list) or len(results) != len(member_ids):
            raise CorvusError("index pipeline response cardinality mismatch")
        posts: list[Post] = []
        for raw in results:
            stored = self._decode_post_raw(raw)
            posts.append(self._parse_post_wire(stored))
        return tuple(posts)

    def _decode_index_members(self, raw: Any) -> tuple[str, ...]:
        """Decode a ``ZREVRANGE`` member list to ordered UTF-8 id strings.

        ``None`` decodes to ``()``. A blank, non-``str``/``bytes``, or non-UTF-8
        member is corruption, not a silently skipped entry. Messages are stable
        generic text — no Redis key is ever echoed.
        """
        if raw is None:
            return ()
        if isinstance(raw, (list, tuple)):
            members: list[str] = []
            for item in raw:
                if isinstance(item, bytes):
                    try:
                        decoded = item.decode("utf-8")
                    except UnicodeDecodeError:
                        raise CorvusError("corrupt index") from None
                elif isinstance(item, str):
                    decoded = item
                else:
                    raise CorvusError("corrupt index") from None
                if not decoded:
                    raise CorvusError("corrupt index") from None
                members.append(decoded)
            return tuple(members)
        raise CorvusError("corrupt index") from None