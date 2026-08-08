# tags: [core, corvus, contracts, blackboard, identity, posts, reactions, domain-events]
"""Package marker for ``munin.corvus`` — Corvus domain contracts.

The pure domain-contract slice lives in :mod:`munin.corvus.contracts`.
Importing this package does not eagerly load that module; tests import it
lazily so a missing implementation surfaces as a clear assertion failure rather
than a collection error (see ``tests/test_corvus_contracts.py``).

The contract layer hardens the server boundary (GREEN review round):

* Every factory boundary converts Pydantic ``ValidationError``,
  ``ZoneInfoNotFoundError``, and raw ``TypeError`` from invalid caller input
  into the single public ``CorvusContractError`` with sanitized messages.
* Server-owned timestamps (``published_at``, reaction ``timestamp``) use a
  private sentinel default, so *any* explicit caller supply — even ``None`` —
  is rejected as a forgery attempt rather than accepted-and-ignored.
* Every frozen Pydantic model closes its config with ``extra="forbid"``, and
  collection fields are immutable tuples projected to JSON lists at the
  ``to_wire()`` boundary.

Architectural note: ``deepagents`` models ``AgentState`` as a ``TypedDict``
whose only durable channel is a ``DeltaChannel`` of messages, subagent
identity lives on the supervisor's ``SubAgentMiddleware`` (not on the
subagent's graph state), and transient domain events are excluded from the
LangGraph checkpoint via ``_EXCLUDED_STATE_KEYS``.  Posts, reactions, and
actor identities are therefore domain records — not graph state — so
checkpoint growth stays bounded and transport replay can restore them
independently of the executor.
"""
