# tags: [tests, core, runtime, metis, agent-routes, red-contract, tdd]
"""Tests for Metis agent model-route contracts.

These tests pin the model-route contract for subagents and workflow nodes so
that legacy bare ``gpt-4o`` model identifiers are never silently accepted. They
rely on fakes only and never touch the network or the real LangGraph runtime.

Required private seams
----------------------

The implementation under ``munin/`` is expected to expose:

* ``SubagentSpec.model_route``  - default ``"generated_specialist"``
* ``SubagentSpec.model``        - serialized field; a bare ``"gpt-4o"`` must be
  rejected by validation (not merely ignored).
* ``Node.model_route``          - default ``"generated_specialist"``
* ``Node.model``                - accepting ``"gpt-4o"`` must raise.
* ``SubagentFactory.__init__(tools, model=None)`` plus
  ``SubagentFactory.route_resolver`` settable from a spec.
* ``workflow_factory._resolve_workflow_model(node, explicit_model=None,
  route_resolver=None)`` - the narrow seam used by the workflow helper.

If any of those seams are missing or behave differently, the implementation in
``munin/`` must be updated to match this contract - this test file is the
source of truth.
"""

from __future__ import annotations

import inspect
import pytest

from munin.core.metis.subagents import (
    Node,
    SubagentFactory,
    SubagentSpec,
)
from munin.core.metis.workflow_factory import _resolve_workflow_model


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChatModel:
    """Minimal stand-in for a chat model instance."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FakeChatModel(name={self.name!r})"


class FakeTool:
    """Minimal stand-in for a tool that a subagent may carry."""

    def __init__(self, name: str = "fake-tool") -> None:
        self.name = name


def _make_route_resolver(calls: list[tuple[str, str | None]]):
    """Return a route resolver that records its invocations.

    The resolver follows the expected callable contract
    ``(route: str, spec_id: str | None) -> str`` where the returned string is
    the resolved model identifier (or a provider-prefixed route). Recording the
    calls lets tests assert that the resolver was or was not consulted.
    """

    def _resolver(route: str, spec_id: str | None = None) -> str:
        calls.append((route, spec_id))
        return f"resolved:{route}"

    return _resolver


# ---------------------------------------------------------------------------
# Schema-source hygiene: no legacy gpt-4o leaks
# ---------------------------------------------------------------------------


_SCHEMA_SOURCE_PATHS = [
    inspect.getsourcefile(SubagentSpec),
    inspect.getsourcefile(Node),
    inspect.getsourcefile(SubagentFactory),
    inspect.getsourcefile(_resolve_workflow_model),
]


@pytest.mark.parametrize("source_path", _SCHEMA_SOURCE_PATHS)
def test_schema_sources_contain_no_gpt_4o(source_path: str | None) -> None:
    """Schema-defining source files must not hard-code ``gpt-4o``.

    The legacy bare ``gpt-4o`` identifier is forbidden because it bypasses the
    route contract. If it reappears in a schema source file that is a
    regression and must be fixed in the implementation, not silenced here.
    """

    assert source_path is not None, "could not resolve source path for schema module"
    with open(source_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert "gpt-4o" not in text, (
        f"legacy 'gpt-4o' present in {source_path}; the route contract forbids it"
    )


# ---------------------------------------------------------------------------
# SubagentSpec
# ---------------------------------------------------------------------------


def test_subagent_spec_model_route_default_is_generated_specialist() -> None:
    """A spec constructed without an explicit route must default to the
    generated-specialist route rather than to any legacy model string."""

    spec = SubagentSpec(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[],
    )
    assert spec.model_route == "generated_specialist"


def test_subagent_spec_model_route_is_serialized() -> None:
    """``model_route`` is part of the serialized representation so that
    downstream consumers (workflow factory, registry) can see it without
    needing the live Python object."""

    spec = SubagentSpec(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[],
        model_route="custom-route",
    )
    serialized = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec.dict())
    assert "model_route" in serialized
    assert serialized["model_route"] == "custom-route"


def test_subagent_spec_rejects_legacy_bare_model() -> None:
    """Constructing a spec with a bare ``model='gpt-4o'`` must fail.

    The route contract requires model identifiers to flow through
    ``model_route`` or an explicit provider-prefixed model; a bare legacy
    string is a programming error. Either pydantic ValidationError or
    ValueError is acceptable depending on whether the field is forbidden or
    validated post-construction.
    """

    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        SubagentSpec(
            id="recon",
            name="Recon",
            description="recon specialist",
            prompt="you are recon",
            tools=[],
            model="gpt-4o",
        )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def test_node_model_route_default_is_generated_specialist() -> None:
    node = Node(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[],
    )
    assert node.model_route == "generated_specialist"


def test_node_rejects_legacy_bare_model() -> None:
    """A ``Node`` must refuse a bare ``model='gpt-4o'`` for the same reason a
    spec does."""

    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        Node(
            id="recon",
            name="Recon",
            description="recon specialist",
            prompt="you are recon",
            tools=[],
            model="gpt-4o",
        )


# ---------------------------------------------------------------------------
# SubagentFactory
# ---------------------------------------------------------------------------


def test_subagent_factory_route_resolver_resolves_when_no_explicit_model() -> None:
    """When no explicit ``model`` is supplied to the factory, the
    ``route_resolver`` must be consulted to resolve the model route from the
    spec's ``model_route``.

    The factory stores ``route_resolver`` (settable from a spec) and uses it
    when ``model`` is ``None``. The resolver receives the spec's route and the
    spec id, and its return value is the resolved model identifier.
    """

    calls: list[tuple[str, str | None]] = []
    resolver = _make_route_resolver(calls)
    spec = SubagentSpec(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[FakeTool()],
    )
    spec.route_resolver = resolver  # type: ignore[attr-defined]

    factory = SubagentFactory(tools=spec.tools, model=None)
    factory.route_resolver = resolver  # type: ignore[attr-defined]
    resolved = factory._resolve_model(spec)

    assert calls, "route_resolver was not consulted when no explicit model was given"
    assert calls[0][0] == "generated_specialist"
    assert resolved == "resolved:generated_specialist"


def test_subagent_factory_explicit_model_wins_without_resolver_call() -> None:
    """When an explicit ``model`` is supplied to the factory, the
    ``route_resolver`` must NOT be called - the explicit model wins.

    This pins the precedence rule: explicit > resolver > fail-closed.
    """

    calls: list[tuple[str, str | None]] = []
    resolver = _make_route_resolver(calls)
    spec = SubagentSpec(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[FakeTool()],
    )
    spec.route_resolver = resolver  # type: ignore[attr-defined]

    explicit = FakeChatModel(name="explicit")
    factory = SubagentFactory(tools=spec.tools, model=explicit)
    factory.route_resolver = resolver  # type: ignore[attr-defined]
    resolved = factory._resolve_model(spec)

    assert resolved is explicit
    assert not calls, "route_resolver must not be called when explicit model is supplied"


def test_subagent_factory_missing_resolver_fails_closed_naming_route() -> None:
    """When no explicit model is given AND no resolver is configured, the
    factory must fail closed by raising an error that names the unresolved
    route so the operator can see what was missing.

    RuntimeError or ValueError are both acceptable; the key requirement is
    that the unresolved route identifier appears in the error text.
    """

    spec = SubagentSpec(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[FakeTool()],
    )

    factory = SubagentFactory(tools=spec.tools, model=None)
    factory.route_resolver = None  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError) as exc_info:
        factory._resolve_model(spec)

    message = str(exc_info.value)
    assert "generated_specialist" in message, (
        "fail-closed error must name the unresolved route, got: " + message
    )


# ---------------------------------------------------------------------------
# workflow_factory._resolve_workflow_model
# ---------------------------------------------------------------------------


def test_resolve_workflow_model_explicit_model_precedence() -> None:
    """The workflow seam must prefer an explicit model over route resolution.

    Passing ``explicit_model`` suppresses the route_resolver call entirely;
    the explicit value is returned unchanged.
    """

    calls: list[tuple[str, str | None]] = []
    resolver = _make_route_resolver(calls)

    node = Node(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[],
    )
    explicit = FakeChatModel(name="explicit-wf")

    resolved = _resolve_workflow_model(node, explicit_model=explicit, route_resolver=resolver)

    assert resolved is explicit
    assert not calls, "route_resolver must not be called when explicit_model is supplied"


def test_resolve_workflow_model_route_callback_resolution() -> None:
    """Without an explicit model, the seam must invoke route_resolver with the
    node's ``model_route`` and return its result."""

    calls: list[tuple[str, str | None]] = []
    resolver = _make_route_resolver(calls)

    node = Node(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[],
    )
    node.model_route = "custom-wf-route"  # type: ignore[attr-defined]

    resolved = _resolve_workflow_model(node, explicit_model=None, route_resolver=resolver)

    assert calls, "route_resolver was not consulted for a node without an explicit model"
    assert calls[0][0] == "custom-wf-route"
    assert resolved == "resolved:custom-wf-route"


def test_resolve_workflow_model_fail_closed_naming_route() -> None:
    """Without an explicit model and without a resolver, the seam must fail
    closed and name the unresolved route in the error message."""

    node = Node(
        id="recon",
        name="Recon",
        description="recon specialist",
        prompt="you are recon",
        tools=[],
    )
    node.model_route = "unsatisfied-route"  # type: ignore[attr-defined]

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        _resolve_workflow_model(node, explicit_model=None, route_resolver=None)

    message = str(exc_info.value)
    assert "unsatisfied-route" in message, (
        "fail-closed error must name the unresolved route, got: " + message
    )
