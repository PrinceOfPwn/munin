# tags: [tests, core, runtime, metis, config, red-contract, tdd]
"""RED contract tests for ``munin.core.metis`` — Munin v2 model routing.

RED phase: tests only, no production code. Local execution is forbidden by
the ticket; the target module is imported lazily inside helpers so a missing
implementation surfaces as a clear pytest assertion failure rather than a
collection error.

Decision-complete contract (from the second architecture review):

``configs/models.json`` is four separate catalogs:
    providers: {provider_id: {adapter, base_url, api_key_env}}
    models:    {model_name: {provider, model_id, capabilities}}
    budgets:   {budget_name: {max_model_calls, max_tool_calls}}
    routes:    {route_name: {model, budget, timeout_seconds, retry_attempts, fallbacks}}

Authority: provider -> model -> route. `provider_id` is the catalog key;
`adapter` is the provider implementation string. They are distinct.

Required routes (all eight): ``munin``, ``heimdall``, ``generated_specialist``,
``tool_selector``, ``valkyrja``, ``volundr``, ``yggdrasil_builder``,
``ariadne_reranker``.

Public error type: ``MetisConfigError`` — the exact one type any validation
violation raises. No adaptive discovery, no `ValueError` fallback.

``MetisConfig.load(path, environ)`` validates and returns a config.
``route(name)`` returns one immutable ``ResolvedModelRoute`` with exactly these
fields:
    route_name, provider_id, adapter, base_url, model_name, model_id,
    capabilities (immutable tuple), budget (immutable ``Mapping[str, int]``),
    timeout_seconds, retry_attempts, fallbacks (immutable tuple),
    api_key (the resolved internal API-key value for the runtime LLM client).

Immutability is deep: mutating capabilities, fallbacks, and budget must fail
or be impossible, not only top-level field assignment.

``public_snapshot()`` returns a JSON-compatible redacted dict retaining
``provider_id``, ``adapter``, route names and model names, and never exposing
``api_key`` or the resolved environment value.

Validation rejects: unknown provider references from models; unknown model,
budget, or fallback references from routes; missing required routes; missing
or blank secrets; raw secret literals anywhere; fallback cycles.

Fixture model ids are literals only — no particular real model id is required.
"""

from __future__ import annotations

import collections.abc
import json
from pathlib import Path
from typing import Any

import pytest


REQUIRED_ROUTES: tuple[str, ...] = (
    "munin",
    "heimdall",
    "generated_specialist",
    "tool_selector",
    "valkyrja",
    "volundr",
    "yggdrasil_builder",
    "ariadne_reranker",
)


# ---------------------------------------------------------------------------
# Approved payload builder — single source of truth for the happy path.
# ---------------------------------------------------------------------------


def _approved_payload() -> dict[str, Any]:
    providers: dict[str, Any] = {
        "provider-a": {
            "adapter": "openai_compatible",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "PROVIDER_A_KEY",
        },
        "provider-b": {
            "adapter": "openai_compatible",
            "base_url": "https://example.invalid/v2",
            "api_key_env": "PROVIDER_B_KEY",
        },
    }
    models: dict[str, Any] = {
        "primary-model": {
            "provider": "provider-a",
            "model_id": "vendor/model-primary",  # fixture literal only
            "capabilities": ["chat", "tools"],
        },
        "secondary-model": {
            "provider": "provider-b",
            "model_id": "vendor/model-secondary",  # fixture literal only
            "capabilities": ["chat", "tools"],
        },
    }
    budgets: dict[str, Any] = {
        "standard": {"max_model_calls": 24, "max_tool_calls": 64},
    }
    routes: dict[str, Any] = {}
    for idx, name in enumerate(REQUIRED_ROUTES):
        routes[name] = {
            "model": "primary-model" if idx % 2 == 0 else "secondary-model",
            "budget": "standard",
            "timeout_seconds": 120,
            "retry_attempts": 3,
            "fallbacks": [],
        }
    routes["munin"]["fallbacks"] = ["heimdall"]
    return {
        "schema_version": 1,
        "providers": providers,
        "models": models,
        "budgets": budgets,
        "routes": routes,
    }


def _payload_env() -> dict[str, str]:
    return {
        "PROVIDER_A_KEY": "secret-value-provider-a",
        "PROVIDER_B_KEY": "secret-value-provider-b",
    }


def _write_models_json(tmp_path: Path, payload: dict[str, Any]) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    path = configs / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Lazy import helpers — absence becomes a RED assertion failure.
# ---------------------------------------------------------------------------


def _metis_module():
    try:
        from munin.core import metis  # noqa: WPS433 — lazy import is required
    except ImportError as exc:  # pragma: no cover
        pytest.fail("RED contract expects munin.core.metis; ImportError: " + str(exc))
    return metis


def _require(symbol: str) -> Any:
    obj = getattr(_metis_module(), symbol, None)
    if obj is None:  # pragma: no cover
        pytest.fail(f"RED contract expects munin.core.metis.{symbol}")
    return obj


def _load(path: Path, environ: dict[str, str]):
    MetisConfig = _require("MetisConfig")
    load = getattr(MetisConfig, "load", None)
    if load is None:  # pragma: no cover
        pytest.fail("RED contract expects MetisConfig.load(path, environ)")
    return load(path=path, environ=environ)


def _route(config: Any, name: str):
    method = getattr(config, "route", None)
    if method is None:  # pragma: no cover
        pytest.fail("RED contract expects MetisConfig.route(name)")
    return method(name)


def _public_snapshot(config: Any):
    method = getattr(config, "public_snapshot", None)
    if method is None:  # pragma: no cover
        pytest.fail("RED contract expects MetisConfig.public_snapshot()")
    return method()


# ---------------------------------------------------------------------------
# 1. load + route(): approved happy path, exact resolved-route fields.
# ---------------------------------------------------------------------------


def test_load_accepts_the_approved_topology_and_resolves_every_required_route(
    tmp_path: Path,
) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    ResolvedModelRoute = _require("ResolvedModelRoute")

    for name in REQUIRED_ROUTES:
        route = _route(config, name)
        assert isinstance(route, ResolvedModelRoute), name
        # The exact public field set the review enumerates.
        assert route.route_name == name
        # provider_id (catalog key) and adapter (implementation string) are distinct.
        assert route.provider_id in ("provider-a", "provider-b"), name
        assert route.adapter == "openai_compatible", name
        assert route.base_url in (
            "https://example.invalid/v1",
            "https://example.invalid/v2",
        ), name
        # model_name (catalog key) and model_id (vendor-supplied) are distinct.
        assert route.model_name in ("primary-model", "secondary-model"), name
        assert route.model_id.startswith("vendor/model-"), name
        assert route.capabilities == ("chat", "tools"), name
        assert route.budget == {"max_model_calls": 24, "max_tool_calls": 64}, name
        assert route.timeout_seconds == 120, name
        assert route.retry_attempts == 3, name
        assert route.fallbacks == (("heimdall",) if name == "munin" else ()), name
        # api_key is the resolved value the runtime LLM client needs.
        if route.provider_id == "provider-a":
            assert route.api_key == "secret-value-provider-a", name
        else:
            assert route.api_key == "secret-value-provider-b", name


# ---------------------------------------------------------------------------
# 2. Deep immutability of the resolved route.
# ---------------------------------------------------------------------------


def test_resolved_route_top_level_assignment_is_rejected(tmp_path: Path) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    route = _route(config, "munin")
    with pytest.raises((TypeError, ValueError)):
        route.model_id = "tampered"  # type: ignore[misc]
    assert route.model_id == "vendor/model-primary"


def test_resolved_route_capabilities_is_an_immutable_tuple(tmp_path: Path) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    route = _route(config, "munin")
    assert isinstance(route.capabilities, tuple), "capabilities must be an immutable tuple"
    # Tuples reject item assignment with TypeError; there is no append method to
    # probe, so immutability is asserted through the sequence protocol itself.
    with pytest.raises(TypeError):
        route.capabilities[0] = "vision"  # type: ignore[index]
    # Reassignment is also rejected (frozen field).
    with pytest.raises((TypeError, ValueError)):
        route.capabilities = ("chat", "tools", "vision")  # type: ignore[misc]


def test_resolved_route_fallbacks_is_an_immutable_tuple(tmp_path: Path) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    route = _route(config, "munin")
    assert isinstance(route.fallbacks, tuple), "fallbacks must be an immutable tuple"
    # Same protocol-level immutability proof as capabilities: item assignment.
    with pytest.raises(TypeError):
        route.fallbacks[0] = "heimdall"  # type: ignore[index]
    with pytest.raises((TypeError, ValueError)):
        route.fallbacks = ("heimdall", "volundr")  # type: ignore[misc]


def test_resolved_route_budget_is_an_immutable_mapping(tmp_path: Path) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    route = _route(config, "munin")
    budget = route.budget
    # One decision-complete type: an immutable collections.abc.Mapping[str, int].
    # A frozen pydantic-style object is NOT acceptable — the contract requires
    # item assignment to raise TypeError, never ValueError, and a MappingProxyType
    # backed mapping is the type that satisfies that.
    assert isinstance(budget, collections.abc.Mapping), "budget must be a Mapping[str, int]"
    with pytest.raises(TypeError):
        budget["max_model_calls"] = 999  # type: ignore[index]
    assert budget["max_model_calls"] == 24
    assert budget["max_tool_calls"] == 64
    assert budget == {"max_model_calls": 24, "max_tool_calls": 64}


# ---------------------------------------------------------------------------
# 3. public_snapshot(): redacted, JSON-compatible, retains names + adapter.
# ---------------------------------------------------------------------------


def test_public_snapshot_is_json_compatible_and_retains_names_and_adapter(
    tmp_path: Path,
) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    snapshot = _public_snapshot(config)
    # Must round-trip through plain json.dumps without raising; no fallback
    # serializer may be needed to make the snapshot JSON-compatible.
    blob = json.dumps(snapshot)
    assert set(snapshot) == {
        "schema_version",
        "providers",
        "models",
        "budgets",
        "routes",
    }
    assert snapshot["schema_version"] == 1
    assert set(snapshot["providers"]) == {"provider-a", "provider-b"}
    assert set(snapshot["models"]) == {"primary-model", "secondary-model"}
    assert set(snapshot["budgets"]) == {"standard"}
    assert set(snapshot["routes"]) == set(REQUIRED_ROUTES)
    # provider_id and adapter survive redaction; the env value does not.
    provider_a = snapshot["providers"]["provider-a"]
    assert provider_a["adapter"] == "openai_compatible"
    assert provider_a["api_key_env"] == "PROVIDER_A_KEY"
    assert "api_key" not in provider_a
    # Every route retains its model name, budget name and fallback route names.
    munin_route = snapshot["routes"]["munin"]
    assert munin_route["model"] == "primary-model"
    assert munin_route["budget"] == "standard"
    assert munin_route["fallbacks"] == ["heimdall"]
    # Every model retains its provider id.
    assert snapshot["models"]["primary-model"]["provider"] == "provider-a"
    # Sanity: the blob below must not contain any resolved secret value.
    assert "secret-value-provider-a" not in blob
    assert "secret-value-provider-b" not in blob


def test_public_snapshot_never_exposes_the_resolved_api_key_value(tmp_path: Path) -> None:
    config = _load(_write_models_json(tmp_path, _approved_payload()), _payload_env())
    snapshot = _public_snapshot(config)
    blob = json.dumps(snapshot)
    for secret in _payload_env().values():
        assert secret not in blob
    # No provider object anywhere in the snapshot carries a resolved `api_key`
    # key (the env-var *name* `api_key_env` is fine and stays visible). Exact-key
    # membership, not substring matching: "api_key" is a substring of
    # "api_key_env", so a blob probe would always be poisoned by the allowed key.
    assert all("api_key" not in provider for provider in snapshot["providers"].values())


# ---------------------------------------------------------------------------
# 4. Validation — exactly one public error type: MetisConfigError.
# ---------------------------------------------------------------------------


def test_validation_rejects_an_unknown_provider_reference_from_a_model(
    tmp_path: Path,
) -> None:
    payload = _approved_payload()
    payload["models"]["primary-model"]["provider"] = "provider-unknown"
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_an_unknown_model_reference_from_a_route(
    tmp_path: Path,
) -> None:
    payload = _approved_payload()
    payload["routes"]["munin"]["model"] = "model-unknown"
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_an_unknown_budget_reference_from_a_route(
    tmp_path: Path,
) -> None:
    payload = _approved_payload()
    payload["routes"]["munin"]["budget"] = "budget-unknown"
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_an_unknown_fallback_reference_from_a_route(
    tmp_path: Path,
) -> None:
    payload = _approved_payload()
    payload["routes"]["munin"]["fallbacks"] = ["route-unknown"]
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


@pytest.mark.parametrize("missing", REQUIRED_ROUTES)
def test_validation_rejects_a_missing_required_route(
    tmp_path: Path, missing: str
) -> None:
    payload = _approved_payload()
    del payload["routes"][missing]
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_a_missing_environment_value(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["providers"]["provider-a"]["api_key_env"] = "PROVIDER_A_DEFINITELY_UNSET"
    env = _payload_env()
    env.pop("PROVIDER_A_DEFINITELY_UNSET", None)
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), env)


def test_validation_rejects_a_blank_environment_value(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["providers"]["provider-a"]["api_key_env"] = "PROVIDER_A_BLANK"
    env = _payload_env()
    env["PROVIDER_A_BLANK"] = "   "  # whitespace-only is blank
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), env)


def test_validation_rejects_a_raw_secret_literal_in_a_provider(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["providers"]["provider-a"]["api_key"] = "sk-raw-inlined-literal"
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_a_raw_secret_literal_hidden_in_a_model(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["models"]["primary-model"]["api_key"] = "sk-raw-in-model"
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_a_raw_secret_literal_hidden_in_a_route(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["routes"]["munin"]["api_key"] = "sk-raw-in-route"
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_a_two_hop_fallback_cycle(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["routes"]["munin"]["fallbacks"] = ["heimdall"]
    payload["routes"]["heimdall"]["fallbacks"] = ["munin"]
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_validation_rejects_a_self_fallback_cycle(tmp_path: Path) -> None:
    payload = _approved_payload()
    payload["routes"]["volundr"]["fallbacks"] = ["volundr"]
    with pytest.raises(_require("MetisConfigError")):
        _load(_write_models_json(tmp_path, payload), _payload_env())


def test_a_non_cyclic_fallback_chain_is_accepted(tmp_path: Path) -> None:
    """Guard against an over-strict validator that rejects all fallbacks."""
    payload = _approved_payload()
    payload["routes"]["munin"]["fallbacks"] = ["heimdall"]
    payload["routes"]["heimdall"]["fallbacks"] = []
    config = _load(_write_models_json(tmp_path, payload), _payload_env())
    assert _route(config, "munin").model_id == "vendor/model-primary"
