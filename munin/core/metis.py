# tags: [core, runtime, metis, config, model-routing, immutable, redaction, json-schema-validation, ResolvedModelRoute, MetisConfig, MetisConfigError, public_snapshot]
"""Metis — Munin v2 model routing configuration contract.

A deterministic, immutable, redacting loader for the four-catalog model
routing topology::

    providers: {provider_id: {adapter, base_url, api_key_env}}
    models:    {model_name:   {provider, model_id, capabilities}}
    budgets:   {budget_name: {max_model_calls, max_tool_calls}}
    routes:    {route_name:   {model, budget, timeout_seconds,
                                retry_attempts, fallbacks}}

Authority is three-layer and unidirectional: ``provider -> model -> route``.
``provider_id`` is the catalog key; ``adapter`` is the provider implementation
string — they are distinct. ``model_name`` is the catalog key; ``model_id``
is the vendor-supplied identifier — they are also distinct.

Design notes (from the v2 architecture review and mandatory research gate):

1. **No default model.** This module loads and validates an operator-supplied
   configuration; it never invents a Munin runtime model. A later inventory
   ticket decides the real runtime catalog. GLM and DeepSeek remain OpenCode
   worker routes only.
2. **Configuration is external to agent state.** DeepWiki on
   ``langchain-ai/deepagents`` (2026-08-08) confirms that ``create_deep_agent``
   receives its model via the ``model`` parameter at construction time; model
   routing and configuration must remain outside ``DeepAgentState``. Metis
   therefore yields ``ResolvedModelRoute`` objects that the runtime consumes
   *when it constructs* an agent, never persisted inside agent state.
3. **Deep immutability.** ``capabilities`` and ``fallbacks`` are tuples (item
   assignment raises ``TypeError``; reassignment is blocked by the frozen
   Pydantic model and raises ``ValidationError``). ``budget`` is a
   ``types.MappingProxyType`` — an immutable ``collections.abc.Mapping[str, int]``
   whose item assignment raises ``TypeError``, never ``ValueError``. A plain
   frozen Pydantic field holding a ``dict`` would *not* satisfy the contract
   because ``dict`` item assignment silently succeeds even on a frozen model
   (Context7 /pydantic/pydantic, 2026-08-08). Hence the explicit proxy — with
   the field's *base annotation* kept as ``Mapping[str, int]`` so JSON Schema
   generation never treats ``MappingProxyType`` as an arbitrary type.
4. **One public error type.** Every validation violation — malformed JSON,
   unknown reference, missing route, blank/missing secret, raw secret literal,
   fallback cycle, type error from Pydantic — surfaces as ``MetisConfigError``.
   No ``ValueError``/``TypeError`` ever leaks from this module's public API.
5. **No secret leakage.** Resolved API-key values are never logged, never
   serialized into ``public_snapshot()``, and never included in exception
   text. ``public_snapshot()`` retains ``api_key_env`` (the variable *name*)
   and never carries a key named ``api_key`` (exact-key membership, not
   substring — ``api_key_env`` is allowed and intentionally retained). The
   blob-probe test on the JSON-serialized snapshot additionally guarantees no
   resolved secret value appears anywhere in the output.

Only existing dependencies (Pydantic v2, stdlib) are used. No registry,
singleton, global mutable state, runtime watcher, or provider client belongs
here — those are concerns of later tickets.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

# The eight route names the runtime depends on. Pulled out so a later ticket
# can pull these from the same single source rather than re-encoding them.
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

# Keys that betray an inlined raw secret. They are rejected wherever they
# appear in any catalog object. ``api_key_env`` is the *only* sanctioned
# secret-bearing key and only on a provider entry. The set is intentionally
# narrow: matching names like ``api_endpoint`` would be a false positive.
_SECRET_LIKE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "api_token",
        "secret",
        "secrets",
        "password",
        "passwd",
        "token",
        "private_key",
        "access_token",
        "bearer_token",
    }
)


class MetisConfigError(RuntimeError):
    """The single public error type for every Metis validation violation.

    Raised by :meth:`MetisConfig.load`, :meth:`MetisConfig.route`, and
    :meth:`MetisConfig.public_snapshot` whenever configuration is malformed,
    references an unknown catalog entry, drops a required route, exposes or
    omits a secret in an unsafe way, or contains a fallback cycle. The error
    message references *names* (provider ids, route names, ``api_key_env``
    variable names) and never the resolved secret value.
    """


# ---------------------------------------------------------------------------
# Immutable resolved route. A frozen Pydantic v2 model with tuple capabilities
# and fallbacks, and a MappingProxyType budget — the only combination that
# satisfies the contract: reassignment must be blocked by the frozen model
# (raising a `ValidationError`); tuple item assignment raises TypeError; budget
# item assignment raises TypeError (a MappingProxyType signature), never
# ValueError.
#
# Critical Pydantic v2 behavior (Context7 /pydantic/pydantic, 2026-08-08): a
# `Mapping[str, int]`-typed field coerces its value to a plain `dict` — the
# input *type* is not preserved for generic collections. A wrap validator that
# returns a `MappingProxyType` after `handler(value)` runs preserves the proxy
# as the final value; a `@field_validator(mode="before")` would be silently
# unwrapped back into a mutable `dict`, which would let
# `budget["max_model_calls"] = 999` succeed and break the immutability
# contract. Therefore:
#
#   * the *base annotation* is `Mapping[str, int]` — schema-friendly, so JSON
#     Schema generation never treats `MappingProxyType` as an arbitrary type;
#   * the *runtime value* is a fresh `MappingProxyType` re-wrapped around the
#     validated dict, so item assignment on `budget` raises TypeError.
# ---------------------------------------------------------------------------


def _wrap_budget_as_proxy(
    value: Any,
    handler: ValidatorFunctionWrapHandler,
    _info: ValidationInfo,
) -> MappingProxyType[str, int]:
    """Wrap validator: validate as ``Mapping[str, int]`` then freeze as a proxy.

    ``handler(value)`` runs Pydantic's default ``Mapping[str, int]`` schema —
    it accepts any mapping, int-coerces values, rejects non-ints as a
    ``ValidationError``, and returns a plain ``dict``. We rebuild a fresh
    ``MappingProxyType`` around the validated dict so item assignment on
    ``budget`` raises ``TypeError`` (mapping-proxy semantics), exactly as the
    ResolvedModelRoute contract requires.
    """

    validated = handler(value)
    # `validated` is a plain dict[str, int] here. Rebuild an owned dict and
    # wrap it so the proxy's backing storage is private to this object.
    return MappingProxyType(dict(validated))


# The base annotation drives both validation (Pydantic runs the
# `Mapping[str, int]` schema via the wrap-handler) and JSON Schema generation
# (schema-friendly `Mapping`, never `MappingProxyType`). The wrap-handler's
# return value — a fresh `MappingProxyType` — is preserved as the runtime
# value, keeping the budget an immutable Mapping while the annotation remains
# schema-friendly.
_BudgetField = Annotated[Mapping[str, int], WrapValidator(_wrap_budget_as_proxy)]


class ResolvedModelRoute(BaseModel):
    """One fully-resolved, immutable model route.

    Field set is exact (the v2 architecture review enumerates it):
        route_name, provider_id, adapter, base_url, model_name, model_id,
        capabilities, budget, timeout_seconds, retry_attempts, fallbacks,
        api_key.

    ``api_key`` carries the *resolved* environment value the runtime LLM
    client needs. It is kept on this object (never on a public snapshot)
    precisely so the runtime can build its provider client at construction
    time, in line with the DeepWiki guidance that model configuration is
    supplied to ``create_deep_agent`` rather than stored in agent state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    route_name: str
    provider_id: str
    adapter: str
    base_url: str
    model_name: str
    model_id: str
    capabilities: tuple[str, ...]
    # Immutable collections.abc.Mapping[str, int] backed by MappingProxyType.
    # The _BudgetField annotation keeps the schema-friendly `Mapping[str, int]`
    # base while the wrap-handler preserves the proxy as the runtime value;
    # see the block comment above _wrap_budget_as_proxy for the rationale.
    budget: _BudgetField
    timeout_seconds: int = Field(ge=0)
    retry_attempts: int = Field(ge=0)
    fallbacks: tuple[str, ...]
    # Resolved secret — kept off public_snapshot by construction (see
    # MetisConfig.public_snapshot). It is the only secret-bearing field on
    # this object.
    api_key: str


# ---------------------------------------------------------------------------
# Public configuration object.
# ---------------------------------------------------------------------------


class MetisConfig:
    """Loaded and validated Metis configuration.

    Construction is private; callers use :meth:`load`. The object holds the
    resolved topology in plain dicts (the redacted snapshot of what was
    loaded) plus a private index of resolved routes. It exposes only
    :meth:`route` and :meth:`public_snapshot`; internals are not part of the
    public API.
    """

    __slots__ = (
        "_schema_version",
        "_providers",
        "_models",
        "_budgets",
        "_routes",
        "_resolved",
        "_environ_keys",
    )

    def __init__(
        self,
        *,
        schema_version: int,
        providers: dict[str, dict[str, Any]],
        models: dict[str, dict[str, Any]],
        budgets: dict[str, dict[str, int]],
        routes: dict[str, dict[str, Any]],
        resolved: dict[str, ResolvedModelRoute],
        environ_keys: frozenset[str],
    ) -> None:
        self._schema_version = schema_version
        self._providers = providers
        self._models = models
        self._budgets = budgets
        self._routes = routes
        self._resolved = resolved
        # Names of env variables that *were* requested by providers, so a
        # public_snapshot can echo them. Values are never retained here.
        self._environ_keys = environ_keys

    # -- public API --------------------------------------------------------

    def route(self, name: str) -> ResolvedModelRoute:
        """Return the immutable resolved route for ``name``.

        Raises ``MetisConfigError`` if ``name`` was not a configured route.
        """

        resolved = self._resolved.get(name)
        if resolved is None:
            raise MetisConfigError(f"unknown route: {name!r}")
        return resolved

    def public_snapshot(self) -> dict[str, Any]:
        """Return a JSON-compatible redacted view of the configuration.

        Mirrors the input topology: ``schema_version``, ``providers``,
        ``models``, ``budgets``, ``routes``. Every provider retains its
        ``adapter``, ``base_url`` and ``api_key_env`` (the variable *name*)
        but never a resolved ``api_key`` value. The returned structure
        round-trips through ``json.dumps`` without a custom serializer.
        """

        return {
            "schema_version": self._schema_version,
            "providers": {
                pid: {
                    "adapter": prov["adapter"],
                    "base_url": prov["base_url"],
                    "api_key_env": prov["api_key_env"],
                }
                for pid, prov in self._providers.items()
            },
            "models": {
                mname: {
                    "provider": model["provider"],
                    "model_id": model["model_id"],
                    "capabilities": list(model["capabilities"]),
                }
                for mname, model in self._models.items()
            },
            "budgets": {
                bname: {
                    "max_model_calls": budget["max_model_calls"],
                    "max_tool_calls": budget["max_tool_calls"],
                }
                for bname, budget in self._budgets.items()
            },
            "routes": {
                rname: {
                    "model": route["model"],
                    "budget": route["budget"],
                    "timeout_seconds": route["timeout_seconds"],
                    "retry_attempts": route["retry_attempts"],
                    "fallbacks": list(route["fallbacks"]),
                }
                for rname, route in self._routes.items()
            },
        }

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, *, path: Path, environ: Mapping[str, str]) -> "MetisConfig":
        """Load and validate ``configs/models.json`` at ``path``.

        ``path`` is the full path to the JSON file (the test helper writes
        ``tmp_path/configs/models.json`` and passes that path). ``environ``
        is the mapping the provider ``api_key_env`` variables are resolved
        from — typically ``os.environ`` in production, but a plain dict in
        tests.

        Returns a fully validated, immutable :class:`MetisConfig`. Raises
        :class:`MetisConfigError` for any malformed topology, unknown
        reference, missing required route, missing/blank secret, raw secret
        literal, or fallback cycle. The root object must contain exactly the
        five schema keys (``schema_version``, ``providers``, ``models``,
        ``budgets``, ``routes``) and ``schema_version`` must equal 1, matching
        ``configs/models.schema.json``.
        """

        raw = cls._read_json(path)
        return cls._build(raw, environ)

    # -- internals (private) ----------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise MetisConfigError(f"cannot read configuration file: {exc}") from exc
        if not text.strip():
            raise MetisConfigError("configuration file is empty")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MetisConfigError(
                f"configuration file is not valid JSON: {exc.msg} at line {exc.lineno}"
            ) from exc

    @classmethod
    def _build(
        cls,
        raw: Any,
        environ: Mapping[str, str],
    ) -> "MetisConfig":
        if not isinstance(raw, dict):
            raise MetisConfigError("configuration root must be a JSON object")

        # Reject any secret-bearing key anywhere in the raw tree first, so
        # inlined literals never make it past the door even before structural
        # validation. We scan only the four known catalogs; a stray secret in
        # a top-level key like ``"api_key"`` is also rejected.
        cls._reject_secret_keys(raw, "<root>")
        for catalog_name in ("providers", "models", "budgets", "routes"):
            catalog = raw.get(catalog_name)
            if catalog is None:
                continue
            if not isinstance(catalog, dict):
                raise MetisConfigError(
                    f"catalog {catalog_name!r} must be a JSON object"
                )
            for entry_name, entry in catalog.items():
                if not isinstance(entry, dict):
                    raise MetisConfigError(
                        f"{catalog_name}.{entry_name!r} must be a JSON object"
                    )
                cls._reject_secret_keys(entry, f"{catalog_name}.{entry_name}")

        # The root object is closed: only the five schema keys are allowed,
        # matching `configs/models.schema.json` (additionalProperties: false).
        allowed_root_keys = frozenset(
            {"schema_version", "providers", "models", "budgets", "routes"}
        )
        extra_root_keys = set(raw) - allowed_root_keys
        if extra_root_keys:
            raise MetisConfigError(
                f"configuration root has unexpected key(s): {sorted(extra_root_keys)!r}"
            )

        # `schema_version` is required and pinned to 1: an absent key is not
        # silently defaulted, and no future integer is accepted. This mirrors
        # the JSON Schema `const: 1` in `configs/models.schema.json`.
        if "schema_version" not in raw:
            raise MetisConfigError(
                "configuration root is missing required key 'schema_version'"
            )
        schema_version = raw["schema_version"]
        if not isinstance(schema_version, int) or isinstance(
            schema_version, bool
        ):
            raise MetisConfigError("schema_version must be an integer")
        if schema_version != 1:
            raise MetisConfigError(
                f"unsupported schema_version {schema_version!r}; expected 1"
            )

        providers_raw = raw.get("providers")
        models_raw = raw.get("models")
        budgets_raw = raw.get("budgets")
        routes_raw = raw.get("routes")

        if not isinstance(providers_raw, dict) or not providers_raw:
            raise MetisConfigError("catalog 'providers' is missing or empty")
        if not isinstance(models_raw, dict) or not models_raw:
            raise MetisConfigError("catalog 'models' is missing or empty")
        if not isinstance(budgets_raw, dict) or not budgets_raw:
            raise MetisConfigError("catalog 'budgets' is missing or empty")
        if not isinstance(routes_raw, dict) or not routes_raw:
            raise MetisConfigError("catalog 'routes' is missing or empty")

        # ---- providers ----
        providers: dict[str, dict[str, Any]] = {}
        for pid, prov in providers_raw.items():
            adapter = prov.get("adapter")
            base_url = prov.get("base_url")
            api_key_env = prov.get("api_key_env")
            if not isinstance(adapter, str) or not adapter.strip():
                raise MetisConfigError(
                    f"providers.{pid!r}.adapter must be a non-empty string"
                )
            if not isinstance(base_url, str) or not base_url.strip():
                raise MetisConfigError(
                    f"providers.{pid!r}.base_url must be a non-empty string"
                )
            if not isinstance(api_key_env, str) or not api_key_env.strip():
                raise MetisConfigError(
                    f"providers.{pid!r}.api_key_env must be a non-empty string"
                )
            # Any extra key here would be a secret-like literal already
            # rejected above; reject any other non-sanctioned key too so the
            # schema stays tight.
            allowed_provider_keys = {"adapter", "base_url", "api_key_env"}
            extra = set(prov) - allowed_provider_keys
            if extra:
                raise MetisConfigError(
                    f"providers.{pid!r} has unexpected keys: {sorted(extra)!r}"
                )
            providers[pid] = {
                "adapter": adapter,
                "base_url": base_url,
                "api_key_env": api_key_env,
            }

        # ---- models ----
        models: dict[str, dict[str, Any]] = {}
        for mname, model in models_raw.items():
            provider_ref = model.get("provider")
            model_id = model.get("model_id")
            capabilities = model.get("capabilities")
            if not isinstance(provider_ref, str) or not provider_ref.strip():
                raise MetisConfigError(
                    f"models.{mname!r}.provider must be a non-empty string"
                )
            if provider_ref not in providers:
                # Unknown provider reference from a model — report by name only.
                raise MetisConfigError(
                    f"models.{mname!r} references unknown provider {provider_ref!r}"
                )
            if not isinstance(model_id, str) or not model_id.strip():
                raise MetisConfigError(
                    f"models.{mname!r}.model_id must be a non-empty string"
                )
            if not isinstance(capabilities, list) or not capabilities:
                raise MetisConfigError(
                    f"models.{mname!r}.capabilities must be a non-empty list"
                )
            for cap in capabilities:
                if not isinstance(cap, str) or not cap.strip():
                    raise MetisConfigError(
                        f"models.{mname!r}.capabilities must be non-empty strings"
                    )
            allowed_model_keys = {"provider", "model_id", "capabilities"}
            extra = set(model) - allowed_model_keys
            if extra:
                raise MetisConfigError(
                    f"models.{mname!r} has unexpected keys: {sorted(extra)!r}"
                )
            models[mname] = {
                "provider": provider_ref,
                "model_id": model_id,
                "capabilities": list(capabilities),
            }

        # ---- budgets ----
        budgets: dict[str, dict[str, int]] = {}
        for bname, budget in budgets_raw.items():
            max_model_calls = budget.get("max_model_calls")
            max_tool_calls = budget.get("max_tool_calls")
            if not _is_int_like(max_model_calls):
                raise MetisConfigError(
                    f"budgets.{bname!r}.max_model_calls must be an integer"
                )
            if not _is_int_like(max_tool_calls):
                raise MetisConfigError(
                    f"budgets.{bname!r}.max_tool_calls must be an integer"
                )
            allowed_budget_keys = {"max_model_calls", "max_tool_calls"}
            extra = set(budget) - allowed_budget_keys
            if extra:
                raise MetisConfigError(
                    f"budgets.{bname!r} has unexpected keys: {sorted(extra)!r}"
                )
            budgets[bname] = {
                "max_model_calls": int(max_model_calls),
                "max_tool_calls": int(max_tool_calls),
            }

        # ---- routes ----
        routes: dict[str, dict[str, Any]] = {}
        for rname, route in routes_raw.items():
            model_ref = route.get("model")
            budget_ref = route.get("budget")
            timeout_seconds = route.get("timeout_seconds")
            retry_attempts = route.get("retry_attempts")
            fallbacks = route.get("fallbacks")
            if not isinstance(model_ref, str) or not model_ref.strip():
                raise MetisConfigError(
                    f"routes.{rname!r}.model must be a non-empty string"
                )
            if model_ref not in models:
                raise MetisConfigError(
                    f"routes.{rname!r} references unknown model {model_ref!r}"
                )
            if not isinstance(budget_ref, str) or not budget_ref.strip():
                raise MetisConfigError(
                    f"routes.{rname!r}.budget must be a non-empty string"
                )
            if budget_ref not in budgets:
                raise MetisConfigError(
                    f"routes.{rname!r} references unknown budget {budget_ref!r}"
                )
            if not _is_int_like(timeout_seconds) or int(timeout_seconds) < 0:
                raise MetisConfigError(
                    f"routes.{rname!r}.timeout_seconds must be a non-negative integer"
                )
            if not _is_int_like(retry_attempts) or int(retry_attempts) < 0:
                raise MetisConfigError(
                    f"routes.{rname!r}.retry_attempts must be a non-negative integer"
                )
            if fallbacks is None:
                fallbacks = []
            if not isinstance(fallbacks, list):
                raise MetisConfigError(
                    f"routes.{rname!r}.fallbacks must be a list"
                )
            for fb in fallbacks:
                if not isinstance(fb, str) or not fb.strip():
                    raise MetisConfigError(
                        f"routes.{rname!r}.fallbacks must be non-empty strings"
                    )
            allowed_route_keys = {
                "model",
                "budget",
                "timeout_seconds",
                "retry_attempts",
                "fallbacks",
            }
            extra = set(route) - allowed_route_keys
            if extra:
                raise MetisConfigError(
                    f"routes.{rname!r} has unexpected keys: {sorted(extra)!r}"
                )
            routes[rname] = {
                "model": model_ref,
                "budget": budget_ref,
                "timeout_seconds": int(timeout_seconds),
                "retry_attempts": int(retry_attempts),
                "fallbacks": list(fallbacks),
            }

        # ---- required routes ----
        missing = [name for name in REQUIRED_ROUTES if name not in routes]
        if missing:
            raise MetisConfigError(
                f"missing required route(s): {missing!r}"
            )

        # ---- fallback references + cycle detection ----
        # Unknown fallback references are checked first; cycle detection
        # runs after every referenced fallback is known to exist.
        for rname, route in routes.items():
            for fb in route["fallbacks"]:
                if fb not in routes:
                    raise MetisConfigError(
                        f"routes.{rname!r} references unknown fallback route {fb!r}"
                    )
        cls._assert_no_fallback_cycles(routes)

        # ---- resolve API keys from the environment ----
        # Names only are retained in the snapshot. Resolved values are placed
        # only on the per-route ResolvedModelRoute (private to this object).
        resolved_api_keys: dict[str, str] = {}
        environ_keys: set[str] = set()
        for pid, prov in providers.items():
            env_name = prov["api_key_env"]
            environ_keys.add(env_name)
            if env_name not in environ:
                raise MetisConfigError(
                    f"providers.{pid!r} requires environment variable "
                    f"{env_name!r} which is not set"
                )
            value = environ[env_name]
            if not isinstance(value, str) or not value.strip():
                raise MetisConfigError(
                    f"providers.{pid!r} environment variable {env_name!r} is blank"
                )
            resolved_api_keys[pid] = value

        # ---- resolve every route into an immutable ResolvedModelRoute ----
        resolved: dict[str, ResolvedModelRoute] = {}
        try:
            for rname, route in routes.items():
                model_name = route["model"]
                model = models[model_name]
                provider_id = model["provider"]
                provider = providers[provider_id]
                budget_name = route["budget"]
                budget = budgets[budget_name]
                resolved_route = ResolvedModelRoute(
                    route_name=rname,
                    provider_id=provider_id,
                    adapter=provider["adapter"],
                    base_url=provider["base_url"],
                    model_name=model_name,
                    model_id=model["model_id"],
                    capabilities=tuple(model["capabilities"]),
                    budget=budget,
                    timeout_seconds=route["timeout_seconds"],
                    retry_attempts=route["retry_attempts"],
                    fallbacks=tuple(route["fallbacks"]),
                    api_key=resolved_api_keys[provider_id],
                )
                resolved[rname] = resolved_route
        except (ValidationError, ValueError, TypeError):
            # Final guard: any Pydantic or coercion failure becomes a single
            # MetisConfigError with a sanitized message that references only
            # the route name. `from None` suppresses the original exception,
            # so neither `__cause__` nor `__context__` can ever expose the
            # resolved api_key value (or any other field value) through the
            # traceback.
            raise MetisConfigError(
                f"failed to resolve route {rname!r}: configuration is invalid"
            ) from None

        return cls(
            schema_version=schema_version,
            providers=providers,
            models=models,
            budgets=budgets,
            routes=routes,
            resolved=resolved,
            environ_keys=frozenset(environ_keys),
        )

    @staticmethod
    def _reject_secret_keys(obj: Mapping[str, Any], location: str) -> None:
        """Reject any secret-bearing key in ``obj``.

        The only sanctioned secret surface in the entire topology is
        ``providers.<id>.api_key_env`` (the variable *name*); an inlined
        ``api_key`` literal on a provider, model, or route — or any sibling
        key naming a secret — is rejected as a raw secret leak. Names only
        are referenced in the error so a leaked literal is never echoed back.
        """

        for key in obj:
            if key in _SECRET_LIKE_KEYS:
                raise MetisConfigError(
                    f"{location} contains secret-like key {key!r}; "
                    f"use 'api_key_env' on a provider instead"
                )

    @staticmethod
    def _assert_no_fallback_cycles(routes: Mapping[str, Mapping[str, Any]]) -> None:
        """Reject any fallback cycle, including self-loops and multi-hop.

        A cycle exists when, starting from route ``r``, the transitive
        closure of ``fallbacks`` edges reaches ``r`` itself. Self-loops
        (``r -> r``) and two-hop cycles (``r -> s -> r``) are both rejected.
        A route may appear in some other route's fallback chain as long as
        no path from a route returns to itself — e.g. ``munin -> heimdall``
        with ``heimdall -> []`` is accepted (the test guards this).
        """

        # Reachability per route — DFS over fallback edges. If at any point
        # we re-enter the start node, it is a cycle.
        for start in routes:
            stack: list[str] = list(routes[start]["fallbacks"])
            seen: set[str] = set()
            while stack:
                current = stack.pop()
                if current == start:
                    raise MetisConfigError(
                        f"fallback cycle detected involving route {start!r}"
                    )
                if current in seen:
                    # Already explored this node from this start; skip.
                    continue
                seen.add(current)
                stack.extend(routes[current]["fallbacks"])


def load_metis_if_enabled(
    *, path: Path | None, environ: Mapping[str, str]
) -> MetisConfig | None:
    """Opt-in Metis loader for runtime composition roots.

    ``path`` is typically ``settings.metis_config_path`` (already ``None`` when
    ``MUNIN_MODELS_JSON`` is blank/unset). A ``None`` path returns ``None``
    without touching the environment or filesystem — Metis stays off and the
    legacy env-driven model construction is unchanged. A non-``None`` path
    delegates exactly to :meth:`MetisConfig.load`; a missing, malformed or
    secret-requiring configuration still fails fast with
    :class:`MetisConfigError`.

    Deliberately stateless: no global cache, no singleton, no default model,
    no env reads and no runtime wiring live here. The decision to load is the
    caller's (composition root), driven purely by ``settings.metis_config_path``.
    """

    if path is None:
        return None
    return MetisConfig.load(path=path, environ=environ)


def _is_int_like(value: Any) -> bool:
    """True if ``value`` is an int (and not a bool), or an int-coercible str.

    JSON parses bare integers as ``int`` already; the str branch is a
    courtesy for hand-written configs and never triggers on the test
    fixtures. ``bool`` is excluded because ``True``/``False`` are ``int``
    subclasses in Python and would silently pass numeric checks.
    """

    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str) and value.strip().lstrip("-+").isdigit():
        return True
    return False
