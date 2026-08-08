# tags: [tests, core, runtime, metis, config, red-contract, tdd, model-routing, llmclient, settings-opt-in]
"""RED contract for the first Metis runtime integration slice.

Scope of this slice (nothing beyond it):
  1. Settings opt-in surface: ``get_settings().metis_config_path`` driven by
     ``MUNIN_MODELS_JSON`` (absent -> None; present -> absolute resolved Path).
  2. ``munin.core.metis.load_metis_if_enabled(*, path, environ)`` — the opt-in
     loader. ``path=None`` -> None; a configured-but-missing path ->
     ``MetisConfigError``; a valid ``models.json`` + matching env ->
     ``MetisConfig`` resolving ``route("munin")``.
  3. ``LLMClient(settings, route=ResolvedModelRoute)`` — the route overrides
     empty/legacy ``settings`` (base_url, api_key, model_id, timeout_seconds,
     retry_attempts, budget max_model_calls/max_tool_calls) BEFORE the existing
     constructor validation and OpenAI construction.
  4. ``make_langchain`` adapter routing: ``adapter == "openai_compatible"`` ->
     ChatOpenAI built from the route; an unsupported adapter ->
     ``LLMConfigError`` with NO ChatOpenAI construction; a route-less legacy
     client keeps today's DeepSeek-substring / generic ChatOpenAI behaviour.

Explicitly NOT in this slice: committing ``configs/models.json``, selecting any
default model (GLM / DeepSeek remain worker routes only, never Munin defaults),
wiring Metis into production composition roots (``production/chat.py:906``,
``mcp/tools/munin_tools.py:555``, ``discord_adapter.py:1723``, ``cli.py:50``,
``core/supervisor.py:473``), BYOK-provider-profile precedence, per-kernel routes
or fallback chains.

Evidence (session research gate):
* DeepWiki langchain-ai/deepagents: ``create_deep_agent`` receives the model via
  the top-level ``model`` parameter (string or BaseChatModel); resolution happens
  before middleware/subagent construction, and model configuration stays OUTSIDE
  agent graph state / checkpoints — only model config metadata is persisted.
* Context7 official LangChain Python docs: ``init_chat_model`` takes
  ``"{provider}:{model}"`` with per-provider env-var credentials; configurable
  models bound by ``configurable_fields`` never leak provider selection into the
  serialized graph.

Audit boundary (live code, this session):
* munin/core/llm_client.py:62-70  LLMClient validation + OpenAI construction
* munin/core/llm_client.py:278-293 make_langchain (DeepSeek substring sniff :283)
* munin/mcp/config.py:24-26,209   llm_* settings fields + get_settings()
* munin/core/metis.py:322,261     MetisConfig.load / route()

RED target (intended first-action failures, each narrowly attributable):
* Settings.metis_config_path missing  -> AttributeError
* metis.load_metis_if_enabled missing -> AttributeError (asserted via fail)
* LLMClient(route=...) unsupported    -> TypeError (tests 3 and 4)

These tests are self-contained (no shared helpers, no cross-module order
coupling), lazy-import the metis / llm_client / config modules, and never touch
the network or a real provider key. The real ``langchain_openai`` import inside
``make_langchain`` is neutralised with a fake module in ``sys.modules`` so the
adapter contract can be asserted without importing the integration package.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

_REQUIRED_ROUTES: tuple[str, ...] = (
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
# Self-contained helpers (pattern mirrors tests/test_metis_config.py).
# ---------------------------------------------------------------------------


def _approved_payload() -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for name in _REQUIRED_ROUTES:
        routes[name] = {
            "model": "primary-model",
            "budget": "standard",
            "timeout_seconds": 120,
            "retry_attempts": 3,
            "fallbacks": [],
        }
    routes["munin"]["fallbacks"] = ["heimdall"]
    return {
        "schema_version": 1,
        "providers": {
            "provider-a": {
                "adapter": "openai_compatible",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "PROVIDER_A_KEY",
            },
        },
        "models": {
            "primary-model": {
                "provider": "provider-a",
                "model_id": "vendor/model-primary",
                "capabilities": ["chat", "tools"],
            },
        },
        "budgets": {
            "standard": {"max_model_calls": 24, "max_tool_calls": 64},
        },
        "routes": routes,
    }


def _payload_env() -> dict[str, str]:
    return {"PROVIDER_A_KEY": "secret-value-provider-a"}


def _write_models_json(tmp_path: Path, payload: dict[str, Any]) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    path = configs / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metis_module() -> Any:
    try:
        from munin.core import metis  # noqa: PLC0415 - lazy import is required
    except ImportError as exc:  # pragma: no cover - collection must not fail
        pytest.fail("RED contract expects munin.core.metis; ImportError: " + str(exc))
    return metis


def _blank_settings(tmp_path: Path) -> Any:
    """Settings with all LLM fields empty, independent of the ambient env."""
    from munin.mcp.config import Settings  # noqa: PLC0415

    return Settings(
        workspace_root=tmp_path,
        default_timeout=300,
        max_output_chars=32000,
        expected_egress_ip="",
        forbidden_egress_ip="",
        route_probe_ip="1.1.1.1",
        job_workers=5,
        github_token="",
        nvd_api_key="",
    )


def _legacy_settings(tmp_path: Path, *, model: str = "test-model") -> Any:
    return replace(
        _blank_settings(tmp_path),
        llm_base_url="https://llm.example.invalid/v1",
        llm_api_key="legacy-key",
        llm_model=model,
    )


def _route_fixture(*, adapter: str = "openai_compatible") -> Any:
    """A real, immutable ResolvedModelRoute — metis already ships the model."""
    from munin.core.metis import ResolvedModelRoute  # noqa: PLC0415

    return ResolvedModelRoute(
        route_name="munin",
        provider_id="provider-a",
        adapter=adapter,
        base_url="https://route.example.invalid/v1",
        model_name="primary-model",
        model_id="vendor/model-primary",
        capabilities=("chat", "tools"),
        budget={"max_model_calls": 24, "max_tool_calls": 64},
        timeout_seconds=90,
        retry_attempts=2,
        fallbacks=("heimdall",),
        api_key="sk-route-secret",
    )


# ---------------------------------------------------------------------------
# 1. Settings opt-in surface (MUNIN_MODELS_JSON).
# ---------------------------------------------------------------------------


def test_settings_metis_config_path_is_opt_in_via_munin_models_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from munin.mcp.config import get_settings  # noqa: PLC0415

    monkeypatch.setenv("OFFX_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("MUNIN_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("MUNIN_SOUL_PATH", str(tmp_path / "soul"))
    monkeypatch.delenv("MUNIN_MODELS_JSON", raising=False)

    # Absent env -> opt-out: no Metis path is selected, no secret required.
    assert get_settings().metis_config_path is None

    # Relative value -> stored as an absolute, resolved Path.
    monkeypatch.setenv("MUNIN_MODELS_JSON", "configs/models.json")
    relative_settings = get_settings()
    assert isinstance(relative_settings.metis_config_path, Path)
    assert relative_settings.metis_config_path.is_absolute()
    assert relative_settings.metis_config_path == Path("configs/models.json").resolve()

    # Absolute value -> kept absolute and resolved.
    absolute = str(tmp_path / "models.json")
    monkeypatch.setenv("MUNIN_MODELS_JSON", absolute)
    absolute_settings = get_settings()
    assert absolute_settings.metis_config_path == Path(absolute).resolve()


# ---------------------------------------------------------------------------
# 2. load_metis_if_enabled(*, path, environ) opt-in loader.
# ---------------------------------------------------------------------------


def test_load_metis_if_enabled_none_missing_and_valid_cases(tmp_path: Path) -> None:
    metis = _metis_module()
    load = getattr(metis, "load_metis_if_enabled", None)
    if load is None:
        pytest.fail(
            "RED contract expects munin.core.metis.load_metis_if_enabled(*, path, environ)"
        )

    # path=None -> opt-out, returns None.
    assert load(path=None, environ={}) is None

    # Configured but missing -> fail-fast MetisConfigError, no secret needed.
    with pytest.raises(metis.MetisConfigError):
        load(path=tmp_path / "does-not-exist" / "models.json", environ=_payload_env())

    # Valid file + matching env -> MetisConfig resolving route("munin").
    config = load(path=_write_models_json(tmp_path, _approved_payload()), environ=_payload_env())
    assert config is not None
    route = config.route("munin")
    assert route.adapter == "openai_compatible"
    assert route.model_id == "vendor/model-primary"
    assert route.base_url == "https://example.invalid/v1"


# ---------------------------------------------------------------------------
# 3. Route-injected LLMClient overrides empty settings BEFORE validation/build.
# ---------------------------------------------------------------------------


def test_llmclient_accepts_a_route_that_overrides_empty_settings_before_openai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from munin.core import llm_client as llm_client_module  # noqa: PLC0415

    route = _route_fixture(adapter="openai_compatible")
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm_client_module, "OpenAI", _FakeClient)

    # Route overrides completely empty/legacy settings; validation must see the
    # route values (non-empty), never raise LLMConfigError.
    client = llm_client_module.LLMClient(_blank_settings(tmp_path), route=route)

    assert captured["base_url"] == route.base_url
    assert captured["api_key"] == route.api_key
    assert client.settings.llm_base_url == route.base_url
    assert client.settings.llm_api_key == route.api_key
    assert client.settings.llm_model == route.model_id
    assert client.settings.llm_timeout_floor == route.timeout_seconds
    assert client.settings.llm_retry_attempts == route.retry_attempts
    assert client.settings.agent_model_call_limit == route.budget["max_model_calls"]
    assert client.settings.agent_tool_call_limit == route.budget["max_tool_calls"]

    # The route object itself stays frozen and immutable after construction.
    assert route.model_id == "vendor/model-primary"
    with pytest.raises(TypeError):
        route.budget["max_model_calls"] = 999  # type: ignore[index]
    assert route.budget["max_model_calls"] == 24


# ---------------------------------------------------------------------------
# 4. Adapter-routed make_langchain + legacy compatibility.
# ---------------------------------------------------------------------------


def test_make_langchain_routes_adapters_and_keeps_legacy_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from munin.core import llm_client as llm_client_module  # noqa: PLC0415

    constructions: list[dict[str, Any]] = []

    class _FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            constructions.append(kwargs)

    # Neutralise the real langchain_openai import inside make_langchain.
    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(llm_client_module, "OpenAI", _FakeChatOpenAI)

    # openai_compatible route -> ChatOpenAI built from the route.
    route = _route_fixture(adapter="openai_compatible")
    route_client = llm_client_module.LLMClient(_blank_settings(tmp_path), route=route)
    constructions.clear()
    routed = route_client.make_langchain()
    assert isinstance(routed, _FakeChatOpenAI)
    assert routed.kwargs["model"] == route.model_id
    assert routed.kwargs["api_key"] == route.api_key
    assert routed.kwargs["base_url"] == route.base_url
    assert routed.kwargs["timeout"] == route.timeout_seconds
    assert routed.kwargs["max_retries"] == route.retry_attempts

    # Unsupported adapter -> LLMConfigError with no ChatOpenAI construction.
    bad_route = _route_fixture(adapter="bogus_adapter")
    bad_client = llm_client_module.LLMClient(_blank_settings(tmp_path), route=bad_route)
    constructions.clear()
    with pytest.raises(llm_client_module.LLMConfigError):
        bad_client.make_langchain()
    assert constructions == []

    # Legacy, no route -> generic ChatOpenAI from plain settings.
    constructions.clear()
    legacy = llm_client_module.LLMClient(_legacy_settings(tmp_path))
    legacy_agent = legacy.make_langchain()
    assert isinstance(legacy_agent, _FakeChatOpenAI)
    assert legacy_agent.kwargs["model"] == "test-model"
    assert legacy_agent.kwargs["base_url"] == "https://llm.example.invalid/v1"
    assert legacy_agent.kwargs["api_key"] == "legacy-key"

    # Legacy DeepSeek substring -> thinking-mode path keeps forcing extra_body.
    constructions.clear()
    deepseek = llm_client_module.LLMClient(
        _legacy_settings(tmp_path, model="deepseek-v4-flash")
    )
    deepseek_agent = deepseek.make_langchain()
    assert isinstance(deepseek_agent, _FakeChatOpenAI)
    assert deepseek_agent.kwargs["extra_body"]["thinking"] == {"type": "enabled"}
