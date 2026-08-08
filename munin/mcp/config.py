from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


@dataclass(frozen=True)
class Settings:
    # --- Workspace & OFFX-original ---
    workspace_root: Path
    default_timeout: int
    max_output_chars: int
    expected_egress_ip: str
    forbidden_egress_ip: str
    route_probe_ip: str
    job_workers: int
    github_token: str
    nvd_api_key: str

    # --- LLM (OpenAI-compatible) ---
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_floor: int = 40
    llm_timeout_ceiling: int = 240
    llm_retry_attempts: int = 5
    llm_retry_base_delay: float = 5.0
    llm_retry_max_delay: float = 60.0
    # Anti-runaway budgets for a single agent invocation. These are safety
    # rails, not product limits: operators can tune or disable either one
    # explicitly with ``0`` when a deployment has an external budget policy.
    agent_model_call_limit: int = 24
    agent_tool_call_limit: int = 64
    operator_language: str = "auto"

    # --- Metis model routing (v2, opt-in) ---
    # Path to ``configs/models.json`` selected by ``MUNIN_MODELS_JSON``. Blank
    # or unset -> None (Metis is off; today's env-driven path is used). A
    # nonblank value is expanded and resolved to an absolute Path so a relative
    # ``configs/models.json`` works from any CWD. ``get_settings`` only resolves
    # the path here; it never loads Metis or requires secrets — loading is
    # delegated to ``munin.core.metis.load_metis_if_enabled``.
    metis_config_path: Path | None = None

    # --- Passive intel providers ---
    tavily_api_key: str = ""
    hugin_url: str = "https://raw.githubusercontent.com/PrinceOfPwn/Hugin/main/hugin/graph.json"
    hugin_ttl_seconds: int = 900

    # --- LDAP ---
    ldap_uri: str = "ldap://localhost:389"
    ldap_base_dn: str = "dc=akatsuki,dc=com"
    ldap_bind_dn: str = ""
    ldap_password: str = ""

    # --- OPSEC policy ---
    #   always      → preflight + postflight in EVERY tool (OFFX legacy behavior)
    #   active_only → preflight + postflight only when the tool is level=='active'
    #   off         → skip preflight (debug only; logs a warning at startup)
    preflight_policy: str = "active_only"

    # --- MCP transport ---
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8890
    mcp_auth_token: str = ""

    # --- Munin paths ---
    munin_soul_path: Path = field(default_factory=lambda: Path("./soul"))
    munin_data_path: Path = field(default_factory=lambda: Path("./data"))

    # --- Persistence backend ---
    # Empty → local sqlite file at munin_data_path/shared_state.sqlite (default).
    # ``libsql://<host>`` + ``MUNIN_DB_AUTH_TOKEN`` → Turso embedded replica.
    # ``file:/abs/path`` → explicit local file path.
    db_url: str = ""
    db_auth_token: str = ""
    # Root secret used to encrypt BYOK provider keys at rest in Turso. It is
    # deliberately environment-only and is never returned by an MCP tool.
    byok_master_key: str = ""

    # --- Fase 4: split-store persistence ---
    # ``MuninStore`` composes a "hot" SQLite backend for high-churn, non-durable
    # state (auth sessions, rate limits, recovery tokens, guidance queue,
    # in-progress agent runs) with a "durable" backend for long-lived rows
    # (users, conversations, messages, artifacts, audit trail, completed runs).
    #
    # ``hot_db_path`` — filesystem path to the SQLite database that backs the
    # hot store.  Defaults to ``/tmp/munin-hot.db``.  Data at this path is
    # intentionally disposable: every reboot invalidates outstanding sessions
    # and drops queued/running runs.  Override with ``MUNIN_HOT_DB_PATH``.
    hot_db_path: Path = field(default_factory=lambda: Path("/tmp/munin-hot.db"))
    # ``durable_db_url`` — libSQL URL of the Turso database that backs the
    # durable store.  Falls back to ``db_url`` (``MUNIN_DB_URL``) when
    # ``MUNIN_DURABLE_DB_URL`` is unset, so existing deployments keep working
    # with a single env var while Fase 4 rolls out.
    durable_db_url: str = ""
    durable_db_auth_token: str = ""

    # --- Fase 5: libsql connection pool ---
    # A bounded pool of libsql connections amortises the TLS + Hrana handshake
    # (~200-500 ms) that every DurableStore operation used to pay per request.
    # ``libsql_pool_size`` caps concurrent Turso sockets from a single Munin
    # process (default 4 — small enough to co-exist with Turso quota, large
    # enough to hide bursts of parallel readers).  ``libsql_pool_timeout_s``
    # bounds the blocking checkout window so a saturated pool surfaces as a
    # fast failure (503) instead of a hung request.  See
    # :class:`munin.mcp.persistence.LibsqlConnectionPool`.
    libsql_pool_size: int = 4
    libsql_pool_timeout_s: float = 10.0

    # --- Local-first delta sync (conversation durability) ---
    # The GUI conversation path writes to the local hot SQLite database only
    # (fast, no network).  Dirty rows are tracked in a local outbox and
    # flushed to the durable backend (Turso) at run end / shutdown.
    # ``sync_at_end`` (MUNIN_SYNC_AT_END, default on) gates the automatic
    # flushes wired to ``complete_run`` and ``close_pools``; the public
    # ``MuninStore.flush_pending_syncs()`` always syncs when called
    # explicitly.  ``sync_interval_s`` (MUNIN_SYNC_INTERVAL, default 0)
    # enables opportunistic idle syncs via ``MuninStore.sync_due()`` —
    # 0 means "only at run end / shutdown".  ``sync_batch_size``
    # (MUNIN_SYNC_BATCH_SIZE) chunks per-table uploads so a large delta
    # never saturates a single Turso transaction.
    sync_at_end: bool = True
    sync_interval_s: int = 0
    sync_batch_size: int = 500

    # --- LangGraph server (PR-11) ---
    #   MUNIN_LANGGRAPH_URL: empty string means LangGraph server not configured
    munin_langgraph_url: str = ""
    munin_langgraph_port: int = 8123
    munin_checkpoint_db: str = "data/langgraph_checkpoints.sqlite"

    # --- Parallel workers (PR-12) ---
    #   Advisory only — not a hard cap; replaces old MUNIN_MAX_PARALLEL_TOOLS
    munin_suggested_workers: int = 4

    # --- Discord adapter (follow-up to Fase 2 of issue #9) ---
    # All three are opt-in.  An empty ``discord_bot_token`` disables the
    # adapter entirely — the ASGI server never imports ``discord.py`` and
    # no background task is scheduled.  The allowlists further narrow the
    # bot's surface when it *is* enabled; empty means "no restriction"
    # (respond wherever the token can see, to whoever pings).  See
    # :mod:`munin.production.discord_adapter` for the request flow.
    discord_bot_token: str = ""
    discord_allowed_channels: str = ""
    discord_allowed_user_ids: str = ""

    @property
    def runs_root(self) -> Path:
        return self.workspace_root / "runs"

    @property
    def reports_root(self) -> Path:
        return self.workspace_root / "reports"

    @property
    def evidence_root(self) -> Path:
        return self.workspace_root / "evidence"

    @property
    def knowledge_sync_root(self) -> Path:
        return self.workspace_root / "knowledge_sync"

    @property
    def shared_state_db(self) -> Path:
        return self.munin_data_path / "shared_state.sqlite"

    @property
    def generated_tools_dir(self) -> Path:
        return self.workspace_root / "munin" / "generated"

    @property
    def generated_graphs_dir(self) -> Path:
        return self.generated_tools_dir / "graphs"

    def ensure_workspace(self) -> None:
        for path in (
            self.workspace_root,
            self.runs_root,
            self.reports_root,
            self.evidence_root,
            self.knowledge_sync_root,
            self.workspace_root / "intel",
            self.workspace_root / "prompts",
            self.workspace_root / "specs",
            self.workspace_root / "templates",
            self.munin_data_path,
            self.munin_soul_path,
            self.generated_tools_dir,
            self.generated_graphs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_root() -> Path:
    env_root = os.environ.get("OFFX_WORKSPACE_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    # `.../munin/munin/mcp/config.py` → workspace root is the project root two levels up
    return Path(__file__).resolve().parents[2]


def _resolve_path(env: str, default: Path) -> Path:
    raw = os.environ.get(env, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _env_bool(env: str, default: bool) -> bool:
    raw = os.environ.get(env, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", ""}


def get_settings() -> Settings:
    workspace = _resolve_root()
    settings = Settings(
        workspace_root=workspace,
        default_timeout=int(os.environ.get("OFFX_TIMEOUT", "300")),
        max_output_chars=int(os.environ.get("OFFX_MAX_OUTPUT_CHARS", "32000")),
        expected_egress_ip=os.environ.get("OFFX_EXPECTED_EGRESS_IP", ""),
        forbidden_egress_ip=os.environ.get("OFFX_FORBIDDEN_EGRESS_IP", ""),
        route_probe_ip=os.environ.get("OFFX_ROUTE_PROBE_IP", "1.1.1.1"),
        job_workers=int(os.environ.get("OFFX_JOB_WORKERS", "5")),
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        nvd_api_key=os.environ.get("NVD_API_KEY", ""),
        # LLM
        llm_base_url=os.environ.get("LLM_BASE_URL", "").strip(),
        llm_api_key=os.environ.get("LLM_API_KEY", "").strip(),
        llm_model=os.environ.get("LLM_MODEL", "").strip(),
        llm_timeout_floor=int(os.environ.get("LLM_TIMEOUT_FLOOR", "40")),
        llm_timeout_ceiling=int(os.environ.get("LLM_TIMEOUT_CEILING", "240")),
        llm_retry_attempts=max(1, int(os.environ.get("LLM_RETRY_ATTEMPTS", "5"))),
        llm_retry_base_delay=max(0.0, float(os.environ.get("LLM_RETRY_BASE_DELAY", "5"))),
        llm_retry_max_delay=max(0.0, float(os.environ.get("LLM_RETRY_MAX_DELAY", "60"))),
        agent_model_call_limit=max(0, int(os.environ.get("MUNIN_MODEL_CALL_LIMIT", "24"))),
        agent_tool_call_limit=max(0, int(os.environ.get("MUNIN_TOOL_CALL_LIMIT", "64"))),
        operator_language=os.environ.get("MUNIN_OPERATOR_LANGUAGE", "auto").strip() or "auto",
        # Intel providers
        tavily_api_key=os.environ.get("TAVILY_API_KEY", "").strip(),
        hugin_url=os.environ.get(
            "HUGIN_URL",
            "https://raw.githubusercontent.com/PrinceOfPwn/Hugin/main/hugin/graph.json",
        ).strip(),
        hugin_ttl_seconds=int(os.environ.get("HUGIN_TTL_SECONDS", "900")),
        # LDAP
        ldap_uri=os.environ.get("LDAP_URI", "ldap://localhost:389").strip(),
        ldap_base_dn=os.environ.get("LDAP_BASE_DN", "dc=akatsuki,dc=com").strip(),
        ldap_bind_dn=os.environ.get("LDAP_BIND_DN", "").strip(),
        ldap_password=os.environ.get("LDAP_PASSWORD", ""),
        # Policy
        preflight_policy=os.environ.get("PREFLIGHT_POLICY", "active_only").strip().lower(),
        # MCP
        mcp_host=os.environ.get("MUNIN_MCP_HOST", "127.0.0.1").strip(),
        mcp_port=int(os.environ.get("MUNIN_MCP_PORT", "8890")),
        # .strip() protects against a common footgun: `.env` files often leave a
        # trailing newline on the last line. Without strip, `hmac.compare_digest`
        # rejects every valid Bearer request because "abc\n" != "abc".
        mcp_auth_token=os.environ.get("MUNIN_MCP_AUTH_TOKEN", "").strip(),
        # Munin paths
        munin_soul_path=_resolve_path("MUNIN_SOUL_PATH", workspace / "soul"),
        munin_data_path=_resolve_path("MUNIN_DATA_PATH", workspace / "data"),
        # Persistence — empty falls back to local file
        db_url=os.environ.get("MUNIN_DB_URL", "").strip(),
        db_auth_token=os.environ.get("MUNIN_DB_AUTH_TOKEN", "").strip(),
        byok_master_key=os.environ.get("MUNIN_BYOK_MASTER_KEY", ""),
        # Fase 4 split-store paths / URLs.  ``MUNIN_DURABLE_DB_URL`` and
        # ``MUNIN_DURABLE_DB_AUTH_TOKEN`` supersede ``MUNIN_DB_URL`` /
        # ``MUNIN_DB_AUTH_TOKEN`` when set, but the legacy names remain as
        # fallbacks so existing deployments keep booting without config edits.
        hot_db_path=_resolve_path("MUNIN_HOT_DB_PATH", Path("/tmp/munin-hot.db")),
        durable_db_url=(
            os.environ.get("MUNIN_DURABLE_DB_URL", "").strip()
            or os.environ.get("MUNIN_DB_URL", "").strip()
        ),
        durable_db_auth_token=(
            os.environ.get("MUNIN_DURABLE_DB_AUTH_TOKEN", "").strip()
            or os.environ.get("MUNIN_DB_AUTH_TOKEN", "").strip()
        ),
        # Fase 5 pool tunables — both fall back to sane defaults so existing
        # deployments benefit from pooling without any env edits.
        libsql_pool_size=max(1, int(os.environ.get("MUNIN_LIBSQL_POOL_SIZE", "4"))),
        libsql_pool_timeout_s=max(
            0.1, float(os.environ.get("MUNIN_LIBSQL_POOL_TIMEOUT_S", "10.0"))
        ),
        # Local-first delta sync knobs.
        sync_at_end=_env_bool("MUNIN_SYNC_AT_END", True),
        sync_interval_s=max(0, int(os.environ.get("MUNIN_SYNC_INTERVAL", "0"))),
        sync_batch_size=max(1, int(os.environ.get("MUNIN_SYNC_BATCH_SIZE", "500"))),
        # LangGraph server (PR-11)
        munin_langgraph_url=os.environ.get("MUNIN_LANGGRAPH_URL", "").strip(),
        munin_langgraph_port=int(os.environ.get("MUNIN_LANGGRAPH_PORT", "8123")),
        munin_checkpoint_db=os.environ.get(
            "MUNIN_CHECKPOINT_DB", "data/langgraph_checkpoints.sqlite"
        ).strip(),
        # Parallel workers (PR-12)
        munin_suggested_workers=int(os.environ.get("MUNIN_SUGGESTED_WORKERS", "4")),
        # Discord adapter (all three opt-in; empty token disables the bot)
        discord_bot_token=os.environ.get("MUNIN_DISCORD_BOT_TOKEN", "").strip(),
        discord_allowed_channels=os.environ.get(
            "MUNIN_DISCORD_ALLOWED_CHANNELS", ""
        ).strip(),
        discord_allowed_user_ids=os.environ.get(
            "MUNIN_DISCORD_ALLOWED_USER_IDS", ""
        ).strip(),
        # Metis model routing (opt-in): ``MUNIN_MODELS_JSON`` selects the
        # ``configs/models.json`` path. Blank/unset -> None (Metis off; the
        # legacy env-driven LLM path is used). Relative values are expanded and
        # resolved to an absolute Path. Nothing is loaded and no secret is
        # required here — ``munin.core.metis.load_metis_if_enabled`` owns that.
        metis_config_path=(
            Path(raw).expanduser().resolve()
            if (raw := os.environ.get("MUNIN_MODELS_JSON", "").strip())
            else None
        ),
    )
    settings.ensure_workspace()
    return settings


def safe_slug(parts: Iterable[str]) -> str:
    raw = "-".join([p.strip().lower() for p in parts if p and p.strip()])
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_") or "item"


# Redact for `repr(settings)` — never spill secrets into logs.
def _redact_db_url(raw_url: str) -> str:
    if not raw_url or "://" not in raw_url:
        return raw_url
    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port:
            netloc += f":{parsed.port}"
        sanitized_query = [
            (
                key,
                "***REDACTED***"
                if any(marker in key.lower() for marker in ("token", "password", "secret", "api_key", "apikey"))
                else value,
            )
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                urlencode(sanitized_query),
                "",
            )
        )
    except (TypeError, ValueError):
        return "***REDACTED_DB_URL***"


def redact_settings(settings: Settings) -> Settings:
    return replace(
        settings,
        llm_api_key="***REDACTED***" if settings.llm_api_key else "",
        tavily_api_key="***REDACTED***" if settings.tavily_api_key else "",
        github_token="***REDACTED***" if settings.github_token else "",
        nvd_api_key="***REDACTED***" if settings.nvd_api_key else "",
        ldap_password="***REDACTED***" if settings.ldap_password else "",
        mcp_auth_token="***REDACTED***" if settings.mcp_auth_token else "",
        db_url=_redact_db_url(settings.db_url),
        db_auth_token="***REDACTED***" if settings.db_auth_token else "",
        byok_master_key="***REDACTED***" if settings.byok_master_key else "",
        durable_db_url=_redact_db_url(settings.durable_db_url),
        durable_db_auth_token=(
            "***REDACTED***" if settings.durable_db_auth_token else ""
        ),
        discord_bot_token="***REDACTED***" if settings.discord_bot_token else "",
    )
