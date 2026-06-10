#!/usr/bin/env python3
"""
AI Client Config Organ — extracted decision logic from discovery-engine.

Source: lib/dataflow_core/ai_client/client.py

A pure decider for resolving the effective AI-client configuration:
  1. Config precedence — app-config keys with optional per-tenant override.
  2. Model display-name mapping — id -> human-readable label.
  3. Model list filtering + provider-ordered sorting (the pure part of
     fetch_available_models, minus the network fetch).

The impure parts of the source (building an `openai.OpenAI` client and the
HTTP fetch in `fetch_available_models`) are deliberately NOT part of this
organ — the spine performs those effects. This organ is handed the raw model
list (if any) in `state.available_models`; it never fetches it.

Contract:
  INPUT state: {
    "app_config": {                       # required (may be empty)
      "OPENROUTER_API_KEY": str,
      "ANTHROPIC_API_KEY": str,
      "OPENROUTER_MODEL": str,
      "CLAUDE_MODEL": str,
      "OPENROUTER_BASE_URL": str,
      "ANTHROPIC_BASE_URL": str
    },
    "tenant": {                           # optional override; null to skip
      "ai_model_id": str | null,
      "ai_api_key": str | null,
      "ai_base_url": str | null
    } | null,
    "available_models": [                 # optional; raw provider model list
      {"id": str, "name": str, "context_length": int, "pricing": {...}}
    ] | null
  }

  OUTPUT: {
    "output": {
      "model_id": str,
      "base_url": str,
      "api_key_present": bool,            # key value is redacted, never echoed
      "model_display_name": str,
      "client_buildable": bool,          # api_key present AND base_url set
      "config_source": {                  # where each value came from
        "model_id": "tenant" | "app" | "default",
        "api_key":  "tenant" | "app" | "default",
        "base_url": "tenant" | "app" | "default"
      },
      "models": [ ... ] | null            # filtered+sorted, only if input given
    },
    "rationale": str,
    "self_metric": {
      "confidence": float,                # 1.0 buildable; 0.5 missing api_key
      "decision_path": str,
      "tenant_overrides": int,            # how many fields tenant overrode
      "models_filtered": int              # count after filter (or 0)
    }
  }

Purity:
  - All inputs via JSON; no DB/network/file effects beyond reading stdin.
  - Deterministic given the same input.
  - Fail-safe: missing api_key -> client_buildable=False, lowered confidence
    (never a confident "buildable" without a key).
  - The api_key value is never echoed in output (presence only).
"""

from __future__ import annotations

import json
import os
import sys

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Provider allow-list + ordering for model filtering (mirrors source).
_ALLOWED_PROVIDER_PREFIXES = ("anthropic/", "openai/", "google/", "meta-llama/")
_PROVIDER_ORDER = {"anthropic": 0, "openai": 1, "google": 2, "meta-llama": 3}

# Human-readable display names (mirrors source map).
_DISPLAY_NAMES = {
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-opus-4.7": "Claude Opus 4.7",
    "~anthropic/claude-sonnet-latest": "Claude Sonnet (Latest)",
    "openai/gpt-4-turbo": "GPT-4 Turbo",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-pro-1.5": "Gemini Pro 1.5",
}


def _model_display_name(model_id: str) -> str:
    """Human-readable display name for a model id (pure; mirrors source)."""
    if not model_id:
        return ""
    if model_id in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[model_id]
    if "/" in model_id:
        provider, name = model_id.split("/", 1)
        return (
            f"{provider.replace('-', ' ').title()}: "
            f"{name.replace('-', ' ').replace('_', ' ').title()}"
        )
    return model_id


def _filter_and_sort_models(models: list) -> list:
    """Filter to allowed providers, then provider-order + name sort (pure)."""
    filtered = []
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not isinstance(mid, str) or not mid.startswith(_ALLOWED_PROVIDER_PREFIXES):
            continue
        filtered.append(
            {
                "id": mid,
                "name": m.get("name", mid),
                "context_length": m.get("context_length", 0),
                "pricing": m.get("pricing", {}),
            }
        )
    filtered.sort(
        key=lambda m: (_PROVIDER_ORDER.get(m["id"].split("/")[0], 99), m["name"])
    )
    return filtered


def decide(state: dict, context: dict | None = None) -> dict:
    """Resolve the effective AI-client config from app config + tenant override.

    Args:
        state: see module docstring INPUT schema.
        context: unused; present for orchestrator compatibility.

    Returns:
        {"output": {...}, "rationale": "...", "self_metric": {...}}
    """
    context = context or {}

    try:
        app_config = state.get("app_config") or {}
        tenant = state.get("tenant")
        available_models = state.get("available_models")

        # --- Config precedence (mirrors get_ai_config) -------------------
        # model_id: OPENROUTER_MODEL -> CLAUDE_MODEL -> DEFAULT
        app_model = app_config.get("OPENROUTER_MODEL") or app_config.get("CLAUDE_MODEL")
        model_id = app_model or DEFAULT_MODEL
        model_source = "app" if app_model else "default"

        # api_key: OPENROUTER_API_KEY -> ANTHROPIC_API_KEY -> "" (none)
        app_key = app_config.get("OPENROUTER_API_KEY") or app_config.get("ANTHROPIC_API_KEY")
        api_key = app_key or ""
        key_source = "app" if app_key else "default"

        # base_url: OPENROUTER_BASE_URL -> ANTHROPIC_BASE_URL -> DEFAULT
        app_base = app_config.get("OPENROUTER_BASE_URL") or app_config.get("ANTHROPIC_BASE_URL")
        base_url = app_base or DEFAULT_BASE_URL
        base_source = "app" if app_base else "default"

        # --- Tenant override (only non-empty values win) -----------------
        tenant_overrides = 0
        if isinstance(tenant, dict):
            if tenant.get("ai_model_id"):
                model_id = tenant["ai_model_id"]
                model_source = "tenant"
                tenant_overrides += 1
            if tenant.get("ai_api_key"):
                api_key = tenant["ai_api_key"]
                key_source = "tenant"
                tenant_overrides += 1
            if tenant.get("ai_base_url"):
                base_url = tenant["ai_base_url"]
                base_source = "tenant"
                tenant_overrides += 1

        api_key_present = bool(api_key)
        # A client can only be built with both a key and a base_url.
        client_buildable = api_key_present and bool(base_url)

        # --- Optional model list filtering -------------------------------
        models_out = None
        models_filtered = 0
        if isinstance(available_models, list):
            models_out = _filter_and_sort_models(available_models)
            models_filtered = len(models_out)

        display_name = _model_display_name(model_id)

        # --- Confidence (fail-safe to conservative) ----------------------
        # Full confidence only when a client could actually be built.
        if client_buildable:
            confidence = 1.0
            decision_path = "buildable"
        else:
            confidence = 0.5
            decision_path = "missing_api_key" if not api_key_present else "missing_base_url"

        rationale = (
            f"Resolved model_id='{model_id}' (from {model_source}), "
            f"base_url='{base_url}' (from {base_source}), "
            f"api_key {'present' if api_key_present else 'MISSING'} (from {key_source}); "
            f"{tenant_overrides} tenant override(s); "
            f"client_buildable={client_buildable}"
        )
        if models_out is not None:
            rationale += f"; filtered {models_filtered} allowed-provider model(s)"

        return {
            "output": {
                "model_id": model_id,
                "base_url": base_url,
                "api_key_present": api_key_present,
                "model_display_name": display_name,
                "client_buildable": client_buildable,
                "config_source": {
                    "model_id": model_source,
                    "api_key": key_source,
                    "base_url": base_source,
                },
                "models": models_out,
            },
            "rationale": rationale,
            "self_metric": {
                "confidence": confidence,
                "decision_path": decision_path,
                "tenant_overrides": tenant_overrides,
                "models_filtered": models_filtered,
            },
        }

    except Exception as e:
        # Fail-safe: emit the conservative "not buildable" verdict on the
        # safe defaults — never a confident-wrong "buildable".
        return {
            "output": {
                "model_id": DEFAULT_MODEL,
                "base_url": DEFAULT_BASE_URL,
                "api_key_present": False,
                "model_display_name": _model_display_name(DEFAULT_MODEL),
                "client_buildable": False,
                "config_source": {
                    "model_id": "default",
                    "api_key": "default",
                    "base_url": "default",
                },
                "models": None,
            },
            "rationale": f"Decision logic error (fail-safe to not-buildable): {e}",
            "self_metric": {
                "confidence": 0.0,
                "decision_path": "error_fallback",
                "tenant_overrides": 0,
                "models_filtered": 0,
            },
        }


def main() -> int:
    path = os.environ.get("ORGAN_INPUT")
    raw = open(path).read() if path else sys.stdin.read()
    try:
        payload = json.loads(raw)
        state = payload["state"]
    except Exception as e:
        print(json.dumps({"error": f"invalid input: {e}"}), file=sys.stderr)
        return 1
    print(json.dumps(decide(state, payload.get("context")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
