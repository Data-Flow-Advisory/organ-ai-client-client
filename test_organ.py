"""
Pytest suite for the AI-client config organ.

Covers:
  - Config precedence (OPENROUTER_* over ANTHROPIC_*/CLAUDE_*, over defaults)
  - Tenant override (only non-empty values win, source tracking)
  - api_key redaction + presence reporting
  - client_buildable / confidence (fail-safe to conservative)
  - Model display-name mapping
  - Model list filter + provider-ordered sort
  - Contract shape + fail-safe on malformed input
  - Committed samples conform to pinned verdicts (catches verdict-flips)
"""

import json
import os

import pytest

from organ import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _filter_and_sort_models,
    _model_display_name,
    decide,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


class TestConfigPrecedence:
    def test_openrouter_keys_win_over_anthropic(self):
        state = {
            "app_config": {
                "OPENROUTER_MODEL": "anthropic/claude-opus-4.7",
                "CLAUDE_MODEL": "anthropic/claude-sonnet-4.6",
                "OPENROUTER_API_KEY": "sk-or-xxx",
                "ANTHROPIC_API_KEY": "sk-ant-yyy",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            }
        }
        out = decide(state)["output"]
        assert out["model_id"] == "anthropic/claude-opus-4.7"
        assert out["base_url"] == "https://openrouter.ai/api/v1"
        assert out["config_source"] == {
            "model_id": "app",
            "api_key": "app",
            "base_url": "app",
        }

    def test_falls_back_to_claude_model_then_anthropic_key(self):
        state = {
            "app_config": {
                "CLAUDE_MODEL": "anthropic/claude-sonnet-4.5",
                "ANTHROPIC_API_KEY": "sk-ant-yyy",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            }
        }
        out = decide(state)["output"]
        assert out["model_id"] == "anthropic/claude-sonnet-4.5"
        assert out["base_url"] == "https://api.anthropic.com"
        assert out["api_key_present"] is True

    def test_empty_app_config_uses_defaults(self):
        out = decide({"app_config": {}})["output"]
        assert out["model_id"] == DEFAULT_MODEL
        assert out["base_url"] == DEFAULT_BASE_URL
        assert out["api_key_present"] is False
        assert out["config_source"]["model_id"] == "default"
        assert out["config_source"]["base_url"] == "default"

    def test_missing_app_config_key_entirely(self):
        # state with no app_config at all -> still defaults, no crash
        out = decide({})["output"]
        assert out["model_id"] == DEFAULT_MODEL
        assert out["client_buildable"] is False


class TestTenantOverride:
    def test_tenant_overrides_all_three(self):
        state = {
            "app_config": {"OPENROUTER_API_KEY": "sk-or-xxx"},
            "tenant": {
                "ai_model_id": "openai/gpt-4o",
                "ai_api_key": "tenant-key",
                "ai_base_url": "https://tenant.example/api",
            },
        }
        res = decide(state)
        out = res["output"]
        assert out["model_id"] == "openai/gpt-4o"
        assert out["base_url"] == "https://tenant.example/api"
        assert out["config_source"] == {
            "model_id": "tenant",
            "api_key": "tenant",
            "base_url": "tenant",
        }
        assert res["self_metric"]["tenant_overrides"] == 3

    def test_tenant_partial_override(self):
        state = {
            "app_config": {
                "OPENROUTER_MODEL": "anthropic/claude-sonnet-4.6",
                "OPENROUTER_API_KEY": "sk-or-xxx",
            },
            "tenant": {"ai_model_id": "anthropic/claude-opus-4.7"},
        }
        res = decide(state)
        out = res["output"]
        assert out["model_id"] == "anthropic/claude-opus-4.7"
        assert out["config_source"]["model_id"] == "tenant"
        assert out["config_source"]["api_key"] == "app"
        assert res["self_metric"]["tenant_overrides"] == 1

    def test_tenant_empty_values_do_not_override(self):
        state = {
            "app_config": {
                "OPENROUTER_MODEL": "anthropic/claude-sonnet-4.6",
                "OPENROUTER_API_KEY": "sk-or-xxx",
            },
            "tenant": {"ai_model_id": "", "ai_api_key": None, "ai_base_url": ""},
        }
        res = decide(state)
        out = res["output"]
        assert out["model_id"] == "anthropic/claude-sonnet-4.6"
        assert out["config_source"]["model_id"] == "app"
        assert res["self_metric"]["tenant_overrides"] == 0

    def test_tenant_null_is_ignored(self):
        state = {
            "app_config": {"OPENROUTER_API_KEY": "sk-or-xxx"},
            "tenant": None,
        }
        assert decide(state)["self_metric"]["tenant_overrides"] == 0


class TestApiKeyRedaction:
    def test_api_key_value_never_echoed(self):
        secret = "sk-or-SUPER-SECRET-123"
        state = {"app_config": {"OPENROUTER_API_KEY": secret}}
        res = decide(state)
        blob = json.dumps(res)
        assert secret not in blob
        assert res["output"]["api_key_present"] is True

    def test_tenant_key_never_echoed(self):
        secret = "tenant-SECRET-xyz"
        state = {
            "app_config": {},
            "tenant": {"ai_api_key": secret},
        }
        assert secret not in json.dumps(decide(state))


class TestBuildableAndConfidence:
    def test_buildable_when_key_present(self):
        res = decide({"app_config": {"OPENROUTER_API_KEY": "k"}})
        assert res["output"]["client_buildable"] is True
        assert res["self_metric"]["confidence"] == 1.0
        assert res["self_metric"]["decision_path"] == "buildable"

    def test_not_buildable_without_key_is_conservative(self):
        res = decide({"app_config": {}})
        assert res["output"]["client_buildable"] is False
        assert res["self_metric"]["confidence"] == 0.5
        assert res["self_metric"]["decision_path"] == "missing_api_key"


class TestModelDisplayName:
    def test_known_id(self):
        assert _model_display_name("anthropic/claude-opus-4.7") == "Claude Opus 4.7"

    def test_unknown_namespaced_id_titlecased(self):
        assert _model_display_name("meta-llama/llama-3-70b") == "Meta Llama: Llama 3 70B"

    def test_unknown_namespaced_id_underscores_normalized(self):
        assert _model_display_name("openai/gpt-4_turbo") == "Openai: Gpt 4 Turbo"

    def test_bare_id_returned_unchanged(self):
        assert _model_display_name("local-model") == "local-model"

    def test_empty_id(self):
        assert _model_display_name("") == ""

    def test_display_name_in_output(self):
        out = decide({"app_config": {"OPENROUTER_MODEL": "anthropic/claude-opus-4.7"}})["output"]
        assert out["model_display_name"] == "Claude Opus 4.7"


class TestModelFiltering:
    def test_filters_disallowed_providers(self):
        models = [
            {"id": "anthropic/claude-sonnet-4.6", "name": "Sonnet"},
            {"id": "cohere/command-r", "name": "Command R"},
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
        ]
        out = _filter_and_sort_models(models)
        ids = [m["id"] for m in out]
        assert "cohere/command-r" not in ids
        assert "anthropic/claude-sonnet-4.6" in ids
        assert "openai/gpt-4o" in ids

    def test_provider_order_anthropic_first(self):
        models = [
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "anthropic/claude-sonnet-4.6", "name": "Sonnet"},
            {"id": "google/gemini-pro-1.5", "name": "Gemini"},
        ]
        out = _filter_and_sort_models(models)
        providers = [m["id"].split("/")[0] for m in out]
        assert providers == ["anthropic", "openai", "google"]

    def test_skips_non_dict_and_missing_id(self):
        models = ["nope", {"name": "no id"}, {"id": "anthropic/x", "name": "X"}]
        out = _filter_and_sort_models(models)
        assert len(out) == 1
        assert out[0]["id"] == "anthropic/x"

    def test_defaults_for_missing_fields(self):
        out = _filter_and_sort_models([{"id": "anthropic/x"}])
        assert out[0]["name"] == "anthropic/x"
        assert out[0]["context_length"] == 0
        assert out[0]["pricing"] == {}

    def test_models_flows_through_decide(self):
        state = {
            "app_config": {"OPENROUTER_API_KEY": "k"},
            "available_models": [
                {"id": "openai/gpt-4o", "name": "GPT-4o"},
                {"id": "anthropic/claude-sonnet-4.6", "name": "Sonnet"},
                {"id": "cohere/command", "name": "Cohere"},
            ],
        }
        res = decide(state)
        assert res["self_metric"]["models_filtered"] == 2
        assert res["output"]["models"][0]["id"] == "anthropic/claude-sonnet-4.6"

    def test_models_none_when_absent(self):
        res = decide({"app_config": {"OPENROUTER_API_KEY": "k"}})
        assert res["output"]["models"] is None
        assert res["self_metric"]["models_filtered"] == 0


class TestContractShape:
    def test_top_level_keys(self):
        res = decide({"app_config": {"OPENROUTER_API_KEY": "k"}})
        assert set(res.keys()) == {"output", "rationale", "self_metric"}
        assert isinstance(res["rationale"], str)
        assert "confidence" in res["self_metric"]
        assert 0.0 <= res["self_metric"]["confidence"] <= 1.0

    def test_failsafe_on_malformed_state(self):
        # app_config is a wrong type -> .get raises inside -> fail-safe path
        res = decide({"app_config": 12345})
        assert res["output"]["client_buildable"] is False
        assert res["self_metric"]["confidence"] in (0.0, 0.5)

    def test_deterministic(self):
        state = {
            "app_config": {"OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "anthropic/claude-opus-4.7"},
            "available_models": [{"id": "anthropic/x", "name": "X"}],
        }
        assert decide(state) == decide(state)


# Pinned expectations for each committed sample. The conformance Action only
# shadow-runs samples and prints their output — it never asserts the verdict,
# so a regression that flips a sample's decision would pass CI silently. These
# assertions are the real gate: each sample is keyed to the verdict it must
# produce.
_SAMPLE_EXPECTATIONS = {
    "app_config_openrouter.json": {
        "model_id": "anthropic/claude-sonnet-4.6",
        "model_source": "app",
        "client_buildable": True,
        "api_key_present": True,
        "decision_path": "buildable",
        "confidence": 1.0,
        "tenant_overrides": 0,
        "models_filtered": 0,
        "models_is_none": True,
    },
    "app_default_no_key.json": {
        "model_id": DEFAULT_MODEL,
        "model_source": "default",
        "client_buildable": False,
        "api_key_present": False,
        "decision_path": "missing_api_key",
        "confidence": 0.5,
        "tenant_overrides": 0,
        "models_filtered": 0,
        "models_is_none": True,
    },
    "tenant_override_with_models.json": {
        "model_id": "anthropic/claude-opus-4.7",
        "model_source": "tenant",
        "client_buildable": True,
        "api_key_present": True,
        "decision_path": "buildable",
        "confidence": 1.0,
        "tenant_overrides": 3,
        "models_filtered": 3,
        "models_is_none": False,
    },
}


class TestSamplesConform:
    """Every committed sample must round-trip AND produce its pinned verdict."""

    def test_all_samples_have_expectations(self):
        on_disk = {f for f in os.listdir(SAMPLES_DIR) if f.endswith(".json")}
        assert on_disk == set(_SAMPLE_EXPECTATIONS), (
            "samples/ and _SAMPLE_EXPECTATIONS drifted; "
            f"on_disk={sorted(on_disk)} pinned={sorted(_SAMPLE_EXPECTATIONS)}"
        )

    @pytest.mark.parametrize("name", sorted(_SAMPLE_EXPECTATIONS))
    def test_sample_conforms(self, name):
        with open(os.path.join(SAMPLES_DIR, name)) as fh:
            payload = json.load(fh)
        # Samples are stored as {"state": {...}} envelopes (what the
        # conformance Action feeds to organ.py via ORGAN_INPUT).
        assert "state" in payload, f"{name} missing 'state' envelope key"
        res = decide(payload["state"], payload.get("context"))

        # Contract shape holds for every sample.
        assert set(res.keys()) == {"output", "rationale", "self_metric"}
        out, sm = res["output"], res["self_metric"]
        exp = _SAMPLE_EXPECTATIONS[name]

        assert out["model_id"] == exp["model_id"]
        assert out["config_source"]["model_id"] == exp["model_source"]
        assert out["client_buildable"] is exp["client_buildable"]
        assert out["api_key_present"] is exp["api_key_present"]
        assert sm["decision_path"] == exp["decision_path"]
        assert sm["confidence"] == exp["confidence"]
        assert sm["tenant_overrides"] == exp["tenant_overrides"]
        assert sm["models_filtered"] == exp["models_filtered"]
        assert (out["models"] is None) is exp["models_is_none"]

    def test_no_sample_echoes_a_secret(self):
        # Defence-in-depth: no api_key value from any sample should ever
        # appear in the rendered output.
        for name in _SAMPLE_EXPECTATIONS:
            with open(os.path.join(SAMPLES_DIR, name)) as fh:
                payload = json.load(fh)
            state = payload["state"]
            blob = json.dumps(decide(state, payload.get("context")))
            for src in (state.get("app_config") or {}, state.get("tenant") or {}):
                for k, v in src.items():
                    if "key" in k.lower() and isinstance(v, str) and v:
                        assert v not in blob, f"{name} leaked secret {k}"
