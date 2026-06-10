# organ-ai-client-client

A pure **decider** organ extracted from discovery-engine
`lib/dataflow_core/ai_client/client.py`. It resolves the effective AI-client
configuration from app config plus an optional per-tenant override, maps a
model id to a display name, and (when handed a raw provider model list)
filters + provider-orders it.

It implements the [orchestrator organ contract](https://github.com/Data-Flow-Advisory/orchestrator/blob/main/CONTRACT.md):
`decide(state, context) -> {output, rationale, self_metric}`, **no side effects**.

## Why this is the *pure* slice

The source `client.py` mixes pure decisions with effects:

| Source function | Nature | In this organ? |
|---|---|---|
| `get_ai_config` | pure precedence resolution | ✅ yes |
| `get_model_display_name` | pure mapping | ✅ yes |
| filtering/sorting inside `fetch_available_models` | pure | ✅ yes (handed the list) |
| `get_ai_client` (builds `openai.OpenAI`) | side effect | ❌ spine does it |
| HTTP fetch inside `fetch_available_models` | network | ❌ spine fetches, hands in `available_models` |

The organ is **handed** its facts; it never fetches them. The api_key is
**never echoed** — output reports presence only.

## Interface

Input — one JSON object on **stdin** (or the file named by `ORGAN_INPUT`):

```json
{
  "state": {
    "app_config": {
      "OPENROUTER_API_KEY": "sk-or-...",
      "ANTHROPIC_API_KEY": "sk-ant-...",
      "OPENROUTER_MODEL": "anthropic/claude-sonnet-4.6",
      "CLAUDE_MODEL": "anthropic/claude-sonnet-4.6",
      "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
      "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
    },
    "tenant": {
      "ai_model_id": "anthropic/claude-opus-4.7",
      "ai_api_key": "tenant-key",
      "ai_base_url": "https://tenant.example/api/v1"
    },
    "available_models": [
      {"id": "anthropic/claude-sonnet-4.6", "name": "Claude Sonnet 4.6",
       "context_length": 200000, "pricing": {}}
    ]
  }
}
```

`state.app_config` is required (may be empty — defaults apply). `state.tenant`
and `state.available_models` are optional.

Output — one JSON object on **stdout**:

```json
{
  "output": {
    "model_id": "anthropic/claude-opus-4.7",
    "base_url": "https://tenant.example/api/v1",
    "api_key_present": true,
    "model_display_name": "Claude Opus 4.7",
    "client_buildable": true,
    "config_source": {"model_id": "tenant", "api_key": "tenant", "base_url": "tenant"},
    "models": [ ... ]
  },
  "rationale": "Resolved model_id=... (from tenant), ...",
  "self_metric": {
    "confidence": 1.0,
    "decision_path": "buildable",
    "tenant_overrides": 3,
    "models_filtered": 2
  }
}
```

### Precedence

- **model_id**: `OPENROUTER_MODEL` → `CLAUDE_MODEL` → `anthropic/claude-sonnet-4.6`
- **api_key**: `OPENROUTER_API_KEY` → `ANTHROPIC_API_KEY` → none
- **base_url**: `OPENROUTER_BASE_URL` → `ANTHROPIC_BASE_URL` → `https://openrouter.ai/api/v1`

A non-empty `tenant.ai_*` value overrides the corresponding app value.

### Fail-safe

On malformed `state`, or when no api_key resolves, the organ returns the
conservative verdict (`client_buildable=false`, confidence ≤ 0.5) — never a
confident "buildable" without a key.

## Run

```bash
echo '{"state": {"app_config": {"OPENROUTER_API_KEY": "k"}}}' | python3 organ.py
ORGAN_INPUT=samples/tenant_override_with_models.json python3 organ.py
python3 -m pytest -v
```

## Samples

- `app_default_no_key.json` — empty config → defaults, not buildable.
- `app_config_openrouter.json` — OpenRouter app config, buildable.
- `tenant_override_with_models.json` — full tenant override + model list filter.
