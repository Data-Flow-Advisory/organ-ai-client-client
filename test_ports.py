"""
Pytest cover for the typed-ports manifest (connection standard, CONNECTORS.md).

Mirrors ports_validate.py but as assertions so the failure shows up in the
pytest leg of the conformance Action too. Also negative-checks that a bad
manifest (unknown type / undeclared output) is actually rejected — so the
validator can't silently pass.
"""

import json
import os

import pytest

import ports_validate as pv
from organ import decide

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def ports():
    return pv._load_json(pv.PORTS_PATH)


@pytest.fixture(scope="module")
def vocab():
    return pv._load_json(pv.TYPES_PATH)


def test_ports_json_parses_and_has_shape(ports):
    pv._check_shape(ports)  # raises SystemExit on bad shape
    assert isinstance(ports["inputs"], list)
    assert isinstance(ports["outputs"], list)


def test_every_declared_type_exists_in_vocabulary(ports, vocab):
    pv._check_types_exist(ports, vocab)
    known = set(vocab["types"])
    for side in ("inputs", "outputs"):
        for port in ports[side]:
            assert port["type"] in known


def test_decide_reads_and_writes_every_declared_port(ports):
    # End-to-end: no SystemExit means every input port is read and every
    # output port is written across the committed samples.
    pv._check_reads_and_writes(ports)


def test_declared_input_names_are_read_directly():
    # decide() reads `tenant` under state on every call (state.get("tenant")).
    tracked = pv._TrackingDict({"app_config": {"OPENROUTER_API_KEY": "k"}})
    decide(tracked)
    assert "tenant" in tracked.accessed


def test_declared_output_names_are_written_directly():
    out = decide({"app_config": {"OPENROUTER_API_KEY": "k"}})["output"]
    assert "ai_client_config" in out


def test_ai_client_config_port_matches_its_type_schema(vocab):
    # The additive output port's value carries exactly the AIClientConfig
    # schema keys — so a real edge would type-validate.
    schema_keys = set(vocab["types"]["AIClientConfig"]["schema"])
    cfg = decide({"app_config": {"OPENROUTER_API_KEY": "k"}})["output"]["ai_client_config"]
    assert set(cfg) == schema_keys


def test_ai_client_config_never_echoes_a_secret():
    secret = "sk-or-PORT-SECRET-999"
    out = decide({"app_config": {"OPENROUTER_API_KEY": secret}})["output"]
    assert secret not in json.dumps(out["ai_client_config"])


# --- Negative checks: a wrong manifest MUST be rejected --------------------

def test_unknown_type_is_rejected(vocab):
    bad = {"inputs": [], "outputs": [{"name": "ai_client_config", "type": "NotARealType"}]}
    with pytest.raises(SystemExit):
        pv._check_types_exist(bad, vocab)


def test_undeclared_output_is_rejected():
    bad = {
        "inputs": [],
        "outputs": [{"name": "nonexistent_output_key", "type": "AIClientConfig"}],
    }
    with pytest.raises(SystemExit):
        pv._check_reads_and_writes(bad)


def test_unread_input_is_rejected():
    bad = {
        "inputs": [{"name": "never_read_key", "type": "TenantContext", "required": False}],
        "outputs": [],
    }
    with pytest.raises(SystemExit):
        pv._check_reads_and_writes(bad)
