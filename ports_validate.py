#!/usr/bin/env python3
"""
Ports conformance validator for the connection standard (CONNECTORS.md).

Asserts three things and exits non-zero (with a human-readable reason) on the
first failure, so the conformance Action goes RED if the organ's typed ports
drift from reality:

  1. ports.json parses and has the {"inputs": [...], "outputs": [...]} shape,
     each port carrying a string `name` and a `type`.
  2. Every declared `type` exists in the shared vocabulary (vendored
     types.json — the orchestrator repo is private, so the canonical
     vocabulary is vendored here and validated against the local copy).
  3. decide() actually READS each declared input `name` under `state` and
     WRITES each declared output `name` under `output`, proven by running the
     organ on its committed samples with a key-access-tracking `state`.

Run:  python3 ports_validate.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORTS_PATH = os.path.join(HERE, "ports.json")
TYPES_PATH = os.path.join(HERE, "types.json")
SAMPLES_DIR = os.path.join(HERE, "samples")


class _TrackingDict(dict):
    """A dict that records every key looked up via .get()/[] — used to prove
    decide() reads a declared input name."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed: set = set()

    def get(self, key, default=None):
        self.accessed.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _check_shape(ports: dict) -> None:
    if not isinstance(ports, dict):
        raise SystemExit("ports.json: top level must be an object")
    for side in ("inputs", "outputs"):
        if side not in ports or not isinstance(ports[side], list):
            raise SystemExit(f"ports.json: missing list '{side}'")
        for port in ports[side]:
            if not isinstance(port, dict):
                raise SystemExit(f"ports.json: each {side} entry must be an object")
            name, typ = port.get("name"), port.get("type")
            if not isinstance(name, str) or not name:
                raise SystemExit(f"ports.json: {side} entry missing string 'name': {port!r}")
            if not isinstance(typ, str) or not typ:
                raise SystemExit(f"ports.json: port '{name}' missing string 'type'")


def _check_types_exist(ports: dict, vocab: dict) -> None:
    known = set((vocab.get("types") or {}).keys())
    if not known:
        raise SystemExit("types.json: no 'types' vocabulary found")
    for side in ("inputs", "outputs"):
        for port in ports[side]:
            if port["type"] not in known:
                raise SystemExit(
                    f"ports.json: {side} port '{port['name']}' has type "
                    f"'{port['type']}' which is not in the vocabulary (types.json)"
                )


def _check_reads_and_writes(ports: dict) -> None:
    from organ import decide

    samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(".json"))
    if not samples:
        raise SystemExit("samples/: no .json samples to validate ports against")

    input_names = {p["name"] for p in ports["inputs"]}
    output_names = {p["name"] for p in ports["outputs"]}

    read_seen: set = set()
    write_seen: set = set()

    for name in samples:
        payload = _load_json(os.path.join(SAMPLES_DIR, name))
        state = payload.get("state", payload)
        tracked = _TrackingDict(state)
        res = decide(tracked, payload.get("context"))
        read_seen |= (tracked.accessed & input_names)

        out = res.get("output")
        if not isinstance(out, dict):
            raise SystemExit(f"{name}: decide() output is not an object")
        write_seen |= (set(out.keys()) & output_names)

    missing_reads = input_names - read_seen
    if missing_reads:
        raise SystemExit(
            "ports.json declares input port(s) decide() never reads under "
            f"state across any sample: {sorted(missing_reads)}"
        )
    missing_writes = output_names - write_seen
    if missing_writes:
        raise SystemExit(
            "ports.json declares output port(s) decide() never writes under "
            f"output across any sample: {sorted(missing_writes)}"
        )


def main() -> int:
    ports = _load_json(PORTS_PATH)
    vocab = _load_json(TYPES_PATH)
    _check_shape(ports)
    _check_types_exist(ports, vocab)
    _check_reads_and_writes(ports)
    ins = ", ".join(f"{p['name']}:{p['type']}" for p in ports["inputs"]) or "(none)"
    outs = ", ".join(f"{p['name']}:{p['type']}" for p in ports["outputs"]) or "(none)"
    print(f"ports OK — inputs[{ins}] outputs[{outs}]; all types in vocabulary; reads+writes proven.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
