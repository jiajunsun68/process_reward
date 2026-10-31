"""G8: separate information state from execution decisions for two G7 SFT adapters.

No training.  The only fitted parameters are one scalar temperature per existing
adapter, fitted on the new calibration N_same set.  State trajectories are
computed once per family/view/adapter; all routing strategies share them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from g7_data import (  # noqa: E402
    G7Worker, LABELS, UNKNOWN_LABEL, FIELD, MESSAGE_CONTENT, event_notice,
)
from g7_process import (  # noqa: E402
    QWEN_SNAPSHOT, SYSTEM_PROMPT, Policy, terminal_metrics, task_text,
)

DATA_DIR = ROOT / "data" / "g8_selective"
RUN_ID = "G8-SELECTIVE-a01"
RUN_DIR = ROOT / "runs" / RUN_ID
LOG_DIR = ROOT / "logs"
STATUS_PATH = ROOT / "status.md"
ADAPTERS = {
    "s11": ROOT / "runs" / "G7-PROCESS_SFT-s11-a01" / "adapter",
    "s23": ROOT / "runs" / "G7-PROCESS_SFT-s23-a01" / "adapter",
}
DATA_SEED = 801
SPLIT_SEED = 802
CANDIDATE_SEED = 803
BOOTSTRAP_SEED = 23
BOOTSTRAP_REPS = 2000
THRESHOLD = 0.975
T_MIN, T_MAX = 0.05, 100.0
WORKFLOWS = ("contact", "reminder")
INITIAL_STATES = ("known", "unknown")
FINAL_STATES = ("known", "unknown")
LENGTHS = (4, 6)
VIEWS = ("N_same", "N_new", "J")
STRATEGIES = ("ARGMAX", "RAW_CONF", "TEMP_CONF", "STRATUM_QUERY", "QUERY_ONCE", "ORACLE")
MODEL_FORWARD_LIMIT = 19456


def now_cst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data generation.
# ---------------------------------------------------------------------------

def _phone(rng: random.Random, used: set[str]) -> str:
    while True:
        value = "+1" + "".join(str(rng.randrange(10)) for _ in range(10))
        if value not in used:
            used.add(value)
            return value


def _timestamp(rng: random.Random, used: set[float]) -> float:
    while True:
        value = float(rng.randrange(1750000000, 1850000000))
        if value not in used:
            used.add(value)
            return value


def _relationship(rng: random.Random) -> str:
    return rng.choice(["neighbor", "mentor", "classmate", "vendor", "teammate", "relative", "friend"])


def _content(rng: random.Random) -> str:
    return rng.choice(["pick up the parcel", "call the clinic", "review the draft",
                       "renew the pass", "send the invoice", "water the plants",
                       "collect the keys", "confirm the booking"])


def state_after(types: list[str], initial_state: str) -> str:
    state = initial_state
    for kind in types:
        if kind == "U":
            state = "unknown"
        elif kind == "F":
            state = "known"
        # S and O leave the target field's supported state unchanged.
    return state


def _make_prefix(initial_state: str, length: int, desired_state: str,
                 special: str, rng: random.Random) -> list[str]:
    prefix_len = length - 1
    if special == "recovery":
        if prefix_len < 2:
            raise ValueError("recovery prefix too short")
        u = rng.randrange(0, prefix_len - 1)
        f = rng.randrange(u + 1, prefix_len)
        types = ["S" if rng.random() < 0.5 else "O" for _ in range(prefix_len)]
        types[u] = "U"
        types[f] = "F"
        # No U after the recovery F.
        for index in range(f + 1, prefix_len):
            if types[index] == "U":
                types[index] = "S" if rng.random() < 0.5 else "O"
        if state_after(types, initial_state) != "known":
            raise ValueError("recovery prefix did not end known")
        return types
    if special == "f_to_u":
        if prefix_len < 2:
            raise ValueError("f_to_u prefix too short")
        f = rng.randrange(0, prefix_len - 1)
        u = rng.randrange(f + 1, prefix_len)
        types = ["S" if rng.random() < 0.5 else "O" for _ in range(prefix_len)]
        types[f] = "F"
        types[u] = "U"
        for index in range(u + 1, prefix_len):
            if types[index] == "F":
                types[index] = "S" if rng.random() < 0.5 else "O"
        if state_after(types, initial_state) != "unknown":
            raise ValueError("f_to_u prefix did not end unknown")
        return types
    if desired_state == "known":
        # Construct a guaranteed-known prefix; if the initial state is unknown,
        # a public F is required.  No U after the last F.
        if initial_state == "unknown":
            f = rng.randrange(0, prefix_len)
            types = ["S" if rng.random() < 0.5 else "O" for _ in range(prefix_len)]
            types[f] = "F"
            for index in range(f + 1, prefix_len):
                if types[index] == "U":
                    types[index] = "S" if rng.random() < 0.5 else "O"
            if state_after(types, initial_state) != "known":
                raise ValueError("known prefix construction failed")
            return types
        return ["S" if rng.random() < 0.5 else "O" for _ in range(prefix_len)]
    if desired_state == "unknown":
        # Construct a guaranteed-unknown prefix.  If the initial state is known,
        # put U after every F; otherwise a no-F prefix is sufficient.
        if initial_state == "known":
            u = rng.randrange(0, prefix_len)
            types = ["S" if rng.random() < 0.5 else "O" for _ in range(prefix_len)]
            types[u] = "U"
            for index in range(u + 1, prefix_len):
                if types[index] == "F":
                    types[index] = "S" if rng.random() < 0.5 else "O"
            if state_after(types, initial_state) != "unknown":
                raise ValueError("unknown prefix construction failed")
            return types
        return ["S" if rng.random() < 0.5 else "O" for _ in range(prefix_len)]
    raise ValueError(desired_state)


def _initial_text(workflow: str, target_id: str, initial_state: str,
                  initial_row: Any) -> str:
    if initial_state == "known":
        row = initial_row[0] if isinstance(initial_row, list) and initial_row else initial_row
        return "[e1]\n" + json.dumps(row, ensure_ascii=False, sort_keys=True)
    if workflow == "contact":
        detail = f"an initial lookup of contact {target_id} returned no applicable phone-number value"
    else:
        detail = f"an initial lookup of reminder {target_id} returned no applicable scheduled-time value"
    return "[e1]\nObservation: " + detail + "."


def _n_new_notice(workflow: str, event: dict[str, Any], target_id: str, other_id: str) -> str:
    kind = event["type"]
    value = event.get("value")
    if workflow == "contact":
        value_text = str(value) if value is not None else ""
        return {
            "U": f"The telephone entry for {target_id} has been replaced; this notice does not disclose the replacement.",
            "F": f"The currently registered number for {target_id} is {value_text}.",
            "S": f"The relationship label for {target_id} was edited to {value!r}; the phone entry was left untouched.",
            "O": f"The currently registered number for {other_id} is {value_text}.",
        }[kind]
    value_text = f"{float(value):.1f}" if kind in {"F", "O"} else str(value)
    return {
        "U": f"Reminder {target_id} was assigned a different scheduled timestamp; the replacement timestamp is absent from this notification.",
        "F": f"The schedule for reminder {target_id} now records Unix timestamp {value_text}.",
        "S": f"Only the description of reminder {target_id} was edited to {value!r}; the scheduled timestamp was left unchanged.",
        "O": f"The schedule for reminder {other_id} now records Unix timestamp {value_text}.",
    }[kind]


def _history(initial_text: str, events: list[dict[str, Any]], upto: int,
             notice_key: str) -> str:
    pieces = [initial_text]
    for event in events[:upto + 1]:
        notice = event[notice_key]
        text = json.dumps(notice, ensure_ascii=False, sort_keys=True) if isinstance(notice, dict) else str(notice)
        pieces.append(f"[{event['event_id']}]\n{text}")
    return "\n".join(pieces)


def _source_entries(family: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if family["initial_state"] == "known":
        entries.append({"source_event": "e1", "source_object": family["target_id"],
                        "source_field": family["field"],
                        "source_value": family["initial_public_value"], "step_added": -1})
    for index, event in enumerate(family["events"]):
        if event["type"] == "F":
            entries.append({"source_event": event["event_id"], "source_object": family["target_id"],
                            "source_field": family["field"], "source_value": event["value"],
                            "step_added": index})
        elif event["type"] == "O":
            entries.append({"source_event": event["event_id"], "source_object": family["other_id"],
                            "source_field": family["field"], "source_value": event["value"],
                            "step_added": index})
    random.Random(f"{CANDIDATE_SEED}:{family['family_id']}").shuffle(entries)
    if len(entries) > len(LABELS):
        raise ValueError(f"too many source candidates: {len(entries)}")
    for index, entry in enumerate(entries):
        entry["label"] = LABELS[index]
    return entries


def _current_value(family: dict[str, Any], upto: int) -> Any:
    current = family["initial_public_value"] if family["initial_state"] == "known" else None
    for event in family["events"][:upto + 1]:
        if event["type"] == "U":
            current = None
        elif event["type"] == "F":
            current = event["value"]
    return current


def _final_subtypes(count: int, final_state: str) -> list[str]:
    if count % 4:
        raise ValueError("cell count must be divisible by four")
    if final_state == "unknown":
        return ["U"] * (count // 2) + ["S"] * (count // 4) + ["O"] * (count // 4)
    return ["F"] * (count // 2) + ["S"] * (count // 4) + ["O"] * (count // 4)


def make_specs() -> list[dict[str, Any]]:
    rng = random.Random(DATA_SEED)
    specs: list[dict[str, Any]] = []
    counter = 0
    for split, count in (("calibration", 16), ("test", 32)):
        for workflow in WORKFLOWS:
            for initial_state in INITIAL_STATES:
                for length in LENGTHS:
                    for final_state in FINAL_STATES:
                        subtypes = _final_subtypes(count, final_state)
                        rng.shuffle(subtypes)
                        u_seen = 0
                        u_count = count // 2
                        for index in range(count):
                            final_type = subtypes[index]
                            recovery = False
                            special_f_to_u = False
                            if length == 6 and final_state == "unknown" and final_type == "U":
                                # Half of the U-final length-6 cases contain U -> F -> final U.
                                if u_seen < u_count // 2:
                                    recovery = True
                                u_seen += 1
                            if final_state == "unknown" and final_type in {"S", "O"}:
                                special_f_to_u = True
                            if recovery:
                                desired_prefix_state = "known"
                                special = "recovery"
                            elif special_f_to_u:
                                desired_prefix_state = "unknown"
                                special = "f_to_u"
                            else:
                                desired_prefix_state = final_state
                                special = ""
                            # For final U, the non-recovery half starts unknown.
                            prefix_types = _make_prefix(initial_state, length, desired_prefix_state,
                                                       special, rng)
                            types = prefix_types + [final_type]
                            family_id = f"G8F{counter:04d}"
                            counter += 1
                            specs.append({
                                "family_id": family_id, "split": split,
                                "workflow": workflow, "initial_state": initial_state,
                                "length": length, "final_state": final_state,
                                "final_type": final_type, "types": types,
                                "recovery": recovery, "special_f_to_u": special_f_to_u,
                                "cell": f"{workflow}|{initial_state}|{length}|{final_state}",
                                "rng": rng,
                            })
    return specs


def _build_family(spec: dict[str, Any], worker: G7Worker) -> dict[str, Any]:
    rng: random.Random = spec["rng"]
    workflow = spec["workflow"]
    final_state = spec["final_state"]
    types = spec["types"]
    initial_state = spec["initial_state"]
    family_id = spec["family_id"]
    target_id = ("c_" if workflow == "contact" else "r_") + family_id[-6:].lower()
    other_id = ("c_" if workflow == "contact" else "r_") + "x" + family_id[-6:].lower()
    world_count = 2 if initial_state == "unknown" or "U" in types else 1
    used_target: set[Any] = set()
    used_other: set[Any] = set()
    field = FIELD[workflow]
    if workflow == "contact":
        initial_value = _phone(rng, used_target)
        initial_values = [initial_value] * world_count
        other_initial = _phone(rng, used_other)
        self_phone = _phone(rng, used_other)
        initial_content = None
        creation_timestamp = None
    else:
        initial_value = _timestamp(rng, used_target)
        initial_values = [initial_value] * world_count
        other_initial = _timestamp(rng, used_other)
        self_phone = None
        initial_content = "Initial reminder"
        creation_timestamp = float(rng.randrange(1700000000, 1749999999))
    if initial_state == "unknown":
        initial_values = []
        for _ in range(world_count):
            initial_values.append(_phone(rng, used_target) if workflow == "contact"
                                  else _timestamp(rng, used_target))
    family = {
        "family_id": family_id, "split": spec["split"], "cell": spec["cell"],
        "workflow": workflow, "target_id": target_id, "other_id": other_id,
        "field": field, "initial_state": initial_state, "length": spec["length"],
        "final_state": final_state, "final_type": spec["final_type"],
        "recovery": spec["recovery"], "special_f_to_u": spec["special_f_to_u"],
        "initial_public_value": initial_value if initial_state == "known" else None,
        "initial_values": initial_values, "other_initial": other_initial,
        "self_id": "self_" + family_id[-6:].lower(), "self_phone": self_phone,
        "initial_content": initial_content, "creation_timestamp": creation_timestamp,
        "events": [],
    }
    for index, kind in enumerate(types):
        event: dict[str, Any] = {"event_id": f"e{index + 2}", "type": kind}
        if kind == "U":
            event["values"] = [_phone(rng, used_target) if workflow == "contact"
                               else _timestamp(rng, used_target) for _ in range(world_count)]
        elif kind == "F":
            value = _phone(rng, used_target) if workflow == "contact" else _timestamp(rng, used_target)
            event["value"] = value
            event["values"] = [value] * world_count
        elif kind == "O":
            event["value"] = _phone(rng, used_other) if workflow == "contact" else _timestamp(rng, used_other)
        elif kind == "S":
            event["value"] = _relationship(rng) if workflow == "contact" else _content(rng)
        family["events"].append(event)
    return family


def _finish_case(family: dict[str, Any], worker: G7Worker) -> dict[str, Any]:
    spec = {
        "workflow": family["workflow"], "target_id": family["target_id"],
        "other_id": family["other_id"], "self_id": family["self_id"],
        "self_phone": family["self_phone"], "initial_state": family["initial_state"],
        "initial_values": family["initial_values"], "other_initial": family["other_initial"],
        "initial_content": family["initial_content"],
        "creation_timestamp": family["creation_timestamp"],
        "events": family["events"],
    }
    built = worker.call({"op": "build", "spec": spec})["worlds"]
    worlds = []
    for index, branch in enumerate(built):
        rows = branch["final_rows"]
        if not rows:
            raise ValueError(f"missing final row {family['family_id']} world {index}")
        final_value = rows[0].get(family["field"])
        worlds.append({
            "world_key": f"w{index}", "weight": 0.5 if len(built) == 2 else 1.0,
            "snapshot": branch["snapshot"], "final_value": final_value,
        })
    initial_row = worlds_first_initial = built[0].get("initial_public") if built else None
    initial_text = _initial_text(family["workflow"], family["target_id"],
                                 family["initial_state"], initial_row)
    # Add notices without modifying worker-facing event values.
    for event in family["events"]:
        event["notice_n_same"] = event_notice("N", "test", family["workflow"], event,
                                               family["target_id"], family["other_id"])
        event["notice_n_new"] = _n_new_notice(family["workflow"], event,
                                               family["target_id"], family["other_id"])
        event["notice_j"] = event_notice("J", "test", family["workflow"], event,
                                          family["target_id"], family["other_id"])
    sources = _source_entries(family)
    prefixes = []
    for step in range(len(family["events"])):
        current = _current_value(family, step)
        visible = [entry for entry in sources if entry["step_added"] <= step]
        if current is None:
            correct = [UNKNOWN_LABEL]
        else:
            correct = [entry["label"] for entry in visible
                       if entry["source_object"] == family["target_id"]
                       and entry["source_field"] == family["field"]
                       and entry["source_value"] == current]
            if not correct:
                raise ValueError(f"known state without source: {family['family_id']} {step}")
        candidates = [{
            "label": entry["label"], "source_event": entry["source_event"],
            "source_object": entry["source_object"], "source_field": entry["source_field"],
            "source_value": entry["source_value"],
        } for entry in visible]
        prefixes.append({
            "step": step, "event_id": family["events"][step]["event_id"],
            "history_n_same": _history(initial_text, family["events"], step, "notice_n_same"),
            "history_n_new": _history(initial_text, family["events"], step, "notice_n_new"),
            "history_j": _history(initial_text, family["events"], step, "notice_j"),
            "candidates": candidates, "correct_labels": correct,
            "unknown_label": UNKNOWN_LABEL,
            "public_state_known": current is not None,
        })
    case = {key: value for key, value in family.items()
            if key not in {"initial_values", "rng"}}
    case.update({
        "initial_public_row": initial_row,
        "initial_text": initial_text,
        "source_entries": sources,
        "prefixes": prefixes,
        "worlds": worlds,
        "data_version": "g8-selective-v1",
    })
    return case


def build_dataset(force: bool = False) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "cases.jsonl"
    manifest_path = DATA_DIR / "manifest.json"
    if cases_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        manifest["reused"] = True
        return manifest
    specs = make_specs()
    worker = G7Worker()
    cases = []
    try:
        for spec in specs:
            family = _build_family(spec, worker)
            cases.append(_finish_case(family, worker))
    finally:
        worker.close()
    tmp = cases_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    os.replace(tmp, cases_path)
    cells = Counter(case["cell"] for case in cases if case["split"] == "test")
    cal_cells = Counter(case["cell"] for case in cases if case["split"] == "calibration")
    final_events = Counter((case["split"], case["final_state"], case["final_type"]) for case in cases)
    special = Counter()
    for case in cases:
        if case["split"] != "test":
            continue
        if case["recovery"] and case["length"] == 6:
            special["recovery_final_u"] += 1
        if case["special_f_to_u"] and case["length"] == 6:
            special["unknown_maintained"] += 1
        if case["workflow"] == "reminder" and case["length"] == 6:
            special["no_prior_subset"] += 1
        if case["workflow"] == "reminder" and case["length"] == 6 and case["recovery"]:
            special["reminder_length6_recovery"] += 1
        if case["workflow"] == "reminder" and case["length"] == 6 and case["special_f_to_u"]:
            special["reminder_length6_unknown_maintained"] += 1
    forward_plan = {
        "calibration_N_same_prefixes_per_adapter": sum(len(case["prefixes"]) for case in cases
                                                       if case["split"] == "calibration"),
        "test_N_same_prefixes_per_adapter": sum(len(case["prefixes"]) for case in cases
                                                if case["split"] == "test"),
        "test_N_new_prefixes_per_adapter": sum(len(case["prefixes"]) for case in cases
                                               if case["split"] == "test"),
        "test_J_prefixes_per_adapter": sum(len(case["prefixes"]) for case in cases
                                           if case["split"] == "test"),
        "no_prior_prefixes_per_adapter": sum(len(case["prefixes"]) for case in cases
                                             if case["split"] == "test"
                                             and case["workflow"] == "reminder"
                                             and case["length"] == 6),
        "adapters": 2,
    }
    forward_plan["total_per_adapter"] = (forward_plan["calibration_N_same_prefixes_per_adapter"]
                                         + forward_plan["test_N_same_prefixes_per_adapter"]
                                         + forward_plan["test_N_new_prefixes_per_adapter"]
                                         + forward_plan["test_J_prefixes_per_adapter"]
                                         + forward_plan["no_prior_prefixes_per_adapter"])
    forward_plan["total_two_adapters"] = 2 * forward_plan["total_per_adapter"]
    if forward_plan["total_two_adapters"] != MODEL_FORWARD_LIMIT:
        raise ValueError(f"forward plan mismatch: {forward_plan}")
    manifest = {
        "dataset_version": "g8-selective-v1", "data_seed": DATA_SEED,
        "split_seed": SPLIT_SEED, "candidate_seed": CANDIDATE_SEED,
        "n_families": len(cases),
        "split_family_counts": dict(Counter(case["split"] for case in cases)),
        "n_prefixes": sum(len(case["prefixes"]) for case in cases),
        "n_worlds": sum(len(case["worlds"]) for case in cases),
        "test_cell_counts": dict(sorted(cells.items())),
        "calibration_cell_counts": dict(sorted(cal_cells.items())),
        "final_event_counts": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(final_events.items())},
        "special_counts": dict(special),
        "forward_plan": forward_plan,
        "model_forward_limit": MODEL_FORWARD_LIMIT,
        "templates": {
            "N_same": "G7 event_notice('N','test',...) frozen renderer",
            "N_new": "G8 fixed renderer in g8_selective.py",
            "J": "G7 event_notice('J','test',...) frozen renderer",
        },
        "cases_path": str(cases_path), "cases_sha256": sha256_file(cases_path),
        "cases_bytes": cases_path.stat().st_size,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reused": False,
    }
    write_json(manifest_path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Prompt construction: same task, candidates and allowed labels as G7.
# ---------------------------------------------------------------------------

def action_messages(case: dict[str, Any], prefix: dict[str, Any], view: str,
                    previous: list[str], include_prior: bool = True) -> list[dict[str, str]]:
    key = {"N_same": "history_n_same", "N_same_no_prior": "history_n_same",
           "N_new": "history_n_new", "J": "history_j"}[view]
    history = prefix[key]
    candidate_lines = []
    for candidate in prefix["candidates"]:
        candidate_lines.append(
            f"{candidate['label']} = event {candidate['source_event']}, object "
            f"{candidate['source_object']}, field {candidate['source_field']}, value "
            f"{json.dumps(candidate['source_value'], ensure_ascii=False)}")
    candidates_text = ("\n".join(candidate_lines) if candidate_lines
                       else "(no candidate source values have been observed yet)")
    allowed = [candidate["label"] for candidate in prefix["candidates"]] + [UNKNOWN_LABEL]
    prior = ""
    if include_prior:
        prior_lines = "\n".join(f"step {index + 1}: {label}" for index, label in enumerate(previous))
        if not prior_lines:
            prior_lines = "(none)"
        prior = ("\n\nPrior choices in this episode (labels only, not necessarily correct):\n"
                 f"{prior_lines}")
    user = (
        f"{task_text(case)}\n\n"
        "Public observations (oldest first):\n"
        f"{history}\n\n"
        "Candidate sources with fixed labels:\n"
        f"{candidates_text}\n"
        f"{UNKNOWN_LABEL} = UNKNOWN (no currently supported public source value)\n"
        f"{prior}\n\n"
        f"Allowed labels: {', '.join(allowed)}\n"
        "Reply with exactly one label character from the allowed labels and nothing else."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Frozen state model: one trajectory per family/view/adapter.
# ---------------------------------------------------------------------------

class StateRunner:
    def __init__(self, adapter: str, device: str, adapter_name: str) -> None:
        import torch
        self.torch = torch
        self.policy = Policy(device=device, adapter=adapter, trainable=False)
        self.policy.model.eval()
        self.adapter = adapter_name
        self.forward_count = 0
        self.input_tokens = 0
        self.prompt_tokens: list[int] = []

    def logits_for(self, messages: list[dict[str, str]], allowed: list[str]) -> list[float]:
        ids, attention = self.policy.encode(messages)
        with self.torch.inference_mode():
            logits = self.policy._logits(ids, attention, grad=False)
        label_ids = [self.policy.label_ids[label] for label in allowed]
        raw = logits[0, label_ids].float().cpu()
        self.forward_count += 1
        self.input_tokens += int(attention.sum().item())
        self.prompt_tokens.append(int(attention.sum().item()))
        return [float(value) for value in raw.tolist()]

    def trajectory(self, case: dict[str, Any], view: str, include_prior: bool,
                   split: str, phase: str) -> dict[str, Any]:
        previous: list[str] = []
        steps = []
        for prefix in case["prefixes"]:
            messages = action_messages(case, prefix, view, previous, include_prior=include_prior)
            allowed = [candidate["label"] for candidate in prefix["candidates"]] + [UNKNOWN_LABEL]
            logits = self.logits_for(messages, allowed)
            index = max(range(len(logits)), key=lambda item: logits[item])
            label = allowed[index]
            correct = label in prefix["correct_labels"]
            steps.append({
                "step": prefix["step"], "event_id": prefix["event_id"],
                "allowed": allowed, "logits": logits, "state_label": label,
                "state_correct": bool(correct),
                "public_state_known": bool(prefix["public_state_known"]),
                "event_type": case["events"][prefix["step"]]["type"],
            })
            previous.append(label)
        final = steps[-1]
        return {
            "record_type": "state_trajectory", "phase": phase, "split": split,
            "adapter": self.adapter, "view": view, "family_id": case["family_id"],
            "cell": case["cell"], "workflow": case["workflow"], "length": case["length"],
            "initial_state": case["initial_state"], "final_state": case["final_state"],
            "final_type": case["final_type"], "recovery": case["recovery"],
            "special_f_to_u": case["special_f_to_u"], "include_prior": include_prior,
            "steps": steps, "final_label": final["state_label"],
            "final_correct": final["state_correct"],
            "final_public_known": final["public_state_known"],
            "final_allowed": final["allowed"],
            "final_logits": final["logits"],
            "final_correct_labels": case["prefixes"][-1]["correct_labels"],
            "final_event_type": case["events"][-1]["type"],
        }


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    exps = [math.exp(value - maximum) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def nll_sample(logits: list[float], allowed: list[str], correct: list[str],
               temperature: float) -> float:
    scaled = [float(value) / temperature for value in logits]
    denominator = logsumexp(scaled)
    numerator = logsumexp([scaled[allowed.index(label)] for label in correct])
    return -(numerator - denominator)


def calibration_nll(records: list[dict[str, Any]], temperature: float) -> float:
    losses = []
    for record in records:
        losses.append(nll_sample(record["final_logits"], record["allowed"],
                                 record["correct_labels"], temperature))
    return sum(losses) / len(losses) if losses else float("nan")


def fit_temperature(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic one-dimensional temperature fit on log T in [.05,100]."""
    log_min, log_max = math.log(T_MIN), math.log(T_MAX)
    def f(log_t: float) -> float:
        return calibration_nll(samples, math.exp(log_t))
    grid = [log_min + (log_max - log_min) * index / 80 for index in range(81)]
    values = [(f(value), value) for value in grid]
    best_index = min(range(len(values)), key=lambda index: values[index][0])
    left = grid[max(0, best_index - 1)]
    right = grid[min(len(grid) - 1, best_index + 1)]
    # Golden-section refinement; 90 deterministic iterations.
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = left, right
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(90):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    log_t = (a + b) / 2.0
    temperature = math.exp(log_t)
    at_boundary = temperature <= T_MIN * 1.000001 or temperature >= T_MAX * 0.999999
    return {
        "temperature": temperature,
        "log_temperature": log_t,
        "at_boundary": at_boundary,
        "nll_T1": f(0.0),
        "nll_optimal": f(log_t),
        "n_samples": len(samples),
        "method": "coarse log-grid 81 + deterministic golden-section 90; bounds [0.05,100]",
        "range": [T_MIN, T_MAX],
        "initial": 1.0,
    }


# ---------------------------------------------------------------------------
# Routing decisions and native execution.
# ---------------------------------------------------------------------------

def final_probability(record: dict[str, Any], temperature: float) -> tuple[float | None, float | None]:
    allowed = record["final_allowed"]
    logits = record["final_logits"]
    raw = softmax(logits, 1.0)
    tempered = softmax(logits, temperature)
    label = record["final_label"]
    if label == UNKNOWN_LABEL:
        return None, None
    index = allowed.index(label)
    return raw[index], tempered[index]


def route_decision(strategy: str, case: dict[str, Any], record: dict[str, Any],
                   temperature: float) -> dict[str, Any]:
    final = case["prefixes"][-1]
    label = record["final_label"]
    raw_p, temp_p = final_probability(record, temperature)
    gold_known = bool(final["public_state_known"])
    state_applicable = strategy in {"ARGMAX", "RAW_CONF", "TEMP_CONF", "STRATUM_QUERY", "NO_PRIOR_ARGMAX"}
    if strategy == "QUERY_ONCE":
        return {"decision": "QUERY", "query_reason": "forced_query", "execute_label": None,
                "state_applicable": False, "low_confidence": False, "stratified": False}
    if strategy == "ORACLE":
        if not gold_known:
            return {"decision": "QUERY", "query_reason": "oracle_public_unknown",
                    "execute_label": None, "state_applicable": False,
                    "low_confidence": False, "stratified": False}
        source = final["correct_labels"][0]
        return {"decision": "EXECUTE", "query_reason": None, "execute_label": source,
                "state_applicable": False, "low_confidence": False, "stratified": False}
    if strategy == "STRATUM_QUERY" and case["workflow"] == "reminder" and case["length"] >= 6:
        return {"decision": "QUERY", "query_reason": "stratified_query", "execute_label": None,
                "state_applicable": True, "low_confidence": False, "stratified": True}
    if label == UNKNOWN_LABEL:
        return {"decision": "QUERY", "query_reason": "predicted_unknown", "execute_label": None,
                "state_applicable": state_applicable, "low_confidence": False, "stratified": False}
    if strategy in {"RAW_CONF", "TEMP_CONF"}:
        probability = raw_p if strategy == "RAW_CONF" else temp_p
        if probability is None or probability <= THRESHOLD:
            return {"decision": "QUERY", "query_reason": "low_model_confidence",
                    "execute_label": None, "state_applicable": True,
                    "low_confidence": True, "stratified": False}
    return {"decision": "EXECUTE", "query_reason": None, "execute_label": label,
            "state_applicable": state_applicable, "low_confidence": False, "stratified": False}


def execute_family(worker, case: dict[str, Any], record: dict[str, Any], strategy: str,
                   temperature: float, cache: dict[tuple[str, str, str], Any],
                   counters: dict[str, int]) -> dict[str, Any]:
    decision = route_decision(strategy, case, record, temperature)
    terminal_label = UNKNOWN_LABEL if decision["decision"] == "QUERY" else decision["execute_label"]
    world_reports = []
    for world in case["worlds"]:
        cache_key_string = f"{case['family_id']}|{world['world_key']}|{terminal_label}"
        cache_tuple = (case["family_id"], world["world_key"], terminal_label)
        hit = cache_tuple in cache
        if hit:
            counters["cache_hits"] += 1
        else:
            counters["physical_executions"] += 1
            counters["physical_queries" if terminal_label == UNKNOWN_LABEL else "physical_direct"] += 1
        terminal = terminal_metrics(worker, case, world, terminal_label, cache)
        world_reports.append({
            "world_key": world["world_key"], "weight": world["weight"],
            "success": bool(terminal["success"]), "wrong_side_effect": bool(terminal["wrong_side_effect"]),
            "tool_error": bool(terminal["tool_error"]), "abort": bool(terminal["abort"]),
            "queries": int(terminal["queries"]), "cache_key": cache_key_string,
            "cache_hit": bool(hit), "terminal_mode": "QUERY" if terminal_label == UNKNOWN_LABEL else "EXECUTE",
            "error_type": terminal.get("error_type"),
        })
    weight_sum = sum(world["weight"] for world in world_reports)
    success = sum(world["weight"] * float(world["success"]) for world in world_reports) / weight_sum
    wrong = sum(world["weight"] * float(world["wrong_side_effect"]) for world in world_reports) / weight_sum
    tool_error = sum(world["weight"] * float(world["tool_error"]) for world in world_reports) / weight_sum
    abort = sum(world["weight"] * float(world["abort"]) for world in world_reports) / weight_sum
    queries = 1 if decision["decision"] == "QUERY" else 0
    utility = 10.0 * success - 30.0 * wrong - queries
    state_correct = bool(record["final_correct"])
    gold_known = bool(record["final_public_known"])
    state_applicable = bool(decision["state_applicable"])
    prefix_count = len(case["prefixes"])
    if strategy in {"QUERY_ONCE", "ORACLE"}:
        logical_model_calls = 0
    elif strategy == "STRATUM_QUERY" and decision["stratified"]:
        logical_model_calls = 0
    else:
        logical_model_calls = prefix_count
    worlds_success = [world["success"] for world in world_reports]
    state_error_rescued = (state_applicable and not state_correct and decision["decision"] == "QUERY"
                           and all(worlds_success))
    return {
        "record_type": "execution", "phase": "native", "split": "test",
        "adapter": record["adapter"], "view": record["view"],
        "family_id": case["family_id"], "cell": case["cell"], "workflow": case["workflow"],
        "length": case["length"], "initial_state": case["initial_state"],
        "final_state": case["final_state"], "final_type": case["final_type"],
        "recovery": case["recovery"], "special_f_to_u": case["special_f_to_u"],
        "strategy": strategy, "state_label": record["final_label"] if state_applicable else None,
        "state_correct": state_correct if state_applicable else None,
        "state_applicable": state_applicable, "gold_public_known": gold_known,
        "decision": decision["decision"], "query_reason": decision["query_reason"],
        "execute_label": decision["execute_label"],
        "direct_execute": decision["decision"] == "EXECUTE",
        "predicted_unknown_query": decision["query_reason"] == "predicted_unknown",
        "low_confidence_query": decision["low_confidence"],
        "stratified_query": decision["stratified"],
        "gold_known_but_query": decision["decision"] == "QUERY" and gold_known,
        "gold_unknown_but_direct": decision["decision"] == "EXECUTE" and not gold_known,
        "state_error_rescued": bool(state_error_rescued),
        "utility": utility, "success": success, "wrong_side_effect": wrong,
        "tool_error": tool_error, "abort": abort, "queries": queries,
        "direct_conditional_wrong": (1.0 if wrong > 0 else 0.0) if decision["decision"] == "EXECUTE" else None,
        "logical_model_calls": logical_model_calls,
        "worlds": world_reports,
    }


# ---------------------------------------------------------------------------
# Aggregation and inference.
# ---------------------------------------------------------------------------

def summarize_execution(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n_families": 0}
    def mean(field: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        source = records if subset is None else subset
        values = [float(record[field]) for record in source if record.get(field) is not None]
        return sum(values) / len(values) if values else None
    direct = [record for record in records if record["direct_execute"]]
    state = [record for record in records if record["state_applicable"]]
    return {
        "n_families": len(records),
        "utility": mean("utility"), "success": mean("success"),
        "wrong_side_effect": mean("wrong_side_effect"),
        "tool_error": mean("tool_error"), "abort": mean("abort"),
        "queries": mean("queries"),
        "direct_execution_coverage": mean("direct_execute"),
        "direct_conditional_wrong": mean("direct_conditional_wrong", direct),
        "state_accuracy": mean("state_correct", state),
        "predicted_unknown_query_rate": mean("predicted_unknown_query"),
        "low_confidence_query_rate": mean("low_confidence_query"),
        "stratified_query_rate": mean("stratified_query"),
        "gold_known_but_query_rate": mean("gold_known_but_query"),
        "gold_unknown_but_direct_rate": mean("gold_unknown_but_direct"),
        "state_error_rescued_rate": (
            sum(float(record["state_error_rescued"]) for record in state)
            / max(1e-12, sum(1.0 - float(record["state_correct"]) for record in state))
            if state else None),
        "logical_model_calls_mean": mean("logical_model_calls"),
        "logical_model_calls_total": sum(int(record["logical_model_calls"]) for record in records),
    }


def average_adapter_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average per-family metrics over adapters; never increase family n."""
    by_family: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[(record["family_id"], record["view"], int(record["length"]),
                   record["strategy"], record.get("workflow", ""))].append(record)
    output = []
    for (family_id, view, length, strategy, workflow), group in sorted(by_family.items()):
        first = group[0]
        merged = dict(first)
        for field in ("utility", "success", "wrong_side_effect", "tool_error", "abort",
                      "queries", "direct_execute", "state_correct"):
            values = [float(record[field]) for record in group if record.get(field) is not None]
            merged[field] = sum(values) / len(values) if values else None
        merged["adapter"] = "adapter_mean"
        merged["logical_model_calls"] = sum(int(record["logical_model_calls"]) for record in group) / len(group)
        merged["direct_conditional_wrong"] = (
            sum(float(record["direct_conditional_wrong"]) for record in group
                if record.get("direct_conditional_wrong") is not None)
            / max(1, sum(1 for record in group if record.get("direct_conditional_wrong") is not None))
        )
        merged["predicted_unknown_query"] = (
            sum(1 for record in group if record["predicted_unknown_query"]) / len(group))
        merged["low_confidence_query"] = (
            sum(1 for record in group if record["low_confidence_query"]) / len(group))
        merged["stratified_query"] = (
            sum(1 for record in group if record["stratified_query"]) / len(group))
        merged["gold_known_but_query"] = (
            sum(1 for record in group if record["gold_known_but_query"]) / len(group))
        merged["gold_unknown_but_direct"] = (
            sum(1 for record in group if record["gold_unknown_but_direct"]) / len(group))
        merged["state_error_rescued"] = (
            sum(1 for record in group if record["state_error_rescued"]) / len(group))
        output.append(merged)
    return output


def records_for(records: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if all(record.get(key) == value for key, value in filters.items()):
            selected.append(record)
    return selected


def nll_from_state(records: list[dict[str, Any]], temperature: float) -> float:
    losses = [nll_sample(record["final_logits"], record["final_allowed"],
                         record["final_correct_labels"], temperature) for record in records]
    return sum(losses) / len(losses) if losses else float("nan")


def stratified_bootstrap(deltas: dict[str, float], cells: dict[str, str],
                         seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for family_id, delta in deltas.items():
        groups[cells[family_id]].append(delta)
    keys = sorted(groups)
    if not keys:
        return {"delta": None, "n": 0, "ci95": None, "n_pos": 0, "n_neg": 0}
    rng = random.Random(seed)
    means = []
    total_n = sum(len(groups[key]) for key in keys)
    for _ in range(reps):
        total = 0.0
        for key in keys:
            values = groups[key]
            total += sum(values[rng.randrange(len(values))] for _ in range(len(values)))
        means.append(total / total_n)
    means.sort()
    values = list(deltas.values())
    return {
        "delta": sum(values) / len(values), "n": len(values),
        "ci95": [means[int(0.025 * (reps - 1))], means[int(0.975 * (reps - 1))]],
        "n_pos": sum(1 for value in values if value > 0),
        "n_neg": sum(1 for value in values if value < 0),
        "strata": {key: len(groups[key]) for key in keys},
    }


def family_utility(records: list[dict[str, Any]], *, adapter: str | None = None,
                   view: str | None = None, strategy: str,
                   length: int | None = None, workflow: str | None = None,
                   extra: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, float]:
    result = {}
    for record in records:
        if adapter is not None and record["adapter"] != adapter:
            continue
        if view is not None and record["view"] != view:
            continue
        if record["strategy"] != strategy:
            continue
        if length is not None and record["length"] != length:
            continue
        if workflow is not None and record["workflow"] != workflow:
            continue
        if extra is not None and not extra(record):
            continue
        result[record["family_id"]] = float(record["utility"])
    return result


def compare_strategies(records: list[dict[str, Any]], view: str, length: int,
                       cells: dict[str, str], adapter: str | None = None) -> dict[str, Any]:
    if adapter is None:
        averaged = average_adapter_records(records_for(records, view=view, length=length))
        base_records = averaged
    else:
        base_records = records_for(records, view=view, length=length, adapter=adapter)
    utilities = {
        strategy: family_utility(base_records, strategy=strategy)
        for strategy in STRATEGIES
    }
    out = {}
    for left in ("TEMP_CONF", "RAW_CONF", "ARGMAX", "STRATUM_QUERY"):
        for right in ("ARGMAX", "RAW_CONF", "STRATUM_QUERY", "QUERY_ONCE"):
            if left == right:
                continue
            left_values = utilities[left]
            right_values = utilities[right]
            keys = sorted(set(left_values) & set(right_values))
            deltas = {key: left_values[key] - right_values[key] for key in keys}
            if deltas:
                out[f"{left}__minus__{right}"] = stratified_bootstrap(deltas, cells)
    return {"utilities": {strategy: {
        "mean": (sum(values.values()) / len(values)) if values else None,
        "n": len(values)} for strategy, values in utilities.items()},
        "comparisons": out}


def confidence_analysis(state_records: list[dict[str, Any]], execution_records: list[dict[str, Any]],
                        adapter: str, view: str, temperature: float) -> dict[str, Any]:
    executions = {(record["family_id"], record["strategy"]): record
                  for record in execution_records
                  if record["adapter"] == adapter and record["view"] == view and record["strategy"] == "ARGMAX"}
    rows = []
    for state in state_records:
        if state["adapter"] != adapter or state["view"] != view or state["split"] != "test":
            continue
        label = state["final_label"]
        if label == UNKNOWN_LABEL:
            continue
        raw_p, temp_p = final_probability(state, temperature)
        execution = executions.get((state["family_id"], "ARGMAX"))
        if raw_p is None or temp_p is None or execution is None:
            continue
        rows.append({"raw": raw_p, "temp": temp_p, "wrong": bool(execution["wrong_side_effect"] > 0),
                     "utility": execution["utility"], "family_id": state["family_id"]})
    thresholds = [0.0, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0]
    curves = {}
    for kind in ("raw", "temp"):
        curve = []
        for threshold in thresholds:
            covered = [row for row in rows if row[kind] > threshold]
            curve.append({
                "threshold": threshold, "n_covered": len(covered),
                "coverage": len(covered) / len(rows) if rows else None,
                "risk": (sum(1 for row in covered if row["wrong"]) / len(covered)) if covered else None,
            })
        curves[kind] = curve
    errors = [row for row in rows if row["wrong"]]
    correct = [row for row in rows if not row["wrong"]]
    def auroc(kind: str) -> float | None:
        positives = [row[kind] for row in errors]
        negatives = [row[kind] for row in correct]
        if not positives or not negatives:
            return None
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                if positive > negative:
                    wins += 1.0
                elif positive == negative:
                    wins += 0.5
        return wins / (len(positives) * len(negatives))
    return {
        "n_source_predictions": len(rows),
        "n_wrong": len(errors), "n_correct": len(correct),
        "raw_confidence_correct_mean": statistics.fmean([row["raw"] for row in correct]) if correct else None,
        "raw_confidence_wrong_mean": statistics.fmean([row["raw"] for row in errors]) if errors else None,
        "temp_confidence_correct_mean": statistics.fmean([row["temp"] for row in correct]) if correct else None,
        "temp_confidence_wrong_mean": statistics.fmean([row["temp"] for row in errors]) if errors else None,
        "raw_ge_0975_errors": sum(1 for row in errors if row["raw"] > THRESHOLD),
        "temp_ge_0975_errors": sum(1 for row in errors if row["temp"] > THRESHOLD),
        "raw_ge_0975_total": sum(1 for row in rows if row["raw"] > THRESHOLD),
        "temp_ge_0975_total": sum(1 for row in rows if row["temp"] > THRESHOLD),
        "risk_coverage": curves,
        "auroc_error_detection_raw_using_low_confidence": (
            None if auroc("raw") is None else 1.0 - auroc("raw")),
        "auroc_error_detection_temp_using_low_confidence": (
            None if auroc("temp") is None else 1.0 - auroc("temp")),
        "auroc_note": "score is low confidence (1-p); not applicable where a subset has no errors or no correct predictions; diagnostic only, no threshold selection",
    }


def build_analysis(cases: list[dict[str, Any]], state_records: list[dict[str, Any]],
                   execution_records: list[dict[str, Any]], temperatures: dict[str, Any],
                   runner_cost: dict[str, Any]) -> dict[str, Any]:
    test_cases = [case for case in cases if case["split"] == "test"]
    cells = {case["family_id"]: case["cell"] for case in test_cases}
    state_by = {(record["adapter"], record["view"], record["family_id"]): record
                for record in state_records if record["split"] == "test"}
    test_state = [record for record in state_records if record["split"] == "test"]
    test_exec = execution_records
    # NLLs.
    nlls = {}
    calibration_state = [record for record in state_records if record["split"] == "calibration"]
    for adapter, fit in temperatures.items():
        adapter_cal = [record for record in calibration_state if record["adapter"] == adapter]
        nlls[adapter] = {
            "calibration_N_same": {
                "n": len(adapter_cal),
                "nll_T1": nll_from_state(adapter_cal, 1.0),
                "nll_fitted": nll_from_state(adapter_cal, fit["temperature"]),
                "temperature": fit["temperature"], "at_boundary": fit["at_boundary"],
                "fit_nll_T1": fit["nll_T1"], "fit_nll_optimal": fit["nll_optimal"],
            },
            "test": {},
        }
        for view in VIEWS:
            subset = [record for record in test_state if record["adapter"] == adapter and record["view"] == view]
            nlls[adapter]["test"][view] = {
                "n": len(subset), "nll_T1": nll_from_state(subset, 1.0),
                "nll_fitted": nll_from_state(subset, fit["temperature"]),
            }
    # Full strategy tables per adapter/view/length.
    tables: dict[str, Any] = {}
    for adapter in ("s11", "s23", "adapter_mean"):
        rows = (average_adapter_records(test_exec)
                if adapter == "adapter_mean" else [r for r in test_exec if r["adapter"] == adapter])
        tables[adapter] = {}
        for view in VIEWS:
            tables[adapter][view] = {}
            for length in (4, 6):
                selected = [r for r in rows if r["view"] == view and r["length"] == length]
                tables[adapter][view][f"length{length}"] = {
                    strategy: summarize_execution([r for r in selected if r["strategy"] == strategy])
                    for strategy in STRATEGIES
                }
    # Main comparisons: N_same length 6, per adapter and adapter mean.
    main = {}
    for adapter in ("s11", "s23", "adapter_mean"):
        main[adapter] = compare_strategies(test_exec, "N_same", 6, cells,
                                           None if adapter == "adapter_mean" else adapter)
    length4 = {}
    for adapter in ("s11", "s23", "adapter_mean"):
        length4[adapter] = compare_strategies(test_exec, "N_same", 4, cells,
                                             None if adapter == "adapter_mean" else adapter)
    # Calibration/confidence diagnostics.
    confidence = {}
    for adapter in ("s11", "s23"):
        confidence[adapter] = {}
        for view in VIEWS:
            confidence[adapter][view] = confidence_analysis(
                state_records, test_exec, adapter, view, temperatures[adapter]["temperature"])
    # NO_PRIOR fixed subset.
    no_prior = {"by_adapter": {}, "aggregate": {}}
    for adapter in ("s11", "s23"):
        original = [r for r in test_exec if r["adapter"] == adapter and r["view"] == "N_same"
                    and r["strategy"] == "ARGMAX" and r["workflow"] == "reminder" and r["length"] == 6]
        no_prior_rows = [r for r in test_exec if r["adapter"] == adapter and r["view"] == "N_same_no_prior"
                         and r["strategy"] == "NO_PRIOR_ARGMAX"]
        original_state = [r for r in test_state if r["adapter"] == adapter and r["view"] == "N_same"
                          and r["workflow"] == "reminder" and r["length"] == 6]
        no_prior_state = [r for r in state_records if r["adapter"] == adapter
                          and r["view"] == "N_same_no_prior" and r["split"] == "test"]
        original_util = {r["family_id"]: r["utility"] for r in original}
        no_prior_util = {r["family_id"]: r["utility"] for r in no_prior_rows}
        delta = {key: no_prior_util[key] - original_util[key] for key in set(original_util) & set(no_prior_util)}
        no_prior["by_adapter"][adapter] = {
            "n": len(delta), "original": summarize_execution(original),
            "no_prior": summarize_execution(no_prior_rows),
            "delta_utility": stratified_bootstrap(delta, cells),
            "original_state_accuracy": summarize_execution(original).get("state_accuracy"),
            "no_prior_state_accuracy": summarize_execution(no_prior_rows).get("state_accuracy"),
            "original_state_n": len(original_state), "no_prior_state_n": len(no_prior_state),
        }
    # Aggregate adapter mean for NO_PRIOR.
    original_by_family = defaultdict(list)
    no_prior_by_family = defaultdict(list)
    for adapter in ("s11", "s23"):
        for record in test_exec:
            if record["adapter"] == adapter and record["view"] == "N_same" and record["strategy"] == "ARGMAX" \
                    and record["workflow"] == "reminder" and record["length"] == 6:
                original_by_family[record["family_id"]].append(record["utility"])
            if record["adapter"] == adapter and record["view"] == "N_same_no_prior" \
                    and record["strategy"] == "NO_PRIOR_ARGMAX":
                no_prior_by_family[record["family_id"]].append(record["utility"])
    delta = {key: sum(no_prior_by_family[key]) / len(no_prior_by_family[key])
             - sum(original_by_family[key]) / len(original_by_family[key])
             for key in set(original_by_family) & set(no_prior_by_family)}
    no_prior["aggregate"] = {"n": len(delta), "delta_utility": stratified_bootstrap(delta, cells)}
    # Fixed pressure subsets: reminder length6, N_same, execution rows.
    pressure = {}
    for subset_name, predicate in (
        ("recovery_final_u", lambda r: r["recovery"]),
        ("unknown_maintained", lambda r: r["special_f_to_u"]),
    ):
        pressure[subset_name] = {}
        for adapter in ("s11", "s23", "adapter_mean"):
            rows = (average_adapter_records(test_exec)
                    if adapter == "adapter_mean" else [r for r in test_exec if r["adapter"] == adapter])
            selected = [r for r in rows if r["view"] == "N_same" and r["workflow"] == "reminder"
                        and r["length"] == 6 and predicate(r)]
            pressure[subset_name][adapter] = {
                strategy: summarize_execution([r for r in selected if r["strategy"] == strategy])
                for strategy in STRATEGIES
            }
    # State accuracies must be route-invariant for ARGMAX/RAW/TEMP/STRATUM.
    state_invariance = {}
    for adapter in ("s11", "s23"):
        for view in VIEWS:
            values = {}
            for strategy in ("ARGMAX", "RAW_CONF", "TEMP_CONF", "STRATUM_QUERY"):
                rows = [r for r in test_exec if r["adapter"] == adapter and r["view"] == view
                        and r["strategy"] == strategy and r["state_applicable"]]
                values[strategy] = summarize_execution(rows).get("state_accuracy")
            state_invariance[f"{adapter}/{view}"] = values
    # Failures: at most 12 representative native failures.
    failure_candidates = [record for record in test_exec
                          if record["wrong_side_effect"] > 0 or record["abort"] > 0 or record["tool_error"] > 0]
    # Diverse representative failures: round-robin over (adapter,view,strategy),
    # then fill from the remainder.  Full failures remain in predictions.jsonl.
    selected_keys = set()
    failures = []
    def add_failure(record: dict[str, Any]) -> None:
        failures.append({
            "family_id": record["family_id"], "adapter": record["adapter"],
            "view": record["view"], "strategy": record["strategy"],
            "cell": record["cell"], "state_label": record["state_label"],
            "decision": record["decision"], "query_reason": record["query_reason"],
            "utility": record["utility"], "success": record["success"],
            "wrong_side_effect": record["wrong_side_effect"],
            "worlds": record["worlds"],
        })
    for record in failure_candidates:
        key = (record["adapter"], record["view"], record["strategy"])
        if key in selected_keys:
            continue
        selected_keys.add(key)
        add_failure(record)
        if len(failures) >= 12:
            break
    if len(failures) < 12:
        existing = {(item["family_id"], item["adapter"], item["view"], item["strategy"]) for item in failures}
        for record in failure_candidates:
            key = (record["family_id"], record["adapter"], record["view"], record["strategy"])
            if key in existing:
                continue
            add_failure(record)
            existing.add(key)
            if len(failures) >= 12:
                break
    return {
        "temperatures": temperatures, "nll": nlls, "tables": tables,
        "main_length6_N_same": main, "length4_N_same_control": length4,
        "confidence": confidence, "no_prior": no_prior,
        "pressure_subsets": pressure, "state_accuracy_invariance": state_invariance,
        "failures": failures, "runner_cost": runner_cost,
        "generator": "records: execution; adapter_mean averages same-family adapters; no view mixes",
    }


# ---------------------------------------------------------------------------
# Coordinator run.
# ---------------------------------------------------------------------------

def _phase(run: dict[str, Any], name: str, status: str, **extra: Any) -> None:
    entry = run["phases"].setdefault(name, {})
    entry.update({"status": status, "updated_at": now_cst(), **extra})
    run["updated_at"] = now_cst()
    write_json(RUN_DIR / "run.json", run)


def run_all(device: str, force_data: bool = False, resume_cache: Path | None = None) -> dict[str, Any]:
    import torch
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_dataset(force=force_data)
    cases = read_jsonl(DATA_DIR / "cases.jsonl")
    calibration_cases = [case for case in cases if case["split"] == "calibration"]
    test_cases = [case for case in cases if case["split"] == "test"]
    no_prior_cases = [case for case in test_cases if case["workflow"] == "reminder"
                      and case["length"] == 6]
    if len(calibration_cases) != 256 or len(test_cases) != 512 or len(no_prior_cases) != 128:
        raise ValueError("G8 data split mismatch")
    predictions_path = RUN_DIR / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        pass
    run = {
        "attempt_id": RUN_ID, "protocol_id": "g8-selective-v1", "status": "RUNNING",
        "started_at": now_cst(), "pid": os.getpid(), "device": device,
        "host": os.uname().nodename, "gpu_inventory_start": None,
        "adapters": {key: str(path) for key, path in ADAPTERS.items()},
        "base_model": str(QWEN_SNAPSHOT),
        "data": {
            "cases": str(DATA_DIR / "cases.jsonl"), "manifest": str(DATA_DIR / "manifest.json"),
            "cases_sha256": sha256_file(DATA_DIR / "cases.jsonl"),
            "manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
            "manifest": manifest,
        },
        "budget": {"gpu_hours_limit": 4.0, "max_gpus": 2,
                   "model_forward_limit": MODEL_FORWARD_LIMIT},
        "phases": {}, "temperatures": {}, "analysis": None,
        "cost": {"started_at": now_cst()}, "failures": [], "notes": {
            "no_training": True, "new_models": 0, "new_adapters": 0,
            "new_checkpoints": 0, "agentrm_calls": 0,
            "state_trajectory_shared_across_strategies": True,
            "state_trajectory_not_repeated_per_world": True,
            "test_used_after_temperature_freeze_only": True,
        },
    }
    try:
        import subprocess
        inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader"], capture_output=True, text=True, check=False)
        run["gpu_inventory_start"] = inventory.stdout.strip().splitlines()
    except Exception:
        run["gpu_inventory_start"] = []
    write_json(RUN_DIR / "run.json", run)
    _phase(run, "G8-DATA", "COMPLETED", cases=len(cases), prefixes=sum(len(case["prefixes"]) for case in cases),
           worlds=sum(len(case["worlds"]) for case in cases), forward_plan=manifest["forward_plan"])
    state_records: list[dict[str, Any]] = []
    no_prior_state: dict[str, list[dict[str, Any]]] = {}
    temperatures: dict[str, Any] = {}
    runner_cost: dict[str, Any] = {"by_adapter": {}, "forward_calls": 0, "input_tokens": 0}
    gpu_started = time.time()
    def write_state(record: dict[str, Any]) -> None:
        append_jsonl(predictions_path, record)

    for adapter_key in ("s11", "s23"):
        phase_name = f"G8-{adapter_key.upper()}-CALIBRATE"
        _phase(run, phase_name, "RUNNING", adapter=adapter_key)
        runner = StateRunner(str(ADAPTERS[adapter_key]), device, adapter_key)
        calibration_records = []
        cached_calibration_calls = 0
        cal_started = time.time()
        if adapter_key == "s11" and resume_cache is not None and resume_cache.exists():
            case_by_id = {case["family_id"]: case for case in calibration_cases}
            for record in read_jsonl(resume_cache):
                final = record["steps"][-1]
                case = case_by_id[record["family_id"]]
                record.setdefault("final_allowed", final["allowed"])
                record.setdefault("final_logits", final["logits"])
                record.setdefault("final_correct_labels", case["prefixes"][-1]["correct_labels"])
                record.setdefault("final_event_type", case["events"][-1]["type"])
                record["temperature"] = None
                record["adapter"] = "s11"
                calibration_records.append(record)
                state_records.append(record)
                write_state(record)
                cached_calibration_calls += len(record["steps"])
            if len(calibration_records) != len(calibration_cases):
                raise ValueError(f"resume calibration incomplete: {len(calibration_records)}")
        else:
            for case in calibration_cases:
                record = runner.trajectory(case, "N_same", True, "calibration", "calibrate")
                record["temperature"] = None
                calibration_records.append(record)
                state_records.append(record)
                write_state(record)
        cached_input_tokens = sum(int(record.get("input_tokens_total", 0)) for record in calibration_records)
        samples = [{"final_logits": record["final_logits"], "allowed": record["final_allowed"],
                    "correct_labels": record["final_correct_labels"]} for record in calibration_records]
        fit = fit_temperature(samples)
        temperatures[adapter_key] = fit
        run["temperatures"][adapter_key] = fit
        _phase(run, phase_name, "COMPLETED", calibration_families=len(calibration_records),
               temperature=fit["temperature"], at_boundary=fit["at_boundary"],
               nll_T1=fit["nll_T1"], nll_optimal=fit["nll_optimal"],
               seconds=time.time() - cal_started, forward_count=runner.forward_count,
               cached_calibration_calls=cached_calibration_calls,
               cached_input_tokens=cached_input_tokens)
        phase_name = f"G8-{adapter_key.upper()}-EVAL"
        _phase(run, phase_name, "RUNNING", adapter=adapter_key,
               temperature=fit["temperature"])
        eval_started = time.time()
        for view in VIEWS:
            for case in test_cases:
                record = runner.trajectory(case, view, True, "test", "eval")
                record["temperature"] = fit["temperature"]
                state_records.append(record)
                write_state(record)
        # NO_PRIOR is deliberately only the fixed reminder/length-6 subset.
        adapter_no_prior = []
        for case in no_prior_cases:
            record = runner.trajectory(case, "N_same_no_prior", False, "test", "no_prior")
            record["view"] = "N_same_no_prior"
            record["include_prior"] = False
            record["temperature"] = None
            record["no_prior"] = True
            adapter_no_prior.append(record)
            state_records.append(record)
            write_state(record)
        no_prior_state[adapter_key] = adapter_no_prior
        forward_this = runner.forward_count
        expected = 9728
        if adapter_key == "s11" and cached_calibration_calls:
            expected = 8448
        if forward_this != expected:
            run.setdefault("errors", []).append({"adapter": adapter_key,
                                                 "forward_count": forward_this,
                                                 "expected": expected})
        total_adapter_forward = runner.forward_count + cached_calibration_calls
        runner_cost["by_adapter"][adapter_key] = {
            "forward_calls": total_adapter_forward,
            "new_forward_calls": runner.forward_count,
            "cached_reused_forward_calls": cached_calibration_calls,
            "input_tokens": runner.input_tokens + cached_input_tokens,
            "new_input_tokens": runner.input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "eval_seconds": time.time() - eval_started,
        }
        runner_cost["forward_calls"] += total_adapter_forward
        runner_cost["input_tokens"] += runner.input_tokens + cached_input_tokens
        runner_cost["new_forward_calls"] = runner_cost.get("new_forward_calls", 0) + runner.forward_count
        run["cost"]["cached_reused_forward_calls"] = run["cost"].get("cached_reused_forward_calls", 0) + cached_calibration_calls
        _phase(run, phase_name, "COMPLETED", adapter=adapter_key,
               test_views=list(VIEWS), no_prior_families=len(adapter_no_prior),
               no_prior_prefixes=sum(len(case["prefixes"]) for case in no_prior_cases),
               eval_seconds=time.time() - eval_started, forward_count=runner.forward_count)
        del runner
        torch.cuda.empty_cache()
    gpu_ended = time.time()
    gpu_hours = (gpu_ended - gpu_started) / 3600.0
    if runner_cost["forward_calls"] != MODEL_FORWARD_LIMIT:
        run.setdefault("errors", []).append({
            "forward_count": runner_cost["forward_calls"], "expected": MODEL_FORWARD_LIMIT})
    # Native execution after both frozen model passes.
    _phase(run, "G8-NATIVE", "RUNNING", model_forward_calls=runner_cost["forward_calls"])
    worker = G7Worker()
    cache: dict[tuple[str, str, str], Any] = {}
    counters = {"physical_executions": 0, "cache_hits": 0,
                "physical_queries": 0, "physical_direct": 0}
    execution_records: list[dict[str, Any]] = []
    native_started = time.time()
    state_index = {(record["adapter"], record["view"], record["family_id"]): record
                   for record in state_records if record["split"] == "test"}
    try:
        for adapter_key in ("s11", "s23"):
            for view in VIEWS:
                for strategy in STRATEGIES:
                    for case in test_cases:
                        record = state_index[(adapter_key, view, case["family_id"])]
                        execution = execute_family(worker, case, record, strategy,
                                                   temperatures[adapter_key]["temperature"],
                                                   cache, counters)
                        execution["no_prior"] = False
                        execution_records.append(execution)
                        append_jsonl(predictions_path, execution)
            # NO_PRIOR intervention: ARGMAX only.
            for case in no_prior_cases:
                record = next(item for item in no_prior_state[adapter_key]
                              if item["family_id"] == case["family_id"])
                execution = execute_family(worker, case, record, "NO_PRIOR_ARGMAX",
                                           temperatures[adapter_key]["temperature"], cache, counters)
                execution["view"] = "N_same_no_prior"
                execution["no_prior"] = True
                execution_records.append(execution)
                append_jsonl(predictions_path, execution)
    finally:
        worker.close()
    native_seconds = time.time() - native_started
    run["cost"].update({
        "gpu_started_at": dt.datetime.fromtimestamp(gpu_started, dt.timezone.utc).isoformat(),
        "gpu_ended_at": dt.datetime.fromtimestamp(gpu_ended, dt.timezone.utc).isoformat(),
        "gpu_hours": gpu_hours, "within_4_gpu_hours": gpu_hours <= 4.0,
        "model_forward_calls": runner_cost["forward_calls"],
        "model_input_tokens": runner_cost["input_tokens"],
        "new_model_forward_calls": runner_cost.get("new_forward_calls"),
        "cached_reused_forward_calls": run["cost"].get("cached_reused_forward_calls", 0),
        "native_seconds": native_seconds,
        "physical_native_executions": counters["physical_executions"],
        "native_cache_hits": counters["cache_hits"],
        "physical_queries": counters["physical_queries"],
        "physical_direct_executions": counters["physical_direct"],
        "logical_queries": {strategy: sum(int(record["queries"]) for record in execution_records
                                          if record["strategy"] == strategy)
                            for strategy in set(record["strategy"] for record in execution_records)},
        "logical_model_calls_by_strategy_N_same_test": {
            strategy: sum(int(record["logical_model_calls"]) for record in execution_records
                          if record["strategy"] == strategy and record["view"] == "N_same")
            for strategy in STRATEGIES},
        "execution_records": len(execution_records),
        "state_prediction_records": len(state_records),
        "worker_calls_native": counters["physical_executions"],
        "adapter_forward_cost": runner_cost["by_adapter"],
    })
    _phase(run, "G8-NATIVE", "COMPLETED",
           physical_executions=counters["physical_executions"], cache_hits=counters["cache_hits"],
           physical_queries=counters["physical_queries"],
           physical_direct=counters["physical_direct"], seconds=native_seconds)
    _phase(run, "G8-ANALYZE", "RUNNING")
    analysis = build_analysis(cases, state_records, execution_records, temperatures, runner_cost)
    run["analysis"] = analysis
    run["failures"] = analysis.get("failures", [])
    run["cost"]["finished_at"] = now_cst()
    _phase(run, "G8-ANALYZE", "COMPLETED")
    run["status"] = "COMPLETED"
    run["finished_at"] = now_cst()
    run["notes"].update({
        "main_comparison": "N_same test length6 TEMP_CONF - STRATUM_QUERY native utility",
        "statistics": "family-paired stratified bootstrap 2000, seed23; adapter mean before inference",
        "no_prior_subset": "test reminder length6 N_same, 128 families, both adapters",
        "model_forward_limit": MODEL_FORWARD_LIMIT,
        "no_prior_is_not_in_main_strategy_table": True,
    })
    write_json(RUN_DIR / "run.json", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="G8 selective execution experiment")
    parser.add_argument("--mode", choices=["data", "run"], default="run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--resume-calibration-cache", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "data":
        print(json.dumps(build_dataset(force=args.force_data), indent=2, ensure_ascii=False))
        return
    run_all(args.device, force_data=args.force_data,
            resume_cache=args.resume_calibration_cache)


if __name__ == "__main__":
    main()
