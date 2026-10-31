"""G7 data: multi-event native ToolSandbox families and pointer-source labels.

The file also provides the sequential CPU worker.  It is run by the existing
G6 CPU environment; no GPU stack is imported here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
TOOL_SANDBOX = ROOT / "external" / "ToolSandbox"
CPU_PYTHON = Path("/data/sjj/g6_tools_venv/bin/python")
DATA_DIR = ROOT / "data" / "g7_process"
DATASET_VERSION = "g7-pointer-v1"
DATA_SEED = 701
SPLIT_SEED = 702
CANDIDATE_SEED = 703
LABELS = "ABCDEFGHIJ"
UNKNOWN_LABEL = "Z"
WORKFLOWS = ("contact", "reminder")
EVENT_TYPES = ("U", "F", "S", "O")
FIELD = {"contact": "phone_number", "reminder": "reminder_timestamp"}
MESSAGE_CONTENT = "G7 local simulation: status update."
INITIAL_KNOWN = "known"
INITIAL_UNKNOWN = "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CPU worker.
# ---------------------------------------------------------------------------

def _native():
    if str(TOOL_SANDBOX) not in sys.path:
        sys.path.insert(0, str(TOOL_SANDBOX))
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace, ExecutionContext, new_context,
    )
    from tool_sandbox.tools.contact import search_contacts, modify_contact  # type: ignore
    from tool_sandbox.tools.messaging import send_message_with_phone_number, search_messages  # type: ignore
    from tool_sandbox.tools.reminder import search_reminder, modify_reminder  # type: ignore
    return {
        "ns": DatabaseNamespace, "ExecutionContext": ExecutionContext, "new_context": new_context,
        "search_contacts": search_contacts, "modify_contact": modify_contact,
        "send": send_message_with_phone_number, "search_messages": search_messages,
        "search_reminder": search_reminder, "modify_reminder": modify_reminder,
    }


def _serialize(ctx) -> dict[str, Any]:
    return json.loads(json.dumps(ctx.to_dict(serialize_console=False), ensure_ascii=False))


def _restore(native, snapshot: dict[str, Any]):
    return native["ExecutionContext"].from_dict(json.loads(json.dumps(snapshot)))


def _query(native, workflow: str, target_id: str) -> list[dict[str, Any]]:
    if workflow == "contact":
        return native["search_contacts"](person_id=target_id)
    return native["search_reminder"](reminder_id=target_id)


def _build_world(native, spec: dict[str, Any], world_index: int) -> dict[str, Any]:
    workflow = spec["workflow"]
    ns = native["ns"]
    ctx = native["ExecutionContext"]()
    initial_value = spec["initial_values"][world_index]
    if workflow == "contact":
        ctx.add_to_database(ns.CONTACT, rows=[
            {"person_id": spec["self_id"], "name": "Self", "phone_number": spec["self_phone"],
             "relationship": "self", "is_self": True},
            {"person_id": spec["target_id"], "name": "Target", "phone_number": initial_value,
             "relationship": "friend", "is_self": False},
            {"person_id": spec["other_id"], "name": "Other", "phone_number": spec["other_initial"],
             "relationship": "coworker", "is_self": False},
        ])
    else:
        ctx.add_to_database(ns.REMINDER, rows=[
            {"reminder_id": spec["target_id"], "content": spec["initial_content"],
             "creation_timestamp": float(spec["creation_timestamp"]),
             "reminder_timestamp": float(initial_value), "latitude": None, "longitude": None},
            {"reminder_id": spec["other_id"], "content": "Other reminder",
             "creation_timestamp": float(spec["creation_timestamp"]) + 10.0,
             "reminder_timestamp": float(spec["other_initial"]), "latitude": None, "longitude": None},
        ])
    initial_public = None
    if spec["initial_state"] == INITIAL_KNOWN:
        with native["new_context"](ctx):
            initial_public = _query(native, workflow, spec["target_id"])
    with native["new_context"](ctx):
        for event in spec["events"]:
            kind = event["type"]
            if workflow == "contact":
                if kind in {"U", "F"}:
                    native["modify_contact"](person_id=spec["target_id"],
                                             phone_number=event["values"][world_index])
                elif kind == "S":
                    native["modify_contact"](person_id=spec["target_id"],
                                             relationship=event["value"])
                elif kind == "O":
                    native["modify_contact"](person_id=spec["other_id"],
                                             phone_number=event["value"])
            else:
                if kind in {"U", "F"}:
                    native["modify_reminder"](reminder_id=spec["target_id"],
                                              reminder_timestamp=float(event["values"][world_index]))
                elif kind == "S":
                    native["modify_reminder"](reminder_id=spec["target_id"], content=event["value"])
                elif kind == "O":
                    native["modify_reminder"](reminder_id=spec["other_id"],
                                              reminder_timestamp=float(event["value"]))
        final_rows = _query(native, workflow, spec["target_id"])
        snapshot = _serialize(ctx)
    return {"world_index": world_index, "snapshot": snapshot, "final_rows": final_rows,
            "initial_public": initial_public}


def native_build(spec: dict[str, Any]) -> dict[str, Any]:
    native = _native()
    worlds = [_build_world(native, spec, index) for index in range(len(spec["initial_values"]))]
    return {"worlds": worlds}


def native_execute(spec: dict[str, Any]) -> dict[str, Any]:
    native = _native()
    ctx = _restore(native, spec["snapshot"])
    workflow = spec["workflow"]
    target_id = spec["target_id"]
    query_value = None
    try:
        with native["new_context"](ctx):
            rows = _query(native, workflow, target_id)
            if spec["mode"] == "query":
                if not rows:
                    return {"ok": False, "error_type": "NoTargetObservation", "error": "empty query"}
                if workflow == "contact":
                    query_value = rows[0].get("phone_number")
                else:
                    query_value = rows[0].get("reminder_timestamp")
                value = query_value
            else:
                value = spec["value"]
            if workflow == "contact":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("contact source value is not a nonempty string")
                message_id = native["send"](phone_number=value, content=MESSAGE_CONTENT)
                result = native["search_messages"](message_id=message_id)
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError("reminder source value is not numeric")
                target_value = float(value) + 1800.0
                native["modify_reminder"](reminder_id=target_id, reminder_timestamp=target_value)
                result = native["search_reminder"](reminder_id=target_id)
        return {"ok": True, "result": result, "query_value": query_value, "snapshot": _serialize(ctx)}
    except Exception as error:
        return {"ok": False, "error_type": type(error).__name__, "error": str(error),
                "query_value": query_value}


def cpu_worker() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "build":
                result = native_build(request["spec"])
            elif op == "execute":
                result = native_execute(request["spec"])
            elif op == "ping":
                _native()
                result = {"ok": True}
            else:
                raise ValueError(f"unknown op {op}")
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()


class G7Worker:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(CPU_PYTHON), str(Path(__file__).resolve()), "--cpu-worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.calls = 0

    def call(self, payload: dict[str, Any]) -> Any:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.calls += 1
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("G7 CPU worker exited")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"G7 CPU worker error: {response}")
        return response["result"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)


# ---------------------------------------------------------------------------
# Public event wording.  Train/dev/test use disjoint surfaces; no event type
# or hidden value is exposed.
# ---------------------------------------------------------------------------

def _field_value_text(workflow: str, value: Any) -> str:
    if workflow == "contact":
        return str(value)
    return f"{float(value):.1f}"


def event_notice(rep: str, split: str, workflow: str, event: dict[str, Any],
                 target_id: str, other_id: str) -> Any:
    """Return one public notice in JSON (rep J) or natural language (rep N)."""
    kind = event["type"]
    value = event.get("value")
    if rep == "J":
        payload: dict[str, Any] = {"event_id": event["event_id"]}
        if kind == "U":
            payload.update({"object_id": target_id, "field": FIELD[workflow],
                            "change": "changed", "new_value_visible": False})
        elif kind == "F":
            payload.update({"object_id": target_id, "field": FIELD[workflow],
                            "change": "changed", "new_value_visible": True,
                            "new_value": value})
        elif kind == "S":
            field = "relationship" if workflow == "contact" else "content"
            payload.update({"object_id": target_id, "field": field,
                            "change": "changed", "new_value": value})
        elif kind == "O":
            payload.update({"object_id": other_id, "field": FIELD[workflow],
                            "change": "changed", "new_value_visible": True,
                            "new_value": value})
        return payload

    vt = _field_value_text(workflow, value) if kind in {"F", "O"} else ""
    # Train surfaces.
    if split == "train":
        if workflow == "contact":
            return {
                "U": f"The phone number on file for {target_id} was updated. The replacement number is not included in the notice.",
                "F": f"The phone number for {target_id} is now {vt}.",
                "S": f"The relationship field for {target_id} is now {value!r}; no phone number was supplied in this change.",
                "O": f"The phone number for {other_id} is now {vt}.",
            }[kind]
        return {
            "U": f"The scheduled time for reminder {target_id} changed. The replacement time is not included in the notice.",
            "F": f"Reminder {target_id} is now scheduled for {vt}.",
            "S": f"The text of reminder {target_id} is now {value!r}; its schedule was not part of this change.",
            "O": f"The scheduled time for reminder {other_id} is now {vt}.",
        }[kind]
    # Dev surfaces.
    if split == "dev":
        if workflow == "contact":
            return {
                "U": f"A contact update touched the phone number of {target_id}, but the new digits were withheld.",
                "F": f"Contact {target_id} now lists the phone number {vt}.",
                "S": f"The relationship label of {target_id} became {value!r}; the phone number stayed outside that update.",
                "O": f"Contact {other_id} now lists the phone number {vt}.",
            }[kind]
        return {
            "U": f"A reminder update touched reminder {target_id}'s time, while the new time was withheld.",
            "F": f"Reminder {target_id} now carries the time {vt}.",
            "S": f"The note attached to reminder {target_id} became {value!r}; the time was left outside that update.",
            "O": f"Reminder {other_id} now carries the time {vt}.",
        }[kind]
    # Held-out test surfaces.
    if workflow == "contact":
        return {
            "U": f"Someone revised the stored number for {target_id}; the revised digits were not shown.",
            "F": f"The stored number for {target_id} has been set to {vt}.",
            "S": f"The stored relationship for {target_id} has been set to {value!r}; the number was not changed by that update.",
            "O": f"The stored number for {other_id} has been set to {vt}.",
        }[kind]
    return {
        "U": f"Someone revised the stored time for reminder {target_id}; the revised time was not shown.",
        "F": f"The stored time for reminder {target_id} has been set to {vt}.",
        "S": f"The stored note for reminder {target_id} has been set to {value!r}; the time was not changed by that update.",
        "O": f"The stored time for reminder {other_id} has been set to {vt}.",
    }[kind]


def _initial_text(known: bool, workflow: str, target_id: str, value: Any,
                  initial_public_row: Any) -> str:
    if known:
        row = initial_public_row[0] if isinstance(initial_public_row, list) and initial_public_row else initial_public_row
        return "[e1]\n" + json.dumps(row, ensure_ascii=False, sort_keys=True)
    if workflow == "contact":
        detail = f"an initial lookup of contact {target_id} returned no applicable phone-number value"
    else:
        detail = f"an initial lookup of reminder {target_id} returned no applicable scheduled-time value"
    return "[e1]\nObservation: " + detail + "."


def _history(rep: str, split: str, initial_text: str, events: list[dict[str, Any]],
             target_id: str, other_id: str, upto: int) -> str:
    pieces = [initial_text]
    for index in range(upto + 1):
        event = events[index]
        notice = event["notice_j"] if rep == "J" else event["notice_n"]
        if isinstance(notice, dict):
            text = json.dumps(notice, ensure_ascii=False, sort_keys=True)
        else:
            text = str(notice)
        pieces.append(f"[{event['event_id']}]\n{text}")
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Family generation.
# ---------------------------------------------------------------------------

def _phone(rng: random.Random, used: set[str]) -> str:
    while True:
        value = "+1" + "".join(str(rng.randrange(10)) for _ in range(10))
        if value not in used:
            used.add(value)
            return value


def _timestamp(rng: random.Random, used: set[float]) -> float:
    while True:
        value = float(rng.randrange(1750000000, 1800000000))
        if value not in used:
            used.add(value)
            return value


def _relationship(rng: random.Random) -> str:
    return rng.choice(["neighbor", "mentor", "classmate", "vendor", "teammate", "relative"])


def _content(rng: random.Random) -> str:
    return rng.choice(["pick up the parcel", "call the clinic", "review the draft",
                       "renew the pass", "send the invoice", "water the plants"])


def _random_sequence(rng: random.Random, length: int) -> list[str]:
    return [rng.choices(EVENT_TYPES, weights=[0.32, 0.30, 0.19, 0.19], k=1)[0]
            for _ in range(length)]


COMPOSITION_PATTERNS = [
    ["U", "F", "S", "O", "F", "S"],  # final known
    ["F", "U", "S", "O", "U", "O"],  # final unknown
    ["U", "S", "F", "O", "U", "S"],  # final unknown
    ["F", "O", "U", "S", "F", "O"],  # final known
    ["U", "O", "F", "S", "U", "F"],  # final known
    ["F", "S", "U", "O", "F", "U"],  # final unknown
    ["U", "F", "O", "U", "S", "O"],  # final unknown, final O
    ["F", "U", "O", "F", "S", "O"],  # final known, final O
]


def make_family(family_id: str, split: str, subset: str, workflow: str,
                initial_state: str, types: list[str], rng: random.Random) -> dict[str, Any]:
    world_count = 2 if initial_state == INITIAL_UNKNOWN or "U" in types else 1
    used_target: set[Any] = set()
    used_other: set[Any] = set()
    if workflow == "contact":
        initial_value = _phone(rng, used_target)
        initial_values = [initial_value] * world_count
        other_initial = _phone(rng, used_other)
        self_phone = _phone(rng, used_other)
        initial_content = None
        creation_timestamp = None
        initial_public_value = initial_value if initial_state == INITIAL_KNOWN else None
    else:
        initial_value = _timestamp(rng, used_target)
        initial_values = [initial_value] * world_count
        other_initial = _timestamp(rng, used_other)
        self_phone = None
        initial_content = "Initial reminder"
        creation_timestamp = float(rng.randrange(1700000000, 1749999999))
        initial_public_value = initial_value if initial_state == INITIAL_KNOWN else None
    if initial_state == INITIAL_UNKNOWN:
        initial_values = []
        for _ in range(world_count):
            initial_values.append(_phone(rng, used_target) if workflow == "contact"
                                  else _timestamp(rng, used_target))
    target_id = ("c_" if workflow == "contact" else "r_") + family_id[-6:].lower()
    other_id = ("c_" if workflow == "contact" else "r_") + "x" + family_id[-6:].lower()
    family = {
        "family_id": family_id, "split": split, "subset": subset,
        "workflow": workflow, "target_id": target_id, "other_id": other_id,
        "field": FIELD[workflow], "initial_state": initial_state,
        "world_count": world_count,
        "initial_values": initial_values,
        "initial_public_value": initial_public_value,
        "other_initial": other_initial,
        "self_id": "self_" + family_id[-6:].lower(), "self_phone": self_phone,
        "initial_content": initial_content, "creation_timestamp": creation_timestamp,
        "event_types": types, "events": [],
    }
    for index, kind in enumerate(types):
        event_id = f"e{index + 2}"
        event: dict[str, Any] = {"event_id": event_id, "type": kind}
        if kind == "U":
            values = []
            for _ in range(world_count):
                values.append(_phone(rng, used_target) if workflow == "contact"
                              else _timestamp(rng, used_target))
            event["values"] = values
        elif kind == "F":
            value = _phone(rng, used_target) if workflow == "contact" else _timestamp(rng, used_target)
            event["values"] = [value] * world_count
            event["value"] = value
        elif kind == "O":
            value = _phone(rng, used_other) if workflow == "contact" else _timestamp(rng, used_other)
            event["value"] = value
        elif kind == "S":
            event["value"] = _relationship(rng) if workflow == "contact" else _content(rng)
        event["notice_j"] = event_notice("J", split, workflow, event, target_id, other_id)
        event["notice_n"] = event_notice("N", split, workflow, event, target_id, other_id)
        family["events"].append(event)
    return family


def _source_entries(family: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if family["initial_state"] == INITIAL_KNOWN:
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
    # Fixed candidate order: one shuffle per family, before any representation.
    random.Random(f"{CANDIDATE_SEED}:{family['family_id']}").shuffle(entries)
    if len(entries) > len(LABELS):
        raise ValueError(f"too many candidate sources: {len(entries)}")
    for index, entry in enumerate(entries):
        entry["label"] = LABELS[index]
    return entries


def _current_state(family: dict[str, Any], upto: int) -> Any:
    current = family["initial_public_value"] if family["initial_state"] == INITIAL_KNOWN else None
    for event in family["events"][:upto + 1]:
        if event["type"] == "U":
            current = None
        elif event["type"] == "F":
            current = event["value"]
        # S and O do not alter the target field's supported state.
    return current


def _worlds_from_worker(family: dict[str, Any], worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    worlds = []
    rows = worker_result["worlds"]
    for index, branch in enumerate(rows):
        final_rows = branch["final_rows"]
        if not final_rows:
            raise ValueError(f"missing final target row for {family['family_id']} world {index}")
        row = final_rows[0]
        final_value = row.get(family["field"])
        if final_value is None:
            raise ValueError(f"missing final field for {family['family_id']}")
        worlds.append({
            "world_key": f"w{index}",
            "weight": 0.5 if len(rows) == 2 else 1.0,
            "snapshot": branch["snapshot"],
            "final_value": final_value,
            "initial_public_row": branch.get("initial_public"),
        })
    return worlds


def build_family_with_worker(family: dict[str, Any], worker: G7Worker) -> dict[str, Any]:
    spec = {
        "workflow": family["workflow"], "target_id": family["target_id"],
        "other_id": family["other_id"], "self_id": family["self_id"],
        "self_phone": family["self_phone"], "initial_state": family["initial_state"],
        "initial_values": family["initial_values"], "other_initial": family["other_initial"],
        "initial_content": family["initial_content"], "creation_timestamp": family["creation_timestamp"],
        "events": family["events"],
    }
    worker_result = worker.call({"op": "build", "spec": spec})
    worlds = _worlds_from_worker(family, worker_result)
    sources = _source_entries(family)
    if family["initial_state"] == INITIAL_KNOWN:
        initial_public_row = worlds[0]["initial_public_row"]
        if not initial_public_row:
            raise ValueError(f"initial public row missing for {family['family_id']}")
        # Public initial row must not expose anything private beyond the target row.
        initial_text = _initial_text(True, family["workflow"], family["target_id"],
                                     family["initial_public_value"], initial_public_row)
    else:
        initial_public_row = None
        initial_text = _initial_text(False, family["workflow"], family["target_id"],
                                     None, None)
    prefixes = []
    for step in range(len(family["events"])):
        current = _current_state(family, step)
        visible_sources = [entry for entry in sources if entry["step_added"] <= step]
        if current is None:
            correct = [UNKNOWN_LABEL]
        else:
            correct = [entry["label"] for entry in visible_sources
                       if entry["source_object"] == family["target_id"]
                       and entry["source_field"] == family["field"]
                       and entry["source_value"] == current]
            if not correct:
                raise ValueError(f"known state without public source: {family['family_id']} step {step}")
        candidates = [{
            "label": entry["label"], "source_event": entry["source_event"],
            "source_object": entry["source_object"], "source_field": entry["source_field"],
            "source_value": entry["source_value"],
        } for entry in visible_sources]
        prefixes.append({
            "step": step,
            "event_id": family["events"][step]["event_id"],
            "history_j": _history("J", family["split"], initial_text, family["events"],
                                  family["target_id"], family["other_id"], step),
            "history_n": _history("N", family["split"], initial_text, family["events"],
                                  family["target_id"], family["other_id"], step),
            "candidates": candidates,
            "correct_labels": correct,
            "unknown_label": UNKNOWN_LABEL,
            "public_state_known": current is not None,
        })
    case = {key: value for key, value in family.items() if key not in {"initial_values"}}
    case.update({
        "initial_public_row": initial_public_row,
        "initial_text": initial_text,
        "source_entries": sources,
        "prefixes": prefixes,
        "worlds": [{key: value for key, value in world.items() if key != "initial_public_row"}
                   for world in worlds],
        "data_version": DATASET_VERSION,
    })
    return case


def make_dataset(force: bool = False) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "cases.jsonl"
    manifest_path = DATA_DIR / "manifest.json"
    if cases_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        manifest["reused"] = True
        return manifest
    rng = random.Random(DATA_SEED)
    specs: list[tuple[str, str, str, str, str, list[str]]] = []
    counter = 0
    def add(count: int, split: str, subset: str, workflow: str, length_mode: str,
            pattern_offset: int = 0):
        nonlocal counter
        for index in range(count):
            state = INITIAL_KNOWN if index % 2 == 0 else INITIAL_UNKNOWN
            if length_mode == "composition":
                types = list(COMPOSITION_PATTERNS[(index + pattern_offset) % len(COMPOSITION_PATTERNS)])
            else:
                length = 2 if index % 2 == 0 else 4
                types = _random_sequence(rng, length)
                # Keep a minimum of structural variety in short chains without
                # making the final event determine the answer.
                if "U" not in types and "F" not in types:
                    types[rng.randrange(length)] = rng.choice(["U", "F"])
            family_id = f"G7F{counter:04d}"
            counter += 1
            specs.append((family_id, split, subset, workflow, state, types))
    add(256, "train", "iid", "contact", "mixed")
    add(256, "train", "iid", "reminder", "mixed")
    add(16, "dev", "iid", "contact", "mixed")
    add(16, "dev", "iid", "reminder", "mixed")
    add(16, "dev", "composition", "contact", "composition")
    add(16, "dev", "composition", "reminder", "composition", pattern_offset=2)
    add(64, "test", "iid", "contact", "mixed")
    add(64, "test", "iid", "reminder", "mixed")
    add(64, "test", "composition", "contact", "composition")
    add(64, "test", "composition", "reminder", "composition", pattern_offset=1)
    families = []
    for family_id, split, subset, workflow, state, types in specs:
        family = make_family(family_id, split, subset, workflow, state, types, rng)
        families.append(family)
    worker = G7Worker()
    cases = []
    try:
        for family in families:
            cases.append(build_family_with_worker(family, worker))
    finally:
        worker.close()
    tmp = cases_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, cases_path)
    counts = Counter((case["split"], case["subset"], case["workflow"]) for case in cases)
    event_counts = Counter(event["type"] for case in cases for event in case["events"])
    initial_counts = Counter((case["split"], case["initial_state"]) for case in cases)
    final_known = Counter()
    for case in cases:
        final_known[(case["split"], case["prefixes"][-1]["public_state_known"])] += 1
    candidate_counts = Counter(len(prefix["candidates"]) for case in cases for prefix in case["prefixes"])
    world_counts = Counter(len(case["worlds"]) for case in cases)
    lengths = Counter((case["split"], case["subset"], len(case["events"])) for case in cases)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": DATA_SEED, "split_seed": SPLIT_SEED, "candidate_seed": CANDIDATE_SEED,
        "n_families": len(cases), "n_prefixes": sum(len(case["prefixes"]) for case in cases),
        "n_worlds": sum(len(case["worlds"]) for case in cases),
        "counts": {f"{a}/{b}/{c}": n for (a, b, c), n in sorted(counts.items())},
        "event_type_counts": dict(sorted(event_counts.items())),
        "initial_state_counts": {f"{a}/{b}": n for (a, b), n in sorted(initial_counts.items())},
        "final_known_counts": {f"{a}/{str(b).lower()}": n for (a, b), n in sorted(final_known.items())},
        "candidate_count_histogram": {str(k): v for k, v in sorted(candidate_counts.items())},
        "world_count_histogram": {str(k): v for k, v in sorted(world_counts.items())},
        "length_counts": {f"{a}/{b}/{c}": n for (a, b, c), n in sorted(lengths.items())},
        "generation_worker_calls": worker.calls,
        "cpu_environment": str(CPU_PYTHON), "tool_sandbox_revision": "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
        "cases_path": str(cases_path), "cases_sha256": sha256_file(cases_path),
        "cases_bytes": cases_path.stat().st_size,
        "candidate_order_rule": "one fixed shuffle per family; same J/N order; labels A..J; Z=UNKNOWN",
        "initial_state_rule": "half known and half unknown within each family block",
        "test_surface_isolated": True,
        "created": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "reused": False,
    }
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-worker", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cpu_worker:
        cpu_worker()
    elif args.build:
        print(json.dumps(make_dataset(force=args.force), indent=2, ensure_ascii=False))
    else:
        parser.print_help()
