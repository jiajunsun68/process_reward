"""G6: validity-aware review and minimal repair on native ToolSandbox workflows.

One coordinator handles the frozen Qwen model and one sequential CPU worker
handles the native ToolSandbox execution contexts.  No model or tool state is
shared across processes beyond explicit public observations and JSON snapshots.
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
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "g6_validity"
RUN_DIR = ROOT / "runs" / "G6-VALIDITY-a01"
LOG_PATH = ROOT / "logs" / "G6-VALIDITY-a01.log"
TOOL_SANDBOX = ROOT / "external" / "ToolSandbox"
CPU_PYTHON = Path("/data/sjj/g6_tools_venv/bin/python")
QWEN_VENV = Path("/data/sjj/process_reward_venv/bin/python")
QWEN_SNAPSHOT = Path("/data4/sjj/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/"
                    "snapshots/2de2439ea21be1dc5cb21f22f88af07e43393cbb")
TOOL_REVISION = "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"
DATA_SEED = 601
SPLIT_SEED = 602
BOOTSTRAP_SEED = 23
BOOTSTRAP_REPS = 2000
DATASET_VERSION = "g6-validity-v1"
PROMPT_VERSION = "g6-prompts-v1"
MODEL_CALL_LIMIT = 3888
PHASES = ("G6-CPU-REANALYSIS", "G6-TOOLS-DATA", "G6-DEV", "G6-AUDIT", "G6-EXECUTE", "G6-ANALYZE")
CONDITIONS = ("U", "F", "S", "O")
REPRESENTATIONS = ("J", "N")
STRATEGIES = ("DIRECT_E", "GENERIC_REVIEW", "VALIDITY_REVIEW", "QUERY_ONCE_V", "DEPENDENCY_RULE")
WORKFLOWS = ("contact", "reminder")
MESSAGE_CONTENT = "G6 local simulation: please confirm the scheduled update."
RELATION_VALUES = {"KEEP", "INVALIDATE", "REPLACE", "UNRESOLVED"}
FIELD_BY_WORKFLOW = {"contact": "phone_number", "reminder": "reminder_timestamp"}

# ---------------------------------------------------------------------------
# Fixed G6 prompts (copied from the task specification).
# ---------------------------------------------------------------------------
E_SYSTEM = """You extract the value currently established for one task-relevant field of one target object.

Use only the public task, the stated field dependency, and the visible interaction history.
Identify the exact target. Determine what value is supported at the current decision point.
A value mentioned earlier is not automatically a current value. Do not invent a missing value.
An event about another object or an unrelated field does not by itself establish a new value for the requested field.

Return knowledge="known" only when the visible evidence establishes the current value.
Otherwise return knowledge="unknown" and value=null.
Cite the supplied event IDs relevant to your conclusion.
Do not choose a tool, calculate task utility, or provide an explanation.

Return exactly:
{"knowledge":"known or unknown","value":null,"evidence":["e1"]}
For known, replace null with the supported phone-number string or integer Unix timestamp."""

R_SYSTEM = """Review a draft extraction against the complete visible history.
Check whether it correctly identifies what is currently known about the requested field of the exact target.
Keep a correct draft and correct an incorrect draft. Use only the provided evidence.

Also classify the relationship between the initial cached value and the current evidence:
KEEP, INVALIDATE, REPLACE, or UNRESOLVED.
KEEP means that the initial value remains current. INVALIDATE means its current support was lost without an established replacement. REPLACE means that visible evidence supplies a current replacement. UNRESOLVED means that this relationship cannot be determined.
Use known with a supported current value, or unknown with null.
Return the final answer, not comments about the draft. Do not select a tool or give a chain of reasoning.

Return exactly:
{"relation":"KEEP or INVALIDATE or REPLACE or UNRESOLVED","knowledge":"known or unknown","value":null,"evidence":["e1"]}"""

V_SYSTEM = """Review a draft extraction by checking whether its evidence still applies to the requested field of the exact target.

Start with the initial cached observation and inspect the later visible events.
Distinguish three situations:
- The event changes another object or an unrelated field: retain the still-supported value.
- A relevant change removes support for the old value but does not reveal the current value: withdraw the old value and report unknown.
- New applicable evidence supplies the current value: use that value.

A later lookup may restore knowledge after an earlier change.
Do not preserve an old value merely because its source event exists.
Do not turn every update into uncertainty, and do not treat an unknown replacement as a known negative value.

Classify the initial cached value as KEEP, INVALIDATE, REPLACE, or UNRESOLVED.
Then independently make the final knowledge/value agree with the current visible support.
Use only public evidence. Do not select a tool or output a chain of reasoning.

Return exactly:
{"relation":"KEEP or INVALIDATE or REPLACE or UNRESOLVED","knowledge":"known or unknown","value":null,"evidence":["e1"]}"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def object_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode())


def now_cst() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")


def initialize_run() -> dict[str, Any]:
    path = RUN_DIR / "run.json"
    if path.is_file():
        run = json.loads(path.read_text())
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        run = {
            "attempt_id": "G6-VALIDITY-a01",
            "dataset_version": DATASET_VERSION,
            "tool_sandbox_revision": TOOL_REVISION,
            "started_at": now_cst(),
            "phase_status": {phase: "PENDING" for phase in PHASES},
            "phase_attempts": [],
            "phase_evidence": {},
            "budget": {
                "model_calls": 0, "model_call_limit": MODEL_CALL_LIMIT,
                "input_tokens": 0, "output_tokens": 0,
                "gpu_hours": 0.0, "training_runs": 0,
                "checkpoints": 0, "adapters": 0,
            },
            "metrics": {},
        }
    for phase in PHASES:
        run.setdefault("phase_status", {}).setdefault(phase, "PENDING")
    run.setdefault("phase_attempts", [])
    run.setdefault("phase_evidence", {})
    run.setdefault("budget", {})
    run.setdefault("metrics", {})
    run["updated_at"] = now_cst()
    write_json(path, run)
    return run


def mark_phase(run: dict[str, Any], phase: str, status: str, **extra: Any) -> None:
    run["phase_status"][phase] = status
    if extra:
        run.setdefault("phase_evidence", {}).setdefault(phase, {}).update(extra)
    run["updated_at"] = now_cst()
    write_json(RUN_DIR / "run.json", run)


# ---------------------------------------------------------------------------
# G5 CPU reanalysis (no G5 model calls, no G5 file changes).
# ---------------------------------------------------------------------------

def g5_reanalysis() -> dict[str, Any]:
    from trajectory_cases import load_cases
    from trajectory_score import (g5_collect_extractions, g5_collect_scores,
                                  g5_get_method_scores, g5_node_index, score_argmax)
    cases = load_cases(ROOT / "data" / "g5_state" / "cases.jsonl")
    index = g5_node_index(cases)
    rows = read_jsonl(ROOT / "runs" / "G5-DECOMPOSE-a01" / "predictions.jsonl")
    ext = g5_collect_extractions(rows)
    qs = g5_collect_scores(rows, "qwen")
    ags = g5_collect_scores(rows, "agentrm")
    out: dict[str, Any] = {"status": "COMPLETED", "source": {
        "cases": str(ROOT / "data/g5_state/cases.jsonl"),
        "predictions": str(ROOT / "runs/G5-DECOMPOSE-a01/predictions.jsonl"),
        "note": "read-only; G5 run.json and predictions are not modified",
    }}
    for kind in ("primary", "secondary"):
        table = {}
        for mechanism in ("versioned_report", "scoped_authorization", "exactly_once"):
            total = correct = 0
            confusion = Counter()
            for key, node in index.items():
                if node["kind"] != kind or node["family"]["split"] != "audit":
                    continue
                if node["family"]["mechanism"] != mechanism:
                    continue
                record = ext.get(key)
                predicted = record["state_used"] if record else "missing"
                truth = node["state"]
                total += 1
                correct += predicted == truth
                confusion[f"{truth}->{predicted}"] += 1
            table[mechanism] = {"correct": correct, "total": total,
                                "accuracy": correct / total, "confusion": dict(confusion)}
        out[kind] = table
    unknown = Counter()
    unknown_total = 0
    for key, node in index.items():
        if (node["kind"] == "primary" and node["family"]["split"] == "audit"
                and node["family"]["mechanism"] == "versioned_report" and node["state"] == "unknown"):
            record = ext.get(key)
            unknown[record["state_used"] if record else "missing"] += 1
            unknown_total += 1
    out["versioned_unknown_primary"] = {"total": unknown_total, "predicted": dict(unknown)}
    loss = Counter()
    total_regret = 0.0
    for key, node in index.items():
        if node["kind"] != "primary" or node["family"]["split"] != "audit":
            continue
        scores, _ = g5_get_method_scores("E_CPU", key, node, ext.get(key), qs, ags)
        argmax = score_argmax(scores)
        best = max(node["q"].values())
        regret = max(0.0, best - statistics.fmean(node["q"][cid] for cid in argmax))
        total_regret += regret
        if regret > 0:
            record = ext.get(key)
            label = f"{node['family']['mechanism']}:{node['state']}->{record['state_used'] if record else 'missing'}"
            loss[label] += regret
    out["e_cpu_primary_regret"] = {"sum": total_regret, "decomposition": dict(loss)}
    reversal = {}
    for stage in ("current", "stale"):
        total = correct = 0
        for key, node in index.items():
            if (node["kind"] != "primary" or node["family"]["split"] != "audit"
                    or node["family"]["mechanism"] != "versioned_report"
                    or node["family"]["pair_type"] != "reversal"):
                continue
            if not node["variant"]["variant_id"].endswith(stage):
                continue
            total += 1
            correct += ext[key]["state_used"] == node["state"]
        reversal[stage] = {"correct": correct, "total": total,
                           "accuracy": correct / total if total else None}
    out["reversal"] = reversal
    return out


# ---------------------------------------------------------------------------
# Native ToolSandbox CPU worker.
# ---------------------------------------------------------------------------

def _native_imports():
    if str(TOOL_SANDBOX) not in sys.path:
        sys.path.insert(0, str(TOOL_SANDBOX))
    from tool_sandbox.common.execution_context import (  # type: ignore
        DatabaseNamespace, ExecutionContext, new_context,
    )
    from tool_sandbox.tools.contact import search_contacts, modify_contact  # type: ignore
    from tool_sandbox.tools.messaging import send_message_with_phone_number, search_messages  # type: ignore
    from tool_sandbox.tools.reminder import search_reminder, modify_reminder  # type: ignore
    return {
        "DatabaseNamespace": DatabaseNamespace, "ExecutionContext": ExecutionContext,
        "new_context": new_context, "search_contacts": search_contacts,
        "modify_contact": modify_contact, "send_message": send_message_with_phone_number,
        "search_messages": search_messages, "search_reminder": search_reminder,
        "modify_reminder": modify_reminder,
    }


def _initial_context(native, spec: dict[str, Any]):
    ns = native["DatabaseNamespace"]
    ctx = native["ExecutionContext"]()
    if spec["workflow"] == "contact":
        ctx.add_to_database(ns.CONTACT, rows=[
            {"person_id": spec["self_id"], "name": "Self", "phone_number": spec["self_phone"],
             "relationship": "self", "is_self": True},
            {"person_id": spec["target_id"], "name": spec["target_name"],
             "phone_number": spec["target_initial"], "relationship": spec["relationship_initial"],
             "is_self": False},
            {"person_id": spec["other_id"], "name": spec["other_name"],
             "phone_number": spec["other_initial"], "relationship": "coworker", "is_self": False},
        ])
    else:
        ctx.add_to_database(ns.REMINDER, rows=[
            {"reminder_id": spec["target_id"], "content": spec["content_initial"],
             "creation_timestamp": float(spec["creation_timestamp"]),
             "reminder_timestamp": float(spec["target_initial"]),
             "latitude": None, "longitude": None},
            {"reminder_id": spec["other_id"], "content": "Other reminder",
             "creation_timestamp": float(spec["creation_timestamp"]) + 10.0,
             "reminder_timestamp": float(spec["other_initial"]),
             "latitude": None, "longitude": None},
        ])
    return ctx


def _serialize_context(ctx) -> dict[str, Any]:
    return json.loads(json.dumps(ctx.to_dict(serialize_console=False), ensure_ascii=False))


def _restore_context(native, snapshot: dict[str, Any]):
    return native["ExecutionContext"].from_dict(json.loads(json.dumps(snapshot)))


def _query(native, workflow: str, target_id: str) -> list[dict[str, Any]]:
    if workflow == "contact":
        return native["search_contacts"](person_id=target_id)
    return native["search_reminder"](reminder_id=target_id)


def _apply_event(native, workflow: str, event_type: str, spec: dict[str, Any], world_index: int):
    if workflow == "contact":
        if event_type in {"U", "F"}:
            native["modify_contact"](person_id=spec["target_id"],
                                     phone_number=spec["new_values"][world_index])
        elif event_type == "S":
            native["modify_contact"](person_id=spec["target_id"],
                                     relationship=spec["relationship_new"])
        elif event_type == "O":
            native["modify_contact"](person_id=spec["other_id"],
                                     phone_number=spec["other_new"])
    else:
        if event_type in {"U", "F"}:
            native["modify_reminder"](reminder_id=spec["target_id"],
                                      reminder_timestamp=float(spec["new_values"][world_index]))
        elif event_type == "S":
            native["modify_reminder"](reminder_id=spec["target_id"],
                                      content=spec["content_new"])
        elif event_type == "O":
            native["modify_reminder"](reminder_id=spec["other_id"],
                                      reminder_timestamp=float(spec["other_new"]))


def native_build_family(spec: dict[str, Any]) -> dict[str, Any]:
    native = _native_imports()
    ctx = _initial_context(native, spec)
    initial_snapshot = _serialize_context(ctx)
    with native["new_context"](ctx):
        initial_query = _query(native, spec["workflow"], spec["target_id"])
    branches = []
    event_type = spec["event_type"]
    for world_index in range(len(spec["new_values"]) if event_type in {"U", "F"} else 1):
        branch_ctx = _restore_context(native, initial_snapshot)
        with native["new_context"](branch_ctx):
            _apply_event(native, spec["workflow"], event_type, spec, world_index)
            snapshot = _serialize_context(branch_ctx)
            query_result = _query(native, spec["workflow"], spec["target_id"])
        branches.append({
            "world_index": world_index,
            "snapshot": snapshot,
            "query_result": query_result,
        })
    return {"initial_query": initial_query, "branches": branches}


def native_execute(spec: dict[str, Any]) -> dict[str, Any]:
    native = _native_imports()
    workflow = spec["workflow"]
    snapshot = spec["snapshot"]
    value = spec["value"]
    ctx = _restore_context(native, snapshot)
    try:
        with native["new_context"](ctx):
            if workflow == "contact":
                message_id = native["send_message"](
                    phone_number=str(value), content=spec["content"])
                result = native["search_messages"](message_id=message_id)
            else:
                numeric = float(value) + 1800.0
                native["modify_reminder"](reminder_id=spec["target_id"],
                                          reminder_timestamp=numeric)
                result = native["search_reminder"](reminder_id=spec["target_id"])
        return {"ok": True, "result": result, "snapshot": _serialize_context(ctx)}
    except Exception as error:  # native tool rejection is a recorded tool error
        return {"ok": False, "error_type": type(error).__name__, "error": str(error)}


def native_query(spec: dict[str, Any]) -> dict[str, Any]:
    native = _native_imports()
    ctx = _restore_context(native, spec["snapshot"])
    try:
        with native["new_context"](ctx):
            return {"ok": True, "result": _query(native, spec["workflow"], spec["target_id"])}
    except Exception as error:
        return {"ok": False, "error_type": type(error).__name__, "error": str(error)}


def cpu_worker() -> None:
    # Keep stdout exclusively for the JSONL protocol.
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "build_family":
                response = native_build_family(request["spec"])
            elif op == "execute":
                response = native_execute(request["spec"])
            elif op == "query":
                response = native_query(request["spec"])
            elif op == "ping":
                _native_imports()
                response = {"ok": True}
            else:
                response = {"ok": False, "error_type": "UnknownOp", "error": str(op)}
            response = {"ok": True, "result": response}
        except Exception as error:
            response = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()


class ToolWorker:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(CPU_PYTHON), str(Path(__file__).resolve()), "--cpu-worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.calls = 0

    def call(self, payload: dict[str, Any]) -> Any:
        self.calls += 1
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ToolSandbox CPU worker exited without a response")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"CPU worker error: {response}")
        return response["result"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

# ---------------------------------------------------------------------------
# Fixed G6 data design.
# ---------------------------------------------------------------------------

N_TEMPLATES: dict[str, dict[str, dict[str, dict[str, str]]]] = {
    "contact": {
        "dev": {
            "U": {
                "a": "The contact number for {target_id} was revised. The replacement number is not included in this notice.",
                "b": "{target_id} now has updated phone details; this notice omits the replacement number.",
            },
            "F": {
                "a": "The current phone number for {target_id} is now {new_value}.",
                "b": "Reach {target_id} at {new_value}; that is the number currently in the contact record.",
            },
            "S": {
                "a": "Only the relationship label for {target_id} was edited to {new_relationship}.",
                "b": "The relationship entry attached to {target_id} was updated. No phone-field edit is reported.",
            },
            "O": {
                "a": "The contact number for {other_id} was revised. This notice concerns {other_id}.",
                "b": "Updated phone details were recorded for {other_id}; the replacement number is not shown here.",
            },
        },
        "audit": {
            "U": {
                "a": "A new number has been registered for {target_id}. This notice does not show its digits.",
                "b": "The dialling details on {target_id}'s record have changed; the revised entry is not reproduced here.",
            },
            "F": {
                "a": "{target_id}'s revised dialling number is {new_value}.",
                "b": "The replacement number now listed under {target_id} is {new_value}.",
            },
            "S": {
                "a": "The relationship description on {target_id}'s record has changed to {new_relationship}; this edit concerns that description only.",
                "b": "{target_id} has a revised relationship label. The edit is confined to that label.",
            },
            "O": {
                "a": "A new number has been registered for {other_id}; its digits are omitted from this notice.",
                "b": "The dialling entry that changed belongs to {other_id}, and its revised value is not included here.",
            },
        },
    },
    "reminder": {
        "dev": {
            "U": {
                "a": "The scheduled time for {target_id} was changed. The new time is not included in this notice.",
                "b": "{target_id} was rescheduled; its current timestamp is not stated in this notice.",
            },
            "F": {
                "a": "The current reminder timestamp for {target_id} is now {new_value}.",
                "b": "{target_id} is now scheduled at Unix timestamp {new_value}.",
            },
            "S": {
                "a": "Only the content text of {target_id} was edited to {new_content}.",
                "b": "The wording stored in {target_id} was updated. No scheduled-time edit is reported.",
            },
            "O": {
                "a": "The scheduled time for {other_id} was changed; its new timestamp is not shown.",
                "b": "{other_id} was rescheduled. This notice concerns {other_id}.",
            },
        },
        "audit": {
            "U": {
                "a": "{target_id} has a revised alert time, which is not displayed in this notice.",
                "b": "The time at which {target_id} should trigger has moved; the replacement time is omitted.",
            },
            "F": {
                "a": "The revised alert time of {target_id} is Unix timestamp {new_value}.",
                "b": "{target_id} should now trigger at Unix timestamp {new_value}.",
            },
            "S": {
                "a": "The description attached to {target_id} was rewritten as {new_content}; that is the only field edited.",
                "b": "An edit changed the text of {target_id}, without changing its alert schedule.",
            },
            "O": {
                "a": "The reminder with revised alert timing is {other_id}; the new time is not displayed.",
                "b": "The trigger time that moved belongs to {other_id}, and its new value is omitted.",
            },
        },
    },
}


def _phone(n: int) -> str:
    return f"+1415555{n:04d}"


def _public_task(workflow: str, target_id: str) -> str:
    if workflow == "contact":
        return (f'Send the fixed message "{MESSAGE_CONTENT}" to the contact with person_id '
                f'{target_id}, using that contact\'s current valid phone number.')
    return (f"Move the reminder with reminder_id {target_id} to 1800 seconds after "
            "its current reminder timestamp.")


def _notice(case: dict[str, Any], condition: str, representation: str,
            world: dict[str, Any] | None = None) -> str:
    workflow = case["workflow"]
    if representation == "J":
        if workflow == "contact":
            payload = {"object_type": "contact", "object_id": (
                case["other_id"] if condition == "O" else case["target_id"]),
                "changed_fields": ["relationship" if condition == "S" else "phone_number"]}
            if condition == "F":
                payload["new_values"] = {"phone_number": case["conditions"][condition]["new_value"]}
            elif condition == "S":
                payload["new_values"] = {"relationship": case["relationship_new"]}
        else:
            payload = {"object_type": "reminder", "object_id": (
                case["other_id"] if condition == "O" else case["target_id"]),
                "changed_fields": ["content" if condition == "S" else "reminder_timestamp"]}
            if condition == "F":
                payload["new_values"] = {"reminder_timestamp": case["conditions"][condition]["new_value"]}
            elif condition == "S":
                payload["new_values"] = {"content": case["content_new"]}
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    values = {
        "target_id": case["target_id"], "other_id": case["other_id"],
        "new_value": case["conditions"][condition].get("new_value", ""),
        "new_relationship": case.get("relationship_new", ""),
        "new_content": case.get("content_new", ""),
    }
    template = N_TEMPLATES[workflow][case["split"]][condition][case["template_variant"]]
    return template.format(**values)


def make_specs() -> list[dict[str, Any]]:
    data_rng = random.Random(DATA_SEED)
    split_rng = random.Random(SPLIT_SEED)
    specs: list[dict[str, Any]] = []
    family_index = 0
    for workflow in WORKFLOWS:
        labels = ["dev"] * 6 + ["audit"] * 24
        split_rng.shuffle(labels)
        ordinal = {"dev": 0, "audit": 0}
        for workflow_index in range(30):
            split = labels[workflow_index]
            variant = "ab"[ordinal[split] % 2]
            ordinal[split] += 1
            fid = f"G6F{family_index:04d}"
            family_index += 1
            if workflow == "contact":
                numbers = data_rng.sample(range(1000, 9999), 6)
                target_id = f"contact-{family_index:03d}-target"
                other_id = f"contact-{family_index:03d}-other"
                spec = {
                    "workflow": "contact", "family_id": fid, "split": split,
                    "template_variant": variant, "target_id": target_id, "other_id": other_id,
                    "self_id": f"self-{family_index:03d}", "self_phone": _phone(numbers[0]),
                    "target_name": f"Target {family_index:03d}", "other_name": f"Other {family_index:03d}",
                    "target_initial": _phone(numbers[1]), "other_initial": _phone(numbers[2]),
                    "relationship_initial": data_rng.choice(["friend", "family", "coworker"]),
                    "relationship_new": data_rng.choice(["manager", "neighbor", "client"]),
                    "content_initial": "unused", "content_new": "unused",
                    "creation_timestamp": 1789000000.0 + family_index * 1000,
                    "new_values": [_phone(numbers[3]), _phone(numbers[4])],
                    "other_new": _phone(numbers[5]),
                }
                spec["conditions"] = {
                    "U": {"new_values": spec["new_values"][:2]},
                    "F": {"new_values": [spec["new_values"][0]]},
                    "S": {}, "O": {},
                }
            else:
                base = 1790000000 + family_index * 100000 + data_rng.randrange(0, 50000)
                target_id = f"reminder-{family_index:03d}-target"
                other_id = f"reminder-{family_index:03d}-other"
                new_values = [base + data_rng.randrange(600, 5000) for _ in range(2)]
                other_initial = base + 50000
                other_new = base + 90000
                spec = {
                    "workflow": "reminder", "family_id": fid, "split": split,
                    "template_variant": variant, "target_id": target_id, "other_id": other_id,
                    "self_id": "", "self_phone": "",
                    "target_name": "", "other_name": "",
                    "target_initial": base, "other_initial": other_initial,
                    "relationship_initial": "", "relationship_new": "",
                    "content_initial": f"Reminder {family_index:03d}",
                    "content_new": f"Updated reminder {family_index:03d}",
                    "creation_timestamp": 1789000000.0 + family_index * 1000,
                    "new_values": new_values, "other_new": other_new,
                }
                spec["conditions"] = {
                    "U": {"new_values": new_values},
                    "F": {"new_values": [new_values[0]]},
                    "S": {}, "O": {},
                }
            specs.append(spec)
    return specs


def _initial_text(workflow: str, target_id: str, query_result: list[dict[str, Any]]) -> str:
    action = (f"search_contacts(person_id={target_id})" if workflow == "contact"
              else f"search_reminder(reminder_id={target_id})")
    return "[e1]\nAction: " + action + "\nObservation: " + json.dumps(
        query_result, sort_keys=True, ensure_ascii=False)


def _event_text(notice: str) -> str:
    return "[e2]\nObservation:\n" + notice


def _query_text(workflow: str, target_id: str, query_result: list[dict[str, Any]]) -> str:
    action = (f"search_contacts(person_id={target_id})" if workflow == "contact"
              else f"search_reminder(reminder_id={target_id})")
    return "[e3]\nAction: " + action + "\nObservation: " + json.dumps(
        query_result, sort_keys=True, ensure_ascii=False)


def generate_data(worker: ToolWorker) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path, manifest_path = DATA_DIR / "cases.jsonl", DATA_DIR / "manifest.json"
    if cases_path.exists() or manifest_path.exists():
        raise FileExistsError("G6 data already exist; refusing to regenerate")
    specs = make_specs()
    cases = []
    for spec in specs:
        base = {key: value for key, value in spec.items() if key != "conditions"}
        condition_data = {}
        initial_query = None
        for condition in CONDITIONS:
            branch_spec = dict(base)
            branch_spec["event_type"] = condition
            new_values = spec["conditions"][condition].get("new_values", [])
            branch_spec["new_values"] = new_values
            result = worker.call({"op": "build_family", "spec": branch_spec})
            if initial_query is None:
                initial_query = result["initial_query"]
            branches = []
            for branch in result["branches"]:
                world_index = branch["world_index"]
                if condition in {"U", "F"}:
                    current = new_values[world_index]
                else:
                    current = spec["target_initial"]
                branches.append({
                    "world_key": f"w{world_index}", "weight": 0.5 if condition == "U" else 1.0,
                    "snapshot": branch["snapshot"], "query_result": branch["query_result"],
                    "current_value": current,
                    "expected_operation_value": (current if spec["workflow"] == "contact"
                                                 else int(current) + 1800),
                })
            condition_data[condition] = {
                "relation_root": {"U": "INVALIDATE", "F": "REPLACE", "S": "KEEP", "O": "KEEP"}[condition],
                "relation_after": {"U": "REPLACE", "F": "REPLACE", "S": "KEEP", "O": "KEEP"}[condition],
                "root_knowledge": "unknown" if condition == "U" else "known",
                "root_value": None if condition == "U" else (
                    new_values[0] if condition == "F" else spec["target_initial"]),
                "worlds": branches,
            }
            if condition == "F":
                condition_data[condition]["new_value"] = new_values[0]
            elif condition == "U":
                condition_data[condition]["world_values"] = new_values
        case = {
            **base,
            "public_task": _public_task(spec["workflow"], spec["target_id"]),
            "field": FIELD_BY_WORKFLOW[spec["workflow"]],
            "initial_value": spec["target_initial"],
            "initial_query": initial_query,
            "conditions": condition_data,
        }
        cases.append(case)
    with cases_path.open("x", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "dataset": "G6 validity-aware native ToolSandbox workflows",
        "dataset_version": DATASET_VERSION, "data_seed": DATA_SEED,
        "split_seed": SPLIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "tool_sandbox_revision": TOOL_REVISION,
        "tool_sandbox_path": str(TOOL_SANDBOX),
        "tool_sandbox_bytes": sum(p.stat().st_size for p in TOOL_SANDBOX.rglob("*") if p.is_file()),
        "cpu_environment": str(CPU_PYTHON),
        "family_counts": dict(Counter(case["split"] for case in cases)),
        "workflow_counts": dict(Counter(case["workflow"] for case in cases)),
        "condition_counts_per_family": {condition: len(cases) for condition in CONDITIONS},
        "node_plan": {"roots": 60 * 4 * 2, "after_query_unique_worlds": 60 * 5 * 2,
                      "total_public_nodes": 60 * (4 * 2 + 5 * 2),
                      "dev_nodes": 12 * (4 * 2 + 5 * 2),
                      "audit_nodes": 48 * (4 * 2 + 5 * 2)},
        "strategy_trajectory_plan": 48 * 10 * 5,
        "prompt_version": PROMPT_VERSION,
        "fixed_message_content": MESSAGE_CONTENT,
        "frozen": True,
        "created_at": now_cst(),
    }
    manifest["cases_sha256"] = sha256_file(cases_path)
    manifest["manifest_self_note"] = "sha256 of cases is authoritative; this file is frozen"
    write_json(manifest_path, manifest)
    return manifest


def case_nodes(case: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    initial_text = _initial_text(case["workflow"], case["target_id"], case["initial_query"])
    for condition in CONDITIONS:
        cd = case["conditions"][condition]
        for representation in REPRESENTATIONS:
            notice = _notice(case, condition, representation)
            history = _event_text(notice)
            node_id = f"{case['family_id']}:{condition}:{representation}:root"
            nodes.append({
                "node_id": node_id, "family_id": case["family_id"], "workflow": case["workflow"],
                "split": case["split"], "condition": condition, "representation": representation,
                "stage": "root", "world_key": None, "world_weight": 1.0,
                "task": case["public_task"], "target_id": case["target_id"], "field": case["field"],
                "initial_text": initial_text, "history": history, "notice": notice,
                "truth_state": "unknown" if condition == "U" else "known",
                "truth_value": cd["root_value"], "truth_relation": cd["relation_root"],
                "public_hash": object_hash({"task": case["public_task"], "target": case["target_id"],
                                            "field": case["field"], "initial": initial_text,
                                            "history": history, "stage": "root"}),
            })
            for world in cd["worlds"]:
                qtext = _query_text(case["workflow"], case["target_id"], world["query_result"])
                full_history = history + "\n" + qtext
                after_id = f"{case['family_id']}:{condition}:{representation}:after:{world['world_key']}"
                nodes.append({
                    "node_id": after_id, "family_id": case["family_id"], "workflow": case["workflow"],
                    "split": case["split"], "condition": condition, "representation": representation,
                    "stage": "after", "world_key": world["world_key"], "world_weight": world["weight"],
                    "task": case["public_task"], "target_id": case["target_id"], "field": case["field"],
                    "initial_text": initial_text, "history": full_history, "notice": notice,
                    "truth_state": "known", "truth_value": world["current_value"],
                    "truth_relation": cd["relation_after"],
                    "public_hash": object_hash({"task": case["public_task"], "target": case["target_id"],
                                                "field": case["field"], "initial": initial_text,
                                                "history": full_history, "stage": "after"}),
                })
    return nodes


def load_cases() -> list[dict[str, Any]]:
    return read_jsonl(DATA_DIR / "cases.jsonl")


def all_nodes(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for case in cases:
        nodes.extend(case_nodes(case))
    return nodes


def g5_stage() -> None:
    run = initialize_run()
    analysis = g5_reanalysis()
    run["g5_reanalysis"] = analysis
    run["phase_evidence"].setdefault("G6-CPU-REANALYSIS", {}).update({
        "completed_at": now_cst(), "model_calls": 0, "gpu_hours": 0.0,
        "source": analysis["source"],
    })
    mark_phase(run, "G6-CPU-REANALYSIS", "COMPLETED")
    print(json.dumps({"G6-CPU-REANALYSIS": "COMPLETED",
                      "primary": analysis["primary"], "unknown": analysis["versioned_unknown_primary"],
                      "regret": analysis["e_cpu_primary_regret"], "reversal": analysis["reversal"]},
                     indent=2, ensure_ascii=False))


def data_stage() -> None:
    run = initialize_run()
    worker = ToolWorker()
    try:
        if (DATA_DIR / "cases.jsonl").exists() and (DATA_DIR / "manifest.json").exists():
            manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        else:
            manifest = generate_data(worker)
    finally:
        worker.close()
    cases = load_cases()
    nodes = all_nodes(cases)
    if len(cases) != 60 or len(nodes) != 1080:
        raise RuntimeError(f"G6 data count mismatch: cases={len(cases)} nodes={len(nodes)}")
    run["data"] = {
        "cases": str(DATA_DIR / "cases.jsonl"), "manifest": str(DATA_DIR / "manifest.json"),
        "cases_sha256": sha256_file(DATA_DIR / "cases.jsonl"),
        "manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "family_counts": manifest["family_counts"], "workflow_counts": manifest["workflow_counts"],
        "node_plan": manifest["node_plan"], "tool_sandbox_revision": TOOL_REVISION,
        "cpu_environment": str(CPU_PYTHON), "tool_worker_calls": worker.calls,
    }
    run["phase_evidence"].setdefault("G6-TOOLS-DATA", {}).update({
        "completed_at": now_cst(), "nodes": len(nodes), "tool_worker_calls": worker.calls,
    })
    mark_phase(run, "G6-TOOLS-DATA", "COMPLETED")
    print(json.dumps({"G6-TOOLS-DATA": "COMPLETED", "cases": len(cases), "nodes": len(nodes),
                      "manifest": manifest["node_plan"], "tool_worker_calls": worker.calls}, indent=2))

# ---------------------------------------------------------------------------
# G6 model prompt construction, parsing and offline cache.
# ---------------------------------------------------------------------------

E_USER = """Public task:
{task}

Target object:
{target_id}

Task-relevant field:
{field}

Public execution convention:
The object is identified by its stable ID.
A contact task uses the target's current phone number.
A reminder task adds 1800 seconds to the target's current reminder timestamp.
Only information visible below is available to you.

Initial cached observation:
{initial_text}

Visible history, in order:
{history}

Extract the current value of the task-relevant field."""

REVIEW_USER = """Public task:
{task}

Target object and required field:
{target_id}
{field}

Initial cached observation:
{initial_text}

Complete visible history, in order:
{history}

Draft extraction:
{draft}

Review the draft and return the required final JSON."""


def e_messages(node: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": E_SYSTEM},
            {"role": "user", "content": E_USER.format(
                task=node["task"], target_id=node["target_id"], field=node["field"],
                initial_text=node["initial_text"], history=node["history"])}]


def review_messages(node: dict[str, Any], draft: str, system: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": REVIEW_USER.format(
                task=node["task"], target_id=node["target_id"], field=node["field"],
                initial_text=node["initial_text"], history=node["history"], draft=draft)}]


def extract_json(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalize_value(workflow: str, value: Any) -> tuple[Any, bool]:
    if value is None:
        return None, False
    if workflow == "contact":
        if not isinstance(value, str) or not value.strip():
            return None, False
        return value.strip(), True
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None, False
    return int(value), True


def parse_extract(text: str, workflow: str, valid_ids: set[str]) -> dict[str, Any]:
    payload = extract_json(text)
    if payload is None:
        return {"valid": False, "error": "invalid_json", "knowledge": "unknown", "value": None,
                "evidence": [], "evidence_invalid": False, "evidence_empty": True}
    knowledge = payload.get("knowledge")
    evidence = payload.get("evidence")
    if knowledge not in {"known", "unknown"}:
        return {"valid": False, "error": "invalid_knowledge", "knowledge": "unknown", "value": None,
                "evidence": [], "evidence_invalid": False, "evidence_empty": True}
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        return {"valid": False, "error": "invalid_evidence_shape", "knowledge": "unknown", "value": None,
                "evidence": [], "evidence_invalid": True, "evidence_empty": True}
    value, value_ok = normalize_value(workflow, payload.get("value"))
    if knowledge == "known" and not value_ok:
        return {"valid": False, "error": "invalid_known_value", "knowledge": "unknown", "value": None,
                "evidence": evidence, "evidence_invalid": any(x not in valid_ids for x in evidence),
                "evidence_empty": not evidence}
    if knowledge == "unknown" and payload.get("value") is not None:
        return {"valid": False, "error": "unknown_with_value", "knowledge": "unknown", "value": None,
                "evidence": evidence, "evidence_invalid": any(x not in valid_ids for x in evidence),
                "evidence_empty": not evidence}
    return {"valid": True, "error": None, "knowledge": knowledge,
            "value": value if knowledge == "known" else None, "evidence": evidence,
            "evidence_invalid": any(x not in valid_ids for x in evidence),
            "evidence_empty": not evidence}


def parse_review(text: str, workflow: str, valid_ids: set[str]) -> dict[str, Any]:
    payload = extract_json(text)
    if payload is None:
        return {"valid": False, "error": "invalid_json", "relation": "UNRESOLVED",
                "knowledge": "unknown", "value": None, "evidence": [],
                "evidence_invalid": False, "evidence_empty": True}
    relation = payload.get("relation")
    if relation not in RELATION_VALUES:
        return {"valid": False, "error": "invalid_relation", "relation": "UNRESOLVED",
                "knowledge": "unknown", "value": None, "evidence": [],
                "evidence_invalid": False, "evidence_empty": True}
    knowledge = payload.get("knowledge")
    if knowledge not in {"known", "unknown"}:
        return {"valid": False, "error": "invalid_knowledge", "relation": "UNRESOLVED",
                "knowledge": "unknown", "value": None, "evidence": [],
                "evidence_invalid": False, "evidence_empty": True}
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        return {"valid": False, "error": "invalid_evidence_shape", "relation": "UNRESOLVED",
                "knowledge": "unknown", "value": None, "evidence": [],
                "evidence_invalid": True, "evidence_empty": True}
    value, value_ok = normalize_value(workflow, payload.get("value"))
    if knowledge == "known" and not value_ok:
        return {"valid": False, "error": "invalid_known_value", "relation": "UNRESOLVED",
                "knowledge": "unknown", "value": None, "evidence": evidence,
                "evidence_invalid": any(x not in valid_ids for x in evidence),
                "evidence_empty": not evidence}
    if knowledge == "unknown" and payload.get("value") is not None:
        return {"valid": False, "error": "unknown_with_value", "relation": "UNRESOLVED",
                "knowledge": "unknown", "value": None, "evidence": evidence,
                "evidence_invalid": any(x not in valid_ids for x in evidence),
                "evidence_empty": not evidence}
    return {"valid": True, "error": None, "relation": relation, "knowledge": knowledge,
            "value": value if knowledge == "known" else None, "evidence": evidence,
            "evidence_invalid": any(x not in valid_ids for x in evidence),
            "evidence_empty": not evidence}


def canonical_draft(parsed: dict[str, Any]) -> str:
    return json.dumps({"knowledge": parsed["knowledge"], "value": parsed["value"],
                       "evidence": parsed.get("evidence", [])}, sort_keys=True, ensure_ascii=False)


class QwenModel:
    def __init__(self) -> None:
        from types import SimpleNamespace
        from trajectory_score import QwenScorer, QWEN_SNAPSHOT as SCORER_QWEN
        self.scorer = QwenScorer(SimpleNamespace(), str(SCORER_QWEN), "cuda:0")
        self.tokenizer = self.scorer.tokenizer
        self.revision = getattr(self.scorer.model.config, "_commit_hash", None)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def call(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.calls >= MODEL_CALL_LIMIT:
            raise RuntimeError("G6 model call hard limit reached")
        raw, prompt_tokens, error = self.scorer._generate(messages, max_new_tokens=128)
        self.calls += 1
        out_tokens = len(self.tokenizer(raw, add_special_tokens=False)["input_ids"])
        self.input_tokens += prompt_tokens
        self.output_tokens += out_tokens
        return {"raw": raw, "input_tokens": prompt_tokens, "output_tokens": out_tokens,
                "error": error, "model_call_index": self.calls}


def _existing_offline() -> dict[tuple[str, str, str], dict[str, Any]]:
    path = RUN_DIR / "predictions.jsonl"
    if not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        if row.get("record_type") in {"extract", "review_generic", "review_validity"}:
            out[(row["phase"], row["node_id"], row["record_type"])] = row
    return out


def _append_prediction(row: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run_offline_phase(phase: str) -> None:
    run = initialize_run()
    cases = load_cases()
    nodes = [node for node in all_nodes(cases) if node["split"] == phase]
    expected = 216 if phase == "dev" else 864
    if len(nodes) != expected:
        raise RuntimeError(f"{phase}: expected {expected} nodes, got {len(nodes)}")
    model = QwenModel()
    existing = _existing_offline()
    run["model"] = {"snapshot": str(QWEN_SNAPSHOT), "revision": model.revision,
                    "dtype": "bfloat16", "do_sample": False, "max_new_tokens": 128,
                    "use_model_defaults": False}
    mark_phase(run, f"G6-{phase.upper()}", "RUNNING",
               started_at=now_cst(), host="local", pid=os.getpid(), gpus=1,
               nodes_total=len(nodes), model_revision=model.revision)
    started = time.perf_counter()
    invalid = Counter()
    calls_before = len(existing)
    for index, node in enumerate(nodes, start=1):
        valid_ids = {"e1", "e2"}
        if node["stage"] == "after":
            valid_ids.add("e3")
        e_key = (phase, node["node_id"], "extract")
        if e_key in existing:
            e_row = existing[e_key]
        else:
            response = model.call(e_messages(node))
            parsed = parse_extract(response["raw"], node["workflow"], valid_ids)
            e_row = {
                "record_type": "extract", "phase": phase, "node_id": node["node_id"],
                "family_id": node["family_id"], "workflow": node["workflow"],
                "condition": node["condition"], "representation": node["representation"],
                "stage": node["stage"], "world_key": node["world_key"],
                "public_hash": node["public_hash"], "prompt_version": PROMPT_VERSION,
                "parsed": parsed, "invalid": not parsed["valid"],
                "raw_output": response["raw"], "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
            }
            _append_prediction(e_row)
            existing[e_key] = e_row
            if not parsed["valid"]:
                invalid["extract"] += 1
        draft = "INVALID_E_DRAFT" if e_row.get("invalid") else canonical_draft(e_row["parsed"])
        for method, record_type, system in (
                ("GENERIC_REVIEW", "review_generic", R_SYSTEM),
                ("VALIDITY_REVIEW", "review_validity", V_SYSTEM)):
            key = (phase, node["node_id"], record_type)
            if key in existing:
                continue
            response = model.call(review_messages(node, draft, system))
            parsed = parse_review(response["raw"], node["workflow"], valid_ids)
            review_row = {
                "record_type": record_type, "phase": phase, "node_id": node["node_id"],
                "family_id": node["family_id"], "workflow": node["workflow"],
                "condition": node["condition"], "representation": node["representation"],
                "stage": node["stage"], "world_key": node["world_key"],
                "public_hash": node["public_hash"], "method": method,
                "prompt_version": PROMPT_VERSION, "draft": draft,
                "parsed": parsed, "invalid": not parsed["valid"],
                "raw_output": response["raw"], "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
            }
            _append_prediction(review_row)
            existing[key] = review_row
            if not parsed["valid"]:
                invalid[record_type] += 1
        if index % 25 == 0:
            print(json.dumps({"phase": phase, "nodes": index, "calls": model.calls,
                              "invalid": dict(invalid)}), flush=True)
    elapsed = time.perf_counter() - started
    all_rows = read_jsonl(RUN_DIR / "predictions.jsonl") if (RUN_DIR / "predictions.jsonl").exists() else []
    model_rows = [row for row in all_rows
                  if row.get("record_type") in {"extract", "review_generic", "review_validity"}]
    run["budget"]["model_calls"] = len(model_rows)
    run["budget"]["input_tokens"] = sum(int(row.get("input_tokens", 0)) for row in model_rows)
    run["budget"]["output_tokens"] = sum(int(row.get("output_tokens", 0)) for row in model_rows)
    run["budget"]["gpu_hours"] = run["budget"].get("gpu_hours", 0.0) + elapsed / 3600.0
    run["phase_evidence"][f"G6-{phase.upper()}"].update({
        "completed_at": now_cst(), "elapsed_seconds": elapsed, "nodes": len(nodes),
        "calls_new": model.calls, "invalid": dict(invalid),
        "calls_total": len(existing),
    })
    mark_phase(run, f"G6-{phase.upper()}", "COMPLETED")
    print(json.dumps({"phase": phase, "completed": True, "nodes": len(nodes),
                      "calls_new": model.calls, "invalid": dict(invalid),
                      "elapsed_seconds": elapsed}, indent=2))

# ---------------------------------------------------------------------------
# Strong deterministic dependency rule and deployment.
# ---------------------------------------------------------------------------

PHONE_RE = re.compile(r"\+?\d[\d\s\-]{6,}\d")
TIMESTAMP_RE = re.compile(r"\b\d{9,11}\b", re.IGNORECASE)


def _query_value(workflow: str, target_id: str, query_result: list[dict[str, Any]]) -> tuple[Any, bool]:
    for row in query_result:
        if workflow == "contact":
            if row.get("person_id") == target_id:
                return row.get("phone_number"), isinstance(row.get("phone_number"), str)
        else:
            if row.get("reminder_id") == target_id:
                value = row.get("reminder_timestamp")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    return int(value), True
    return None, False


def rule_decide(workflow: str, notice: Any, target_id: str, other_id: str,
                initial_value: Any, query_result: list[dict[str, Any]] | None = None
                ) -> dict[str, Any]:
    """Strong public dependency rule.  No family/condition/template lookup."""
    if query_result is not None:
        value, ok = _query_value(workflow, target_id, query_result)
        return {"knowledge": "known" if ok else "unknown", "value": value if ok else None,
                "parsed": ok, "fallback": not ok, "branch": "query"}
    if isinstance(notice, dict):
        if notice.get("object_id") != target_id:
            return {"knowledge": "known", "value": initial_value, "parsed": True,
                    "fallback": False, "branch": "structured-other"}
        changed = set(notice.get("changed_fields", []))
        field = FIELD_BY_WORKFLOW[workflow]
        if field not in changed:
            return {"knowledge": "known", "value": initial_value, "parsed": True,
                    "fallback": False, "branch": "structured-irrelevant"}
        new_values = notice.get("new_values") or {}
        if field in new_values:
            value = new_values[field]
            if workflow == "reminder":
                value = int(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
            if workflow == "contact":
                value = value.strip() if isinstance(value, str) else None
            if value is not None:
                return {"knowledge": "known", "value": value, "parsed": True,
                        "fallback": False, "branch": "structured-new"}
        return {"knowledge": "unknown", "value": None, "parsed": True,
                "fallback": False, "branch": "structured-invalidate"}
    text = str(notice)
    lower = text.lower()
    target_present = target_id in text
    other_present = other_id in text
    if not target_present and other_present:
        return {"knowledge": "known", "value": initial_value, "parsed": True,
                "fallback": False, "branch": "natural-other"}
    if not target_present:
        return {"knowledge": "known", "value": initial_value, "parsed": False,
                "fallback": True, "branch": "natural-unmatched"}
    if workflow == "contact":
        if "relationship" in lower or "label" in lower or "description" in lower:
            return {"knowledge": "known", "value": initial_value, "parsed": True,
                    "fallback": False, "branch": "natural-irrelevant"}
        phones = PHONE_RE.findall(text)
        if phones:
            return {"knowledge": "known", "value": phones[0].strip(), "parsed": True,
                    "fallback": False, "branch": "natural-new"}
        if any(word in lower for word in ("revised", "updated", "changed", "new number",
                                         "registered", "dialling", "phone details", "replacement")):
            return {"knowledge": "unknown", "value": None, "parsed": True,
                    "fallback": False, "branch": "natural-invalidate"}
    else:
        if "content" in lower or "wording" in lower or "description" in lower or "text" in lower:
            return {"knowledge": "known", "value": initial_value, "parsed": True,
                    "fallback": False, "branch": "natural-irrelevant"}
        stamps = TIMESTAMP_RE.findall(text)
        if stamps:
            return {"knowledge": "known", "value": int(stamps[-1]), "parsed": True,
                    "fallback": False, "branch": "natural-new"}
        if any(word in lower for word in ("revised", "rescheduled", "scheduled time",
                                         "alert time", "trigger", "changed", "moved")):
            return {"knowledge": "unknown", "value": None, "parsed": True,
                    "fallback": False, "branch": "natural-invalidate"}
    return {"knowledge": "known", "value": initial_value, "parsed": False,
            "fallback": True, "branch": "natural-unmatched"}


def _prediction_map(rows: list[dict[str, Any]], phase: str, record_type: str) -> dict[str, dict[str, Any]]:
    return {row["node_id"]: row for row in rows
            if row.get("phase") == phase and row.get("record_type") == record_type}


def _node_key(node: dict[str, Any]) -> tuple[str, str, str, str, str | None]:
    return (node["family_id"], node["condition"], node["representation"], node["stage"], node["world_key"])


def _token_sum(records: list[dict[str, Any]]) -> tuple[int, int]:
    return (sum(int(row.get("input_tokens", 0)) for row in records),
            sum(int(row.get("output_tokens", 0)) for row in records))


def _model_final(record: dict[str, Any]) -> dict[str, Any]:
    return record["parsed"]


def execute_strategy(worker: ToolWorker, cases: list[dict[str, Any]], nodes: list[dict[str, Any]],
                     predictions: dict[str, dict[str, dict[str, Any]]], strategy: str,
                     family: dict[str, Any], condition: str, representation: str,
                     world: dict[str, Any]) -> dict[str, Any]:
    workflow = family["workflow"]
    stage_key = (family["family_id"], condition, representation, "root", None)
    root_node = next(node for node in nodes if _node_key(node) == stage_key)
    after_key = (family["family_id"], condition, representation, "after", world["world_key"])
    after_node = next(node for node in nodes if _node_key(node) == after_key)
    notice = _notice(family, condition, representation)
    if representation == "J":
        notice = json.loads(notice)
    snapshots = world["snapshot"]
    world_current = world["current_value"]
    expected_operation = world["expected_operation_value"]
    steps: list[dict[str, Any]] = []
    queries = 0
    model_records: list[dict[str, Any]] = []
    fallback_root = fallback_after = False
    used_methods: list[str] = []

    def model_decision(node: dict[str, Any], method: str) -> dict[str, Any]:
        if method == "DIRECT_E":
            record = predictions["extract"][node["node_id"]]
        elif method == "GENERIC_REVIEW":
            record = predictions["review_generic"][node["node_id"]]
        else:
            record = predictions["review_validity"][node["node_id"]]
        model_records.append(record)
        used_methods.append(method)
        return record["parsed"]

    root_decision = None
    if strategy == "QUERY_ONCE_V":
        queries = 1
    elif strategy == "DEPENDENCY_RULE":
        rule = rule_decide(workflow, notice, family["target_id"], family["other_id"],
                           family["initial_value"], None)
        fallback_root = bool(rule["fallback"])
        root_decision = rule
        if not rule["parsed"] or rule["knowledge"] != "known":
            queries = 1
    elif strategy == "DIRECT_E":
        root_decision = model_decision(root_node, "DIRECT_E")
    elif strategy == "GENERIC_REVIEW":
        model_decision(root_node, "DIRECT_E")
        root_decision = model_decision(root_node, "GENERIC_REVIEW")
    elif strategy == "VALIDITY_REVIEW":
        model_decision(root_node, "DIRECT_E")
        root_decision = model_decision(root_node, "VALIDITY_REVIEW")
    else:
        raise ValueError(strategy)
    steps.append({"step": 0, "stage": "root", "strategy": strategy,
                  "decision": root_decision, "queried": strategy == "QUERY_ONCE_V" or (
                      root_decision is not None and root_decision["knowledge"] != "known")})
    final = root_decision
    if strategy == "QUERY_ONCE_V" or final is None or final["knowledge"] != "known":
        if queries == 0:
            queries = 1
        steps.append({"step": 1, "stage": "query", "query_result": world["query_result"]})
        if strategy == "DEPENDENCY_RULE":
            final = rule_decide(workflow, None, family["target_id"], family["other_id"],
                                family["initial_value"], world["query_result"])
            fallback_after = bool(final["fallback"])
        elif strategy == "DIRECT_E":
            final = model_decision(after_node, "DIRECT_E")
        elif strategy == "GENERIC_REVIEW":
            model_decision(after_node, "DIRECT_E")
            final = model_decision(after_node, "GENERIC_REVIEW")
        elif strategy in {"VALIDITY_REVIEW", "QUERY_ONCE_V"}:
            model_decision(after_node, "DIRECT_E")
            final = model_decision(after_node, "VALIDITY_REVIEW")
        else:
            raise ValueError(strategy)
    knowledge = final.get("knowledge") if final else "unknown"
    value = final.get("value") if final else None
    if strategy == "DEPENDENCY_RULE":
        value_ok = knowledge == "known" and value is not None
    else:
        value_ok = knowledge == "known" and value is not None
    success = wrong = tool_error = abort = False
    native_result = None
    if not value_ok:
        abort = True
        action = "abort"
        steps.append({"step": len(steps), "stage": "action", "action": "abort",
                      "reason": "unknown_or_invalid"})
    else:
        action = "execute"
        response = worker.call({"op": "execute", "spec": {
            "workflow": workflow, "snapshot": snapshots, "target_id": family["target_id"],
            "value": value, "content": MESSAGE_CONTENT,
        }})
        native_result = response
        if not response.get("ok"):
            tool_error = True
            success = wrong = False
            steps.append({"step": len(steps), "stage": "action", "action": "execute",
                          "value": value, "tool_error": response})
        else:
            rows = response.get("result", [])
            if workflow == "contact":
                row = rows[0] if rows else {}
                success = (row.get("recipient_person_id") == family["target_id"]
                           and row.get("recipient_phone_number") == world_current
                           and row.get("content") == MESSAGE_CONTENT)
                wrong = bool(rows) and not success
            else:
                row = rows[0] if rows else {}
                timestamp = row.get("reminder_timestamp")
                success = (row.get("reminder_id") == family["target_id"]
                           and isinstance(timestamp, (int, float))
                           and abs(float(timestamp) - float(expected_operation)) <= 1e-6)
                wrong = bool(rows) and not success
            steps.append({"step": len(steps), "stage": "action", "action": "execute",
                          "value": value, "success": success, "wrong_side_effect": wrong,
                          "native_result": rows})
    utility = (10.0 if success else (-30.0 if wrong else 0.0)) - queries
    input_tokens, output_tokens = _token_sum(model_records)
    return {
        "record_type": "trajectory", "phase": "execution", "family_id": family["family_id"],
        "workflow": workflow, "split": family["split"], "condition": condition,
        "representation": representation, "world_key": world["world_key"],
        "world_weight": world["weight"], "strategy": strategy,
        "steps": steps, "queries": queries, "success": success,
        "wrong_side_effect": wrong, "tool_error": tool_error, "abort": abort,
        "utility": utility, "model_calls": len(model_records), "used_methods": used_methods,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "rule_fallback_root": fallback_root, "rule_fallback_after": fallback_after,
        "native_result": native_result,
    }

# ---------------------------------------------------------------------------
# Execution and analysis.
# ---------------------------------------------------------------------------

def run_execution() -> None:
    run = initialize_run()
    cases = [case for case in load_cases() if case["split"] == "audit"]
    nodes = [node for node in all_nodes(load_cases()) if node["split"] == "audit"]
    rows = read_jsonl(RUN_DIR / "predictions.jsonl")
    predictions = {
        "extract": _prediction_map(rows, "audit", "extract"),
        "review_generic": _prediction_map(rows, "audit", "review_generic"),
        "review_validity": _prediction_map(rows, "audit", "review_validity"),
    }
    missing = []
    for node in nodes:
        for name in predictions:
            if node["node_id"] not in predictions[name]:
                missing.append((node["node_id"], name))
    if missing:
        raise RuntimeError(f"audit offline cache incomplete: {missing[:5]} ({len(missing)} missing)")
    existing = {(row["strategy"], row["family_id"], row["condition"], row["representation"],
                 row["world_key"]) for row in rows if row.get("record_type") == "trajectory"}
    worker = ToolWorker()
    mark_phase(run, "G6-EXECUTE", "RUNNING", started_at=now_cst(), pid=os.getpid(),
               gpus=0, trajectories_total=2400)
    started = time.perf_counter()
    count = 0
    try:
        for family in cases:
            for condition in CONDITIONS:
                cd = family["conditions"][condition]
                for representation in REPRESENTATIONS:
                    for world in cd["worlds"]:
                        for strategy in STRATEGIES:
                            key = (strategy, family["family_id"], condition, representation, world["world_key"])
                            if key in existing:
                                continue
                            record = execute_strategy(worker, cases, nodes, predictions, strategy,
                                                      family, condition, representation, world)
                            record["traj_key"] = "|".join(key)
                            _append_prediction(record)
                            existing.add(key)
                            count += 1
            print(json.dumps({"phase": "execution", "family": family["family_id"],
                              "new_trajectories": count}), flush=True)
    finally:
        worker.close()
    elapsed = time.perf_counter() - started
    run["phase_evidence"]["G6-EXECUTE"].update({
        "completed_at": now_cst(), "elapsed_seconds": elapsed, "new_trajectories": count,
        "tool_worker_calls": worker.calls,
    })
    run["budget"]["tool_worker_calls_execution"] = worker.calls
    mark_phase(run, "G6-EXECUTE", "COMPLETED")
    print(json.dumps({"G6-EXECUTE": "COMPLETED", "new_trajectories": count,
                      "tool_worker_calls": worker.calls, "elapsed_seconds": elapsed}, indent=2))


def _value_matches(workflow: str, left: Any, right: Any) -> bool:
    if workflow == "contact":
        return isinstance(left, str) and isinstance(right, str) and left.strip() == right.strip()
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def _state_correct(workflow: str, parsed: dict[str, Any], node: dict[str, Any]) -> bool:
    if node["truth_state"] == "unknown":
        return parsed.get("knowledge") == "unknown" and parsed.get("value") is None
    return parsed.get("knowledge") == "known" and _value_matches(
        workflow, parsed.get("value"), node["truth_value"])


def _mean_cells(family_values: dict[str, dict[str, float]], field: str) -> float | None:
    cells = defaultdict(list)
    for values in family_values.values():
        if field in values:
            cells[values["cell"]].append(values[field])
    if not cells:
        return None
    return statistics.fmean(statistics.fmean(values) for values in cells.values())


def _stratified_bootstrap(family_values: dict[str, dict[str, float]], field: str,
                          seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    cells = defaultdict(list)
    for values in family_values.values():
        if field in values:
            cells[values["cell"]].append(values[field])
    point = _mean_cells(family_values, field)
    rng = random.Random(seed)
    samples = []
    keys = sorted(cells)
    for _ in range(reps):
        means = []
        for cell in keys:
            values = cells[cell]
            means.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
        samples.append(statistics.fmean(means))
    samples.sort()
    return {"mean": point, "ci95": [
        samples[int(0.025 * reps)], samples[int(0.975 * reps) - 1]
    ], "n_families": sum(len(values) for values in cells.values()),
        "cells": {cell: {"mean": statistics.fmean(values), "n": len(values)}
                  for cell, values in cells.items()}}


def _paired_bootstrap(family_deltas: dict[str, dict[str, float]], field: str,
                      seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    cells = defaultdict(list)
    for values in family_deltas.values():
        cells[values["cell"]].append(values[field])
    point = statistics.fmean(statistics.fmean(v) for v in cells.values()) if cells else None
    rng = random.Random(seed)
    samples = []
    keys = sorted(cells)
    for _ in range(reps):
        means = []
        for cell in keys:
            values = cells[cell]
            means.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
        samples.append(statistics.fmean(means))
    samples.sort()
    return {"delta": point,
            "ci95": [samples[int(0.025 * reps)], samples[int(0.975 * reps) - 1]],
            "n_families": sum(len(values) for values in cells.values()),
            "cells": {cell: {"mean": statistics.fmean(values), "n": len(values)}
                      for cell, values in cells.items()}}


def _node_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cases = load_cases()
    nodes = all_nodes(cases)
    rows = read_jsonl(RUN_DIR / "predictions.jsonl")
    preds = {
        "extract": _prediction_map(rows, "audit", "extract"),
        "review_generic": _prediction_map(rows, "audit", "review_generic"),
        "review_validity": _prediction_map(rows, "audit", "review_validity"),
    }
    return cases, nodes, preds, {row.get("traj_key"): row for row in rows
                                 if row.get("record_type") == "trajectory"}


def analyze_nodes(nodes: list[dict[str, Any]], preds: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    method_records = {"E": preds["extract"], "R": preds["review_generic"], "V": preds["review_validity"]}
    out: dict[str, Any] = {"methods": {}, "joint_success": {}, "relation": {}, "evidence": {}}
    for method, records in method_records.items():
        method_out: dict[str, Any] = {"by_representation": {}}
        for rep in REPRESENTATIONS:
            rep_out = {"by_workflow": {}}
            for workflow in WORKFLOWS:
                workflow_out = {"root": {}, "after": {}, "joint_by_condition": {}}
                for stage in ("root", "after"):
                    condition_stats = {}
                    for condition in CONDITIONS:
                        selected = [node for node in nodes if node["workflow"] == workflow
                                    and node["representation"] == rep and node["stage"] == stage
                                    and node["condition"] == condition]
                        correct = sum(_state_correct(workflow, records[node["node_id"]]["parsed"], node)
                                      for node in selected)
                        total = len(selected)
                        invalid = sum(records[node["node_id"]].get("invalid", False) for node in selected)
                        condition_stats[condition] = {"correct": correct, "total": total,
                                                      "accuracy": correct / total if total else None,
                                                      "invalid": invalid}
                        if stage == "root":
                            if condition == "U":
                                condition_stats[condition]["retract_correct"] = correct
                                initial_by_family = {case["family_id"]: case["initial_value"]
                                                     for case in load_cases()}
                                condition_stats[condition]["old_value_use"] = sum(
                                    records[node["node_id"]]["parsed"].get("knowledge") == "known"
                                    and _value_matches(workflow, records[node["node_id"]]["parsed"].get("value"),
                                                       initial_by_family[node["family_id"]])
                                    for node in selected)
                            elif condition == "F":
                                condition_stats[condition]["new_value_adopt"] = correct
                                initial_by_family = {case["family_id"]: case["initial_value"]
                                                     for case in load_cases()}
                                condition_stats[condition]["old_value_use"] = sum(
                                    records[node["node_id"]]["parsed"].get("knowledge") == "known"
                                    and _value_matches(workflow, records[node["node_id"]]["parsed"].get("value"),
                                                       initial_by_family[node["family_id"]])
                                    for node in selected)
                            else:
                                condition_stats[condition]["erroneous_retraction"] = sum(
                                    records[node["node_id"]]["parsed"].get("knowledge") == "unknown"
                                    for node in selected)
                    workflow_out[stage] = condition_stats
                # Joint success uses the four root conditions for each family/representation.
                family_values = {}
                for family_id in sorted({node["family_id"] for node in nodes if node["workflow"] == workflow}):
                    ok = True
                    for condition in CONDITIONS:
                        node = next(node for node in nodes if node["family_id"] == family_id
                                    and node["representation"] == rep and node["stage"] == "root"
                                    and node["condition"] == condition)
                        ok = ok and _state_correct(workflow, records[node["node_id"]]["parsed"], node)
                    family_values[family_id] = {"cell": workflow, "joint": 1.0 if ok else 0.0}
                workflow_out["joint"] = _stratified_bootstrap(family_values, "joint")
                rep_out["by_workflow"][workflow] = workflow_out
            method_out["by_representation"][rep] = rep_out
        out["methods"][method] = method_out
    # R/V relation and relation-correct-but-final-wrong.
    for method, records in (("R", preds["review_generic"]), ("V", preds["review_validity"])):
        out["relation"][method] = {}
        for rep in REPRESENTATIONS:
            per_condition = {}
            relation_correct = total = relation_wrong_final = 0
            for node in nodes:
                if node["representation"] != rep:
                    continue
                record = records[node["node_id"]]
                parsed = record["parsed"]
                total += 1
                rel_ok = parsed.get("relation") == node["truth_relation"]
                relation_correct += rel_ok
                if rel_ok and not _state_correct(node["workflow"], parsed, node):
                    relation_wrong_final += 1
                key = f"{node['workflow']}:{node['condition']}:{node['stage']}"
                entry = per_condition.setdefault(key, {"correct": 0, "total": 0,
                                                       "relation_correct_final_wrong": 0})
                entry["total"] += 1
                entry["correct"] += rel_ok
                entry["relation_correct_final_wrong"] += int(
                    rel_ok and not _state_correct(node["workflow"], parsed, node))
            out["relation"][method][rep] = {
                "relation_accuracy": relation_correct / total if total else None,
                "n": total, "relation_correct_final_wrong": relation_wrong_final,
                "by_condition": per_condition,
            }
    # Main N V-R joint difference.
    for method in ("E", "R", "V"):
        # Keep common family keys for the main comparison.
        pass
    v_values = {}
    r_values = {}
    for family_id in sorted({node["family_id"] for node in nodes}):
        workflow = next(case["workflow"] for case in load_cases() if case["family_id"] == family_id)
        for target, method in ((v_values, "V"), (r_values, "R")):
            ok = True
            for condition in CONDITIONS:
                node = next(node for node in nodes if node["family_id"] == family_id
                            and node["representation"] == "N" and node["stage"] == "root"
                            and node["condition"] == condition)
                ok = ok and _state_correct(workflow, method_records[method][node["node_id"]]["parsed"], node)
            target[family_id] = {"cell": workflow, "joint": 1.0 if ok else 0.0}
    deltas = {family_id: {"cell": v_values[family_id]["cell"],
                          "joint": v_values[family_id]["joint"] - r_values[family_id]["joint"]}
              for family_id in v_values}
    out["main_N_joint_V_minus_R"] = _paired_bootstrap(deltas, "joint")
    out["evidence"] = {"E": {}, "R": {}, "V": {}}
    for method, records in method_records.items():
        citations = valid = invalid_outputs = empty_outputs = 0
        for node in nodes:
            record = records.get(node["node_id"])
            if record is None:
                continue
            ids = {"e1", "e2"} | ({"e3"} if node["stage"] == "after" else set())
            evidence = record["parsed"].get("evidence", [])
            citations += len(evidence)
            valid += sum(1 for item in evidence if item in ids)
            invalid_outputs += int(record["parsed"].get("evidence_invalid", False))
            empty_outputs += int(record["parsed"].get("evidence_empty", False))
        out["evidence"][method] = {"citations": citations, "valid_citations": valid,
                                   "valid_citation_precision": valid / citations if citations else None,
                                   "invalid_outputs": invalid_outputs, "empty_outputs": empty_outputs}
    return out

def analyze_strategies(cases: list[dict[str, Any]], trajectories: list[dict[str, Any]],
                       run: dict[str, Any]) -> dict[str, Any]:
    audit = [case for case in cases if case["split"] == "audit"]
    by_key = defaultdict(list)
    for row in trajectories:
        by_key[(row["strategy"], row["representation"], row["family_id"],
                row["condition"])].append(row)
    family_cell = {case["family_id"]: case["workflow"] for case in audit}
    fields = {
        "utility": lambda row: float(row["utility"]),
        "success": lambda row: 1.0 if row["success"] else 0.0,
        "wrong_side_effect": lambda row: 1.0 if row["wrong_side_effect"] else 0.0,
        "tool_error": lambda row: 1.0 if row["tool_error"] else 0.0,
        "abort": lambda row: 1.0 if row["abort"] else 0.0,
        "queries": lambda row: float(row["queries"]),
        "model_calls": lambda row: float(row["model_calls"]),
        "input_tokens": lambda row: float(row["input_tokens"]),
        "output_tokens": lambda row: float(row["output_tokens"]),
        "rule_fallback_root": lambda row: float(row.get("rule_fallback_root", False)),
        "rule_fallback_after": lambda row: float(row.get("rule_fallback_after", False)),
    }
    result: dict[str, Any] = {"by_representation": {}, "main_N": {}, "comparisons": {},
                              "rule_coverage": {}, "condition_N": {}}
    family_maps: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for representation in REPRESENTATIONS:
        rep_out: dict[str, Any] = {"strategies": {}}
        for strategy in STRATEGIES:
            family_values: dict[str, dict[str, float]] = {}
            for family in audit:
                fid = family["family_id"]
                condition_values = defaultdict(list)
                for condition in CONDITIONS:
                    rows = by_key.get((strategy, representation, fid, condition), [])
                    if not rows:
                        raise RuntimeError(f"missing execution row: {strategy}/{representation}/{fid}/{condition}")
                    weight_sum = sum(row["world_weight"] for row in rows)
                    for field, getter in fields.items():
                        condition_values[field].append(
                            sum(row["world_weight"] * getter(row) for row in rows) / weight_sum)
                family_values[fid] = {
                    "cell": family_cell[fid],
                    **{field: statistics.fmean(values) for field, values in condition_values.items()},
                }
                # Query-cost sensitivity, no re-selection.
                for cost in (0.0, 0.25, 1.0, 4.0):
                    q = family_values[fid]["queries"]
                    base_utility = statistics.fmean(
                        [sum(r["world_weight"] * r["utility"] for r in by_key[(strategy, representation, fid, c)])
                         / sum(r["world_weight"] for r in by_key[(strategy, representation, fid, c)])
                         for c in CONDITIONS])
                    family_values[fid][f"utility_cost_{cost:g}"] = base_utility + q * (1.0 - cost)
            rep_out["strategies"][strategy] = {
                field: _stratified_bootstrap(family_values, field) for field in fields
            }
            rep_out["strategies"][strategy].update({
                field: _stratified_bootstrap(family_values, field)
                for field in ("utility_cost_0", "utility_cost_0.25", "utility_cost_1", "utility_cost_4")
            })
            family_maps[(representation, strategy)] = family_values
        result["by_representation"][representation] = rep_out
    # Main table: natural-language representation.
    for strategy in STRATEGIES:
        result["main_N"][strategy] = result["by_representation"]["N"]["strategies"][strategy]
    for left, right in (("VALIDITY_REVIEW", "QUERY_ONCE_V"),
                        ("VALIDITY_REVIEW", "GENERIC_REVIEW"),
                        ("DEPENDENCY_RULE", "VALIDITY_REVIEW"),
                        ("DEPENDENCY_RULE", "QUERY_ONCE_V")):
        lv = family_maps[("N", left)]
        rv = family_maps[("N", right)]
        deltas = {fid: {"cell": lv[fid]["cell"],
                        **{field: lv[fid][field] - rv[fid][field]
                           for field in ("utility", "success", "wrong_side_effect", "queries")}}
                  for fid in lv}
        result["comparisons"][f"{left}__minus__{right}"] = {
            field: _paired_bootstrap(deltas, field)
            for field in ("utility", "success", "wrong_side_effect", "queries")
        }
    # Rule coverage/fallback by representation.
    for representation in REPRESENTATIONS:
        for workflow in WORKFLOWS:
            rows = [row for row in trajectories if row["strategy"] == "DEPENDENCY_RULE"
                    and row["representation"] == representation and row["workflow"] == workflow]
            result["rule_coverage"][f"{representation}:{workflow}"] = {
                "n": len(rows),
                "fallback_root": sum(row.get("rule_fallback_root", False) for row in rows),
                "fallback_after": sum(row.get("rule_fallback_after", False) for row in rows),
                "queries": statistics.fmean(row["queries"] for row in rows) if rows else None,
                "success_rate": statistics.fmean(1.0 if row["success"] else 0.0 for row in rows)
                if rows else None,
            }
    # Condition-level strategy table, N only.
    for workflow in WORKFLOWS:
        for condition in CONDITIONS:
            entry = {}
            for strategy in STRATEGIES:
                rows = [row for row in trajectories if row["strategy"] == strategy
                        and row["representation"] == "N" and row["workflow"] == workflow
                        and row["condition"] == condition]
                entry[strategy] = {
                    "n": len(rows),
                    "utility": statistics.fmean(row["utility"] for row in rows) if rows else None,
                    "success": statistics.fmean(row["success"] for row in rows) if rows else None,
                    "queries": statistics.fmean(row["queries"] for row in rows) if rows else None,
                }
            result["condition_N"][f"{workflow}:{condition}"] = entry
    # Sequential inference estimate from measured audit token throughput.
    audit_evidence = run.get("phase_evidence", {}).get("G6-AUDIT", {})
    elapsed = float(audit_evidence.get("elapsed_seconds", 0.0))
    audit_calls = int(audit_evidence.get("calls_new", 0))
    tokens = int(run.get("budget", {}).get("input_tokens", 0)) + int(run.get("budget", {}).get("output_tokens", 0))
    rate = tokens / elapsed if elapsed > 0 else None
    result["estimated_sequential_seconds"] = {
        strategy: ((result["main_N"][strategy]["input_tokens"]["mean"]
                    + result["main_N"][strategy]["output_tokens"]["mean"]) / rate)
        if rate else None for strategy in STRATEGIES
    }
    result["measured_audit_seconds_per_call"] = elapsed / audit_calls if audit_calls else None
    return result

def _condition_rate(node_metrics: dict[str, Any], method: str, rep: str, workflow: str,
                    condition: str, field: str = "accuracy") -> float | None:
    try:
        entry = node_metrics["methods"][method]["by_representation"][rep]["by_workflow"][workflow]["root"][condition]
    except KeyError:
        return None
    if field == "accuracy":
        return entry.get("accuracy")
    value = entry.get(field)
    return value / entry["total"] if entry.get("total") else None


def _strategy_summary(strategy_metrics: dict[str, Any], strategy: str) -> dict[str, Any]:
    table = strategy_metrics["main_N"][strategy]
    return {field: table[field]["mean"] for field in
            ("utility", "success", "wrong_side_effect", "tool_error", "abort", "queries",
             "model_calls", "input_tokens", "output_tokens")}


def _build_decision(node_metrics: dict[str, Any], strategy_metrics: dict[str, Any],
                    run: dict[str, Any]) -> dict[str, Any]:
    phenomenon = {}
    for workflow in WORKFLOWS:
        for rep in REPRESENTATIONS:
            stats = node_metrics["methods"]["E"]["by_representation"][rep]["by_workflow"][workflow]["root"]["U"]
            phenomenon[f"{workflow}:{rep}"] = {
                "retract_correct": stats["retract_correct"] / stats["total"] if stats["total"] else None,
                "old_value_use": stats["old_value_use"] / stats["total"] if stats["total"] else None,
                "n": stats["total"],
            }
    both_retract = all(phenomenon[f"{workflow}:{rep}"]["retract_correct"] is not None
                       and phenomenon[f"{workflow}:{rep}"]["retract_correct"] >= 0.95
                       for workflow in WORKFLOWS for rep in REPRESENTATIONS)
    joint = node_metrics["main_N_joint_V_minus_R"]
    s_retract = {}
    for workflow in WORKFLOWS:
        v = _condition_rate(node_metrics, "V", "N", workflow, "S", "erroneous_retraction")
        o = _condition_rate(node_metrics, "V", "N", workflow, "O", "erroneous_retraction")
        rv = _condition_rate(node_metrics, "R", "N", workflow, "S", "erroneous_retraction")
        ro = _condition_rate(node_metrics, "R", "N", workflow, "O", "erroneous_retraction")
        s_retract[workflow] = {"V_S": v, "V_O": o, "R_S": rv, "R_O": ro,
                               "max_V_minus_R": max((v or 0) - (rv or 0), (o or 0) - (ro or 0))}
    intervention_pass = bool(joint["ci95"][0] > 0 and joint["delta"] >= 0.10
                             and all(v["max_V_minus_R"] <= 0.05 for v in s_retract.values()))
    comp = strategy_metrics["comparisons"]["VALIDITY_REVIEW__minus__QUERY_ONCE_V"]
    v = _strategy_summary(strategy_metrics, "VALIDITY_REVIEW")
    q = _strategy_summary(strategy_metrics, "QUERY_ONCE_V")
    deployment_pass = bool(comp["utility"]["ci95"][0] > 0 and comp["utility"]["delta"] >= 0.25
                          and comp["wrong_side_effect"]["delta"] <= 0
                          and comp["success"]["delta"] >= -0.01
                          and (q["queries"] - v["queries"]) >= 0.25)
    rule = _strategy_summary(strategy_metrics, "DEPENDENCY_RULE")
    rule_v_utility = rule["utility"] - v["utility"]
    rule_solved = bool(rule["success"] >= 0.999 and rule["wrong_side_effect"] == 0
                       and rule["model_calls"] == 0)
    learning_case = {
        "rule": rule, "VALIDITY_REVIEW": v,
        "rule_minus_V_utility": rule_v_utility,
        "rule_minus_V_success": rule["success"] - v["success"],
        "rule_minus_V_wrong": rule["wrong_side_effect"] - v["wrong_side_effect"],
        "rule_model_calls": rule["model_calls"],
        "rule_solved_all": rule_solved,
        "learning_needed": not (rule_solved or (rule["utility"] >= v["utility"] and rule["model_calls"] < v["model_calls"])),
    }
    relation = {}
    for rep in REPRESENTATIONS:
        entry = node_metrics["relation"]["V"][rep]
        relation[rep] = {
            "relation_accuracy": entry["relation_accuracy"],
            "relation_correct_final_wrong": entry["relation_correct_final_wrong"],
            "n": entry["n"],
        }
    if both_retract:
        next_direction = ("External phenomenon not reproduced strongly at this external setting; "
                          "record the range limitation and do not generalize the G5 failure. Stop this external line.")
    elif learning_case["rule_solved_all"] or (rule["utility"] >= v["utility"] and rule["model_calls"] < v["model_calls"]):
        next_direction = ("Strong dependency rule is sufficient at this task; keep it as an engineering result "
                          "and do not start process-supervision training from G6.")
    elif intervention_pass and deployment_pass:
        next_direction = ("Evidence supports one next process-supervision study: compare terminal reward vs "
                          "generic state supervision vs observation-consistent revoke/keep/replace process "
                          "supervision under the same actor and budget, including actual tool utility.")
    else:
        next_direction = ("No threshold-clearing reason for process-supervision training; stop this G6 line "
                          "and do not add a larger same-template matrix.")
    return {
        "phenomenon": {"by_workflow_representation": phenomenon,
                       "both_workflow_retract_ge_95": both_retract,
                       "judgement": "NOT_REPRODUCED" if both_retract else "REPRODUCED_LOCALLY"},
        "intervention": {"V_minus_R_joint_N": joint, "S_O_erroneous_retraction": s_retract,
                         "threshold": "delta>=0.10, CI lower>0, V retraction <= R + 5pp",
                         "pass": intervention_pass},
        "deployment": {"VALIDITY_REVIEW": v, "QUERY_ONCE_V": q,
                       "V_minus_Q": comp,
                       "threshold": "utility>=0.25, CI lower>0, no error increase, success drop<=1pp, query saving>=0.25",
                       "pass": deployment_pass},
        "learning_case": learning_case,
        "relation_composition": relation,
        "next_direction": next_direction,
        "not_claims": ["not a ToolSandbox benchmark score", "not a new PRM method",
                       "not evidence of general asynchronous-change or TOCTOU safety"],
    }


def run_analysis() -> None:
    run = initialize_run()
    cases = load_cases()
    all_nodes_ = all_nodes(cases)
    audit_nodes = [node for node in all_nodes_ if node["split"] == "audit"]
    rows = read_jsonl(RUN_DIR / "predictions.jsonl")
    preds = {
        "extract": _prediction_map(rows, "audit", "extract"),
        "review_generic": _prediction_map(rows, "audit", "review_generic"),
        "review_validity": _prediction_map(rows, "audit", "review_validity"),
    }
    trajectories = [row for row in rows if row.get("record_type") == "trajectory"]
    node_metrics = analyze_nodes(audit_nodes, preds)
    strategy_metrics = analyze_strategies(cases, trajectories, run)
    decision = _build_decision(node_metrics, strategy_metrics, run)
    run["metrics"] = {
        "g6": {
            "nodes": node_metrics,
            "strategies": strategy_metrics,
            "decision": decision,
            "trajectory_count": len(trajectories),
            "notes": "audit-only n=48 families; U uses two hidden worlds with weight 1/2; "
                     "J/N are not independent samples.",
        }
    }
    run["analysis_completed_at"] = now_cst()
    run["phase_evidence"]["G6-ANALYZE"] = {"completed_at": now_cst(), "model_calls": 0}
    mark_phase(run, "G6-ANALYZE", "COMPLETED")
    print(json.dumps({"decision": decision,
                      "offline": {method: node_metrics["methods"][method]["by_representation"]["N"]
                                  ["by_workflow"] for method in ("E", "R", "V")},
                      "main_N": strategy_metrics["main_N"],
                      "comparisons": strategy_metrics["comparisons"]},
                     indent=2, ensure_ascii=False, default=str))



def main() -> None:
    parser = argparse.ArgumentParser(description="G6 validity and minimal-repair experiment")
    parser.add_argument("--phase", choices=("g5-reanalysis", "data", "dev", "audit",
                                            "execute", "analyze"), required=True)
    parser.add_argument("--cpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_worker:
        cpu_worker()
        return
    if args.phase == "g5-reanalysis":
        g5_stage()
    elif args.phase == "data":
        data_stage()
    elif args.phase in {"dev", "audit"}:
        run_offline_phase(args.phase)
    elif args.phase == "execute":
        run_execution()
    elif args.phase == "analyze":
        run_analysis()
    else:
        raise ValueError(args.phase)


if __name__ == "__main__":
    if "--cpu-worker" in sys.argv:
        cpu_worker()
    else:
        main()
