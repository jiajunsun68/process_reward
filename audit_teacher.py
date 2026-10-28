#!/usr/bin/env python3
"""Run, freeze, and analyze the centralized G1 teacher audit.

This is a VPR-style adaptation for a local frozen model.  It is not a claim
that the original VPR model or prompting was reproduced.  Every attempted
teacher call produces at most one predictions.jsonl row; malformed output is
kept and receives the preregistered uniform-action fallback during analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Iterable

from audit_cases import (
    INFO_ACTION,
    MECHANISMS,
    STRATA,
    canonical_json,
    load_cases,
    sha256_bytes,
    sha256_file,
    stable_seed,
    validate_cases,
)


PROMPT_VERSION = "v1"
PROMPT_METHOD = "VPR-style adaptation; local model and task-specific short scoring prompt"
CONDITIONS = ("H", "P", "PR", "PR-IC", "PR-full", "PR-full-IC")
WORLD_CONDITIONS = set(CONDITIONS) - {"H"}
FULL_CONDITIONS = {"PR-full", "PR-full-IC"}
MAX_TEACHER_CALLS = 4272
MAX_NEW_TOKENS = 128
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 23

SYSTEM_PROMPT = (
    "You are a process-action scorer. Score every listed next action with exactly one integer in {-1,0,1}: "
    "-1 is poor or unsafe, 0 is acceptable/uncertain, and 1 is preferred. Return only one JSON object of the form "
    '{"scores":{"ACTION_ID":1}} with every supplied ACTION_ID exactly once and no explanation.'
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prompt_hash() -> str:
    return sha256_bytes((PROMPT_VERSION + "\n" + SYSTEM_PROMPT).encode("utf-8"))


def candidate_block(case: dict[str, Any]) -> str:
    return "\n".join(f"- {action}: {case['candidates'][action]}" for action in case["candidate_order"])


def build_messages(case: dict[str, Any], condition: str, world: dict[str, Any] | None) -> list[dict[str, str]]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if (condition == "H") != (world is None):
        raise ValueError("H has no private world; all other conditions require exactly one world")
    body = [
        f"Public history and rules:\n{case['public_context']['history']}",
        f"Available next actions in fixed order:\n{candidate_block(case)}",
    ]
    if condition != "H":
        body.append(f"Privileged current hidden state for this scoring call: {world['state']}")
    if condition in {"PR", "PR-IC"}:
        body.append("Reference action IDs from the privileged minimal reference workflow: " + json.dumps(world["reference_actions"]))
    if condition in FULL_CONDITIONS:
        body.append("Reference action IDs from the preregistered completeness expansion: " + json.dumps(world["full_reference_actions"]))
        body.append("The expansion follows the same hidden-state reference rule; it is not a history-only oracle answer.")
    if condition in {"PR-IC", "PR-full-IC"}:
        body.append(
            "Information-feasibility constraint: score the next action for a student that can use only the public history now and only observations "
            "actually produced by its chosen action later. Do not reward an action merely because this scorer or the reference generator sees the hidden state."
        )
    else:
        body.append("Use all context explicitly supplied in this scoring call, including any privileged state or references.")
    body.append("Return scores for exactly these IDs: " + json.dumps(case["candidate_order"]))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(body)}]


def parse_scores(text: str, candidate_order: list[str]) -> tuple[dict[str, int] | None, str | None]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], dict):
        return None, "schema_must_be_exact_scores_object"
    scores = payload["scores"]
    if set(scores) != set(candidate_order) or len(scores) != len(candidate_order):
        return None, "candidate_keys_mismatch"
    if any(type(scores[action]) is not int or scores[action] not in (-1, 0, 1) for action in candidate_order):
        return None, "scores_must_be_integer_minus1_0_1"
    return {action: scores[action] for action in candidate_order}, None


def select_dev24(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = {
        "static_diagnosis": {"valuable": 3, "unhelpful": 3, "boundary": 2},
        "dynamic_evidence": {"valuable": 2, "unhelpful": 3, "boundary": 3},
        "progress_uncertainty": {"valuable": 3, "unhelpful": 2, "boundary": 3},
    }
    selected = []
    for mechanism in MECHANISMS:
        for stratum in STRATA:
            group = [case for case in cases if case["split"] == "dev" and case["mechanism"] == mechanism and case["stratum"] == stratum]
            group.sort(key=lambda case: stable_seed(17, "dev24", case["case_id"]))
            selected.extend(group[: quotas[mechanism][stratum]])
    return sorted(selected, key=lambda case: case["case_id"])


def cases_for_phase(cases: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    if phase == "dev24":
        return select_dev24(cases)
    if phase == "dev72":
        return [case for case in cases if case["split"] == "dev"]
    if phase == "audit288":
        return [case for case in cases if case["split"] == "audit"]
    raise ValueError(phase)


def default_conditions(phase: str) -> tuple[str, ...]:
    return ("H", "P", "PR", "PR-IC") if phase.startswith("dev") else CONDITIONS


def call_plan(cases: list[dict[str, Any]], phase: str, conditions: Iterable[str], model_commit: str, seed: int) -> list[dict[str, Any]]:
    plan = []
    for case in cases_for_phase(cases, phase):
        for condition in conditions:
            if condition in FULL_CONDITIONS and not case["full_reference_control"]:
                continue
            worlds: list[dict[str, Any] | None] = [None] if condition == "H" else case["private_context"]["worlds"]
            for world in worlds:
                state = None if world is None else world["state"]
                key_payload = [case["case_id"], condition, state, PROMPT_VERSION, model_commit, seed]
                plan.append({
                    "call_key": sha256_bytes(canonical_json(key_payload).encode("utf-8")),
                    "case": case,
                    "condition": condition,
                    "world": world,
                    "world_state": state,
                })
    return plan


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [row["call_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("predictions.jsonl contains duplicate call_key values")
    return rows


def validate_output_dir(output_dir: Path, attempt_id: str) -> tuple[Path, Path]:
    allowed = {"run.json", "predictions.jsonl"}
    if output_dir.exists():
        unexpected = {path.name for path in output_dir.iterdir()} - allowed
        if unexpected:
            raise ValueError(f"audit output directory has unexpected files: {sorted(unexpected)}")
    else:
        output_dir.mkdir(parents=True)
    run_path, predictions_path = output_dir / "run.json", output_dir / "predictions.jsonl"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("attempt_id") != attempt_id:
            raise ValueError("existing run.json attempt_id differs")
    return run_path, predictions_path


def new_run(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": args.attempt_id,
        "run_id": "G1-TEACHER",
        "scope": "centralized G1 dev/audit teacher scoring; no training and no model saving",
        "status": "INITIALIZED",
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
        "model": {"path": str(args.model), "commit": args.model_commit, "local_files_only": True},
        "prompt": {"version": PROMPT_VERSION, "hash": prompt_hash(), "method": PROMPT_METHOD},
        "data": {"cases_sha256": manifest["hashes"]["cases_sha256"], "split_seed": manifest["split_seed"]},
        "decoding": {"seed": args.seed, "max_new_tokens": args.max_new_tokens, "do_sample": False, "batch_size": args.batch_size},
        "phases_completed": [],
        "teacher_calls_persisted": 0,
        "teacher_calls_unknown_upper_bound": 0,
        "teacher_calls_prior_attempts": args.prior_teacher_calls,
        "teacher_calls_total_budgeted": 0,
        "inflight_call_keys": [],
        "errors": [],
        "model_files_saved": 0,
        "resume_supported": "inference call-key deduplication only; not a training checkpoint",
    }


def run_teacher(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1 or args.max_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be in [1,{MAX_NEW_TOKENS}]")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.prior_teacher_calls < 0:
        raise ValueError("prior_teacher_calls must be nonnegative")
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.model_commit):
        raise ValueError("--model-commit must be a resolved hexadecimal commit")
    cases = load_cases(args.cases)
    validate_cases(cases)
    cases_payload_hash = sha256_bytes(args.cases.read_bytes())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["hashes"]["cases_sha256"] != cases_payload_hash:
        raise ValueError("cases hash differs from manifest")
    if args.phase == "audit288":
        if not manifest.get("audit_frozen"):
            raise ValueError("audit manifest is not frozen")
        if manifest["teacher"]["resolved_commit"] != args.model_commit or manifest["teacher"]["prompt_version"] != PROMPT_VERSION:
            raise ValueError("frozen model commit or prompt version differs")
    conditions = tuple(args.conditions.split(",")) if args.conditions else default_conditions(args.phase)
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError("unknown condition")
    plan = call_plan(cases, args.phase, conditions, args.model_commit, args.seed)
    expected = {"dev24": 216, "dev72": 648, "audit288": 2976}[args.phase]
    if set(conditions) == set(default_conditions(args.phase)) and len(plan) != expected:
        raise AssertionError(f"call plan has {len(plan)}, expected {expected}")
    run_path, predictions_path = validate_output_dir(args.output_dir, args.attempt_id)
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else new_run(args, manifest)
    if run["model"]["commit"] != args.model_commit or run["prompt"]["hash"] != prompt_hash():
        raise ValueError("existing centralized run uses another model commit or prompt")
    frozen_decoding = run["decoding"]
    requested_decoding = {"seed": args.seed, "max_new_tokens": args.max_new_tokens, "do_sample": False, "batch_size": args.batch_size}
    if frozen_decoding != requested_decoding or run["model"]["path"] != str(args.model):
        raise ValueError("model path or decoding settings differ from the centralized run")
    if run.get("teacher_calls_prior_attempts", 0) != args.prior_teacher_calls:
        raise ValueError("--prior-teacher-calls differs from the value frozen in run.json")
    if run.get("inflight_call_keys"):
        run["teacher_calls_unknown_upper_bound"] += len(run["inflight_call_keys"])
        run["errors"].append({"at": now_utc(), "type": "prior_inflight_unknown", "count": len(run["inflight_call_keys"])})
        run["inflight_call_keys"] = []
    existing_rows = read_predictions(predictions_path)
    existing = {row["call_key"] for row in existing_rows}
    remaining = [item for item in plan if item["call_key"] not in existing]
    projected = args.prior_teacher_calls + len(existing) + run["teacher_calls_unknown_upper_bound"] + len(remaining)
    if projected > MAX_TEACHER_CALLS:
        raise RuntimeError(f"teacher call budget would be exceeded: {projected}>{MAX_TEACHER_CALLS}")
    run.update({"status": "LOADING_MODEL", "updated_at_utc": now_utc(), "active_phase": args.phase, "planned_phase_calls": len(plan), "remaining_phase_calls": len(remaining)})
    run["teacher_calls_persisted"] = len(existing)
    run["teacher_calls_total_budgeted"] = args.prior_teacher_calls + len(existing) + run["teacher_calls_unknown_upper_bound"]
    atomic_json(run_path, run)
    if args.phase == "audit288" and not manifest.get("audit_started"):
        manifest["audit_started"] = True
        manifest["audit_started_at_utc"] = now_utc()
        atomic_json(args.manifest, manifest)

    os.environ.setdefault("WANDB_DISABLED", "true")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"unavailable explicit CUDA device: {args.device}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is not None and resolved != args.model_commit:
        raise RuntimeError(f"loaded model commit {resolved} differs from frozen {args.model_commit}")
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    run.update({"status": "RUNNING", "model_loaded_at_utc": now_utc(), "gpu_name": torch.cuda.get_device_name(device), "versions": {"torch": torch.__version__, "transformers": transformers.__version__}})
    atomic_json(run_path, run)
    predictions_path.touch(exist_ok=True)
    try:
        with predictions_path.open("a", encoding="utf-8") as stream:
            for offset in range(0, len(remaining), args.batch_size):
                if time.monotonic() >= deadline:
                    run["status"] = "DEADLINE_REACHED"
                    break
                batch = remaining[offset : offset + args.batch_size]
                run["inflight_call_keys"] = [item["call_key"] for item in batch]
                run["updated_at_utc"] = now_utc()
                atomic_json(run_path, run)
                messages = [build_messages(item["case"], item["condition"], item["world"]) for item in batch]
                prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
                inputs = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(device)
                prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
                before = time.monotonic()
                with torch.inference_mode():
                    outputs = model.generate(**inputs, generation_config=generation_config)
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - before
                input_width = inputs["input_ids"].shape[1]
                for item, output, prompt_length in zip(batch, outputs, prompt_lengths):
                    generated = output[input_width:]
                    raw = tokenizer.decode(generated, skip_special_tokens=True)
                    scores, error = parse_scores(raw, item["case"]["candidate_order"])
                    record = {
                        "call_key": item["call_key"],
                        "case_id": item["case"]["case_id"],
                        "split": item["case"]["split"],
                        "mechanism": item["case"]["mechanism"],
                        "stratum": item["case"]["stratum"],
                        "condition": item["condition"],
                        "world_state": item["world_state"],
                        "prompt_version": PROMPT_VERSION,
                        "model_commit": args.model_commit,
                        "seed": args.seed,
                        "format_valid": scores is not None,
                        "format_error": error,
                        "scores": scores,
                        "raw_output": raw,
                        "prompt_tokens": int(prompt_length),
                        "generated_tokens": int(generated.numel()),
                        "batch_generation_seconds": elapsed,
                        "batch_size": len(batch),
                        "created_at_utc": now_utc(),
                    }
                    stream.write(canonical_json(record) + "\n")
                    stream.flush()
                    existing.add(item["call_key"])
                run["inflight_call_keys"] = []
                run["teacher_calls_persisted"] = len(existing)
                run["teacher_calls_total_budgeted"] = args.prior_teacher_calls + len(existing) + run["teacher_calls_unknown_upper_bound"]
                run["remaining_phase_calls"] = len(plan) - sum(item["call_key"] in existing for item in plan)
                run["updated_at_utc"] = now_utc()
                atomic_json(run_path, run)
    except BaseException as exc:
        run["status"] = "FAILED"
        run["errors"].append({"at": now_utc(), "type": type(exc).__name__, "message": str(exc)})
        run["updated_at_utc"] = now_utc()
        atomic_json(run_path, run)
        raise
    finally:
        run["elapsed_model_seconds"] = time.monotonic() - started
        run["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        run["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
        run["model_files_saved"] = 0
        run["updated_at_utc"] = now_utc()
        atomic_json(run_path, run)
    completed = all(item["call_key"] in existing for item in plan)
    if completed:
        run["status"] = "SUCCEEDED"
        if args.phase not in run["phases_completed"]:
            run["phases_completed"].append(args.phase)
    elif run["status"] != "DEADLINE_REACHED":
        run["status"] = "PARTIAL"
    run["completed_at_utc"] = now_utc()
    run["updated_at_utc"] = now_utc()
    atomic_json(run_path, run)
    print(json.dumps({"status": run["status"], "phase": args.phase, "persisted_calls": len(existing), "remaining_phase_calls": run["remaining_phase_calls"]}, indent=2))


def rows_by_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    lookup = {}
    for row in rows:
        key = (row["case_id"], row["condition"], row["world_state"])
        if key in lookup:
            raise ValueError(f"duplicate logical prediction: {key}")
        lookup[key] = row
    return lookup


def aggregate_case_condition(case: dict[str, Any], condition: str, lookup: dict[tuple[str, str, str | None], dict[str, Any]]) -> dict[str, Any] | None:
    worlds: list[dict[str, Any] | None] = [None] if condition == "H" else case["private_context"]["worlds"]
    weighted = {action: 0.0 for action in case["candidate_order"]}
    valid = 0
    rows = []
    for world in worlds:
        state = None if world is None else world["state"]
        row = lookup.get((case["case_id"], condition, state))
        if row is None:
            return None
        rows.append(row)
        weight = 1.0 if world is None else world["posterior"]
        scores = row["scores"] if row["format_valid"] else {action: 0 for action in case["candidate_order"]}
        valid += int(row["format_valid"])
        for action in weighted:
            weighted[action] += weight * scores[action]
    maximum = max(weighted.values())
    selected = [action for action in case["candidate_order"] if abs(weighted[action] - maximum) <= 1e-12]
    q_pi = case["gold"]["q_pi"]
    utility = statistics.fmean(q_pi[action] for action in selected)
    optimum = max(q_pi.values())
    regret = (optimum - utility) / case["gold"]["reward_span"]
    probability_each = 1.0 / len(selected)
    outcomes = case["gold"]["candidate_outcomes_under_pi"]
    information_probability = probability_each if INFO_ACTION[case["mechanism"]] in selected else 0.0
    return {
        "case_id": case["case_id"],
        "mechanism": case["mechanism"],
        "stratum": case["stratum"],
        "condition": condition,
        "aggregated_scores": weighted,
        "selected_actions": selected,
        "expected_q_pi": utility,
        "normalized_regret": regret,
        "information_action_probability": information_probability,
        "information_cost": information_probability * case["public_context"]["information_cost"],
        "harm_probability": statistics.fmean(outcomes[action]["harm_probability"] for action in selected),
        "pause_probability": statistics.fmean(outcomes[action]["pause_probability"] for action in selected),
        "format_valid_calls": valid,
        "format_total_calls": len(rows),
        "all_calls_valid": valid == len(rows),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_only = [item for item in results if item["all_calls_valid"]]
    valuable = [item for item in results if item["stratum"] == "valuable"]
    unhelpful = [item for item in results if item["stratum"] == "unhelpful"]
    return {
        "root_cases": len(results),
        "mean_normalized_regret": statistics.fmean(item["normalized_regret"] for item in results) if results else None,
        "valid_only_root_cases": len(valid_only),
        "valid_only_mean_normalized_regret": statistics.fmean(item["normalized_regret"] for item in valid_only) if valid_only else None,
        "mean_expected_q_pi": statistics.fmean(item["expected_q_pi"] for item in results) if results else None,
        "mean_harm_probability": statistics.fmean(item["harm_probability"] for item in results) if results else None,
        "mean_pause_probability": statistics.fmean(item["pause_probability"] for item in results) if results else None,
        "mean_information_action_probability": statistics.fmean(item["information_action_probability"] for item in results) if results else None,
        "valuable_information_miss_probability": statistics.fmean(1.0 - item["information_action_probability"] for item in valuable) if valuable else None,
        "unhelpful_information_selection_probability": statistics.fmean(item["information_action_probability"] for item in unhelpful) if unhelpful else None,
        "unhelpful_mean_information_cost": statistics.fmean(item["information_cost"] for item in unhelpful) if unhelpful else None,
        "format_valid_calls": sum(item["format_valid_calls"] for item in results),
        "format_total_calls": sum(item["format_total_calls"] for item in results),
        "format_valid_rate": (
            sum(item["format_valid_calls"] for item in results) / sum(item["format_total_calls"] for item in results) if results else None
        ),
        "all_calls_valid_case_rate": statistics.fmean(item["all_calls_valid"] for item in results) if results else None,
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - position) + sorted_values[high] * (position - low)


def stratified_bootstrap(differences: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {mechanism: [item["difference"] for item in differences if item["mechanism"] == mechanism] for mechanism in MECHANISMS}
    if any(not values for values in groups.values()):
        raise ValueError("bootstrap requires every mechanism")
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        replicate = []
        for values in groups.values():
            replicate.extend(rng.choice(values) for _ in range(len(values)))
        samples.append(statistics.fmean(replicate))
    samples.sort()
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "unit": "independent_root_case",
        "stratified_by": "mechanism",
        "mean": statistics.fmean(item["difference"] for item in differences),
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
    }


def analyze(args: argparse.Namespace) -> None:
    cases = load_cases(args.cases)
    validate_cases(cases)
    case_map = {case["case_id"]: case for case in cases}
    rows = read_predictions(args.predictions)
    lookup = rows_by_lookup(rows)
    audit_cases = [case for case in cases if case["split"] == "audit"]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = Counter()
    for case in audit_cases:
        for condition in CONDITIONS:
            if condition in FULL_CONDITIONS and not case["full_reference_control"]:
                continue
            result = aggregate_case_condition(case, condition, lookup)
            if result is None:
                missing[condition] += 1
            else:
                by_condition[condition].append(result)
    metrics: dict[str, Any] = {
        "analysis_scope": "posterior-aggregate scores first, then induce uniform-tie student choice under exact Q^pi",
        "independent_n": len(audit_cases),
        "prediction_rows": len(rows),
        "teacher_calls_counted": len(rows),
        "missing_root_cases_by_condition": dict(missing),
        "conditions": {condition: summarize_results(by_condition[condition]) for condition in CONDITIONS},
    }
    complete_primary = len(by_condition["PR"]) == 288 and len(by_condition["PR-IC"]) == 288
    if complete_primary:
        pr = {item["case_id"]: item for item in by_condition["PR"]}
        pric = {item["case_id"]: item for item in by_condition["PR-IC"]}
        differences = [{"case_id": cid, "mechanism": case_map[cid]["mechanism"], "difference": pr[cid]["normalized_regret"] - pric[cid]["normalized_regret"]} for cid in pr]
        primary = stratified_bootstrap(differences)
        valuable = [item for item in by_condition["PR"] if item["stratum"] == "valuable"]
        overall_problem_rate = statistics.fmean(item["normalized_regret"] >= 0.05 for item in valuable)
        rates = {
            mechanism: statistics.fmean(item["normalized_regret"] >= 0.05 for item in valuable if item["mechanism"] == mechanism)
            for mechanism in MECHANISMS
        }
        mechanisms_at_gate = sum(rate >= 0.15 for rate in rates.values())
        primary["valuable_PR_regret_ge_0.05_rate"] = overall_problem_rate
        primary["valuable_PR_rate_by_mechanism"] = rates
        primary["mechanisms_at_or_above_0.15"] = mechanisms_at_gate
        valid_differences = [item for item in differences if pr[item["case_id"]]["all_calls_valid"] and pric[item["case_id"]]["all_calls_valid"]]
        primary["valid_only_sensitivity"] = {
            "root_cases": len(valid_differences),
            "bootstrap": stratified_bootstrap(valid_differences)
            if valid_differences and all(any(item["mechanism"] == mechanism for item in valid_differences) for mechanism in MECHANISMS)
            else None,
        }
        metrics["primary_PR_minus_PRIC"] = primary
    complete_full = len(by_condition["PR-full"]) == 72 and len(by_condition["PR-full-IC"]) == 72
    if complete_full:
        full = {item["case_id"]: item for item in by_condition["PR-full"]}
        full_ic = {item["case_id"]: item for item in by_condition["PR-full-IC"]}
        full_differences = [{"case_id": cid, "mechanism": case_map[cid]["mechanism"], "difference": full[cid]["normalized_regret"] - full_ic[cid]["normalized_regret"]} for cid in full]
        control = {"residual_PRfull_minus_PRfullIC": stratified_bootstrap(full_differences)}
        if complete_primary:
            coverage_pr = [{"case_id": cid, "mechanism": case_map[cid]["mechanism"], "difference": pr[cid]["normalized_regret"] - full[cid]["normalized_regret"]} for cid in full]
            coverage_pric = [{"case_id": cid, "mechanism": case_map[cid]["mechanism"], "difference": pric[cid]["normalized_regret"] - full_ic[cid]["normalized_regret"]} for cid in full]
            control["coverage_effect_PR_minus_PRfull"] = stratified_bootstrap(coverage_pr)
            control["coverage_effect_PRIC_minus_PRfullIC"] = stratified_bootstrap(coverage_pric)
        metrics["reference_completeness_control"] = control
    if complete_primary and complete_full:
        primary = metrics["primary_PR_minus_PRIC"]
        control = metrics["reference_completeness_control"]["residual_PRfull_minus_PRfullIC"]
        format_rates = [metrics["conditions"][condition]["format_valid_rate"] for condition in ("H", "P", "PR", "PR-IC")]
        gate = {
            "PR_problem_rate": primary["valuable_PR_regret_ge_0.05_rate"] >= 0.15 and primary["mechanisms_at_or_above_0.15"] >= 2,
            "PR_to_PRIC_improvement": primary["mean"] >= 0.01 and primary["ci95"][0] > 0.0,
            "not_fully_explained_by_reference_coverage": control["mean"] > 0.0,
            "audit_format_rate_gap_le_0.02": max(format_rates) - min(format_rates) <= 0.02,
        }
        metrics["gate"] = gate
        if all(gate.values()):
            decision = "continue only if a concrete post-constraint effect, cost, or transfer gap is separately identified; G2 is not automatic"
        elif gate["PR_problem_rate"] and gate["PR_to_PRIC_improvement"] and not gate["not_fully_explained_by_reference_coverage"]:
            decision = "simple reference-completeness/information-feasibility repair appears sufficient on this audit"
        else:
            decision = "insufficient mechanism evidence; no-go for automatic expansion"
        metrics["decision"] = decision
    else:
        metrics["decision"] = "NOT_EVALUATED: fixed audit coverage is incomplete"
    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    run["analysis"] = metrics
    run["updated_at_utc"] = now_utc()
    atomic_json(args.run_json, run)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))


def freeze(args: argparse.Namespace) -> None:
    cases = load_cases(args.cases)
    validate_cases(cases)
    rows = read_predictions(args.predictions)
    lookup = rows_by_lookup(rows)
    dev = [case for case in cases if case["split"] == "dev"]
    valid, total, missing = 0, 0, 0
    for case in dev:
        for condition in ("H", "P", "PR", "PR-IC"):
            worlds: list[dict[str, Any] | None] = [None] if condition == "H" else case["private_context"]["worlds"]
            for world in worlds:
                state = None if world is None else world["state"]
                row = lookup.get((case["case_id"], condition, state))
                if row is None:
                    missing += 1
                else:
                    total += 1
                    valid += int(row["format_valid"])
    if missing or total != 648:
        raise RuntimeError(f"cannot freeze: dev calls total={total}, missing={missing}, expected=648")
    valid_rate = valid / total
    if valid_rate < 0.98:
        raise RuntimeError(f"cannot freeze: dev format validity {valid_rate:.4%}<98%; at most one prompt revision is allowed")
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.model_commit):
        raise ValueError("resolved hexadecimal model commit required")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("audit_started"):
        raise RuntimeError("cannot refreeze after audit started")
    if sha256_bytes(args.cases.read_bytes()) != manifest["hashes"]["cases_sha256"]:
        raise RuntimeError("cases hash mismatch")
    manifest["audit_frozen"] = True
    manifest["frozen_at_utc"] = now_utc()
    manifest["teacher"].update({
        "requested_revision": args.model_commit,
        "resolved_commit": args.model_commit,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "method_label": PROMPT_METHOD,
    })
    manifest["dev_format_validation"] = {"calls": total, "valid": valid, "valid_rate": valid_rate, "prompt_revisions_used": args.prompt_revisions_used}
    manifest["hashes"]["audit_teacher_py_sha256"] = sha256_file(Path(__file__))
    atomic_json(args.manifest, manifest)
    print(json.dumps({"audit_frozen": True, "model_commit": args.model_commit, "prompt_version": PROMPT_VERSION, "dev_format_valid_rate": valid_rate}, indent=2))


def validate_prompt(args: argparse.Namespace) -> None:
    cases = load_cases(args.cases)
    summary = validate_cases(cases)
    counts = Counter()
    for phase in ("dev24", "dev72", "audit288"):
        conditions = default_conditions(phase)
        plan = call_plan(cases, phase, conditions, args.model_commit, 42)
        counts[phase] = len(plan)
        for item in plan[: min(len(plan), 100)]:
            messages = build_messages(item["case"], item["condition"], item["world"])
            serialized = canonical_json(messages)
            if item["condition"] == "H" and any(token in serialized for token in ("Privileged current hidden state", "reference action IDs")):
                raise AssertionError("private context leaked into H prompt")
    if counts != Counter({"audit288": 2976, "dev72": 648, "dev24": 216}):
        raise AssertionError(counts)
    parser_checks = {
        "valid": parse_scores('{"scores":{"a":1,"b":0,"c":-1}}', ["a", "b", "c"])[0] is not None,
        "reject_missing": parse_scores('{"scores":{"a":1}}', ["a", "b", "c"])[0] is None,
        "reject_float": parse_scores('{"scores":{"a":1.0,"b":0,"c":-1}}', ["a", "b", "c"])[0] is None,
    }
    if not all(parser_checks.values()):
        raise AssertionError(parser_checks)
    print(json.dumps({"cases": summary["independent_root_cases"], "call_plans": dict(counts), "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(), "parser_checks": parser_checks}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--cases", type=Path, default=Path(__file__).parent / "data" / "audit" / "cases.jsonl")
    validate.add_argument("--model-commit", default="0" * 40)
    run = subparsers.add_parser("run")
    run.add_argument("--cases", type=Path, default=Path(__file__).parent / "data" / "audit" / "cases.jsonl")
    run.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data" / "audit" / "manifest.json")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--attempt-id", required=True)
    run.add_argument("--phase", choices=("dev24", "dev72", "audit288"), required=True)
    run.add_argument("--conditions", default=None, help="Comma-separated subset; defaults to the preregistered phase conditions")
    run.add_argument("--model", type=Path, required=True, help="One already-cached local snapshot; network is never used here")
    run.add_argument("--model-commit", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--prior-teacher-calls", type=int, default=0, help="Conservative calls consumed by all earlier attempts, including unknown inflight calls")
    run.add_argument("--deadline-seconds", type=int, default=4 * 60 * 60)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--cases", type=Path, default=Path(__file__).parent / "data" / "audit" / "cases.jsonl")
    freeze_parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data" / "audit" / "manifest.json")
    freeze_parser.add_argument("--predictions", type=Path, required=True)
    freeze_parser.add_argument("--model-commit", required=True)
    freeze_parser.add_argument("--prompt-revisions-used", type=int, choices=(0, 1), default=0)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--cases", type=Path, default=Path(__file__).parent / "data" / "audit" / "cases.jsonl")
    analysis.add_argument("--predictions", type=Path, required=True)
    analysis.add_argument("--run-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        validate_prompt(args)
    elif args.command == "run":
        run_teacher(args)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
