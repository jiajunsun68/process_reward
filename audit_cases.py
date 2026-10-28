#!/usr/bin/env python3
"""Build and validate the frozen three-mechanism G1 audit cases.

The exact gold is Q^pi for one fixed, history-only continuation policy.  The
same policy is also rolled out by a seeded CPU Monte Carlo control.  Hidden
states, candidate actions, and Monte Carlo samples remain nested within one
root case and never increase the independent sample count.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any, Callable


SPLIT_SEED = 17
MC_SEED = 31
MC_ROLLOUTS_PER_ACTION = 256
REWARD_CORRECT = 10.0
REWARD_WRONG = -30.0
REWARD_SPAN = 40.0
MECHANISMS = ("static_diagnosis", "dynamic_evidence", "progress_uncertainty")
STRATA = ("valuable", "unhelpful", "boundary")
STATE_COUNTS = {"static_diagnosis": 2, "dynamic_evidence": 2, "progress_uncertainty": 4}
INFO_ACTION = {"static_diagnosis": "inspect", "dynamic_evidence": "refresh", "progress_uncertainty": "verify"}
PROGRESS_REWARDS = {
    "not_started": {"retry": 10.0, "wait": -10.0, "compensate": -30.0},
    "in_progress": {"retry": -8.0, "wait": 8.0, "compensate": -20.0},
    "committed": {"retry": -30.0, "wait": 6.0, "compensate": 10.0},
    "completed": {"retry": -30.0, "wait": 10.0, "compensate": -20.0},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.exists() else None


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def binary_reward(action: str, state: str, prefix: str) -> float:
    if action == "pause":
        return 0.0
    target = action.removeprefix(prefix)
    return REWARD_CORRECT if target == state else REWARD_WRONG


def binary_terminal_action(p_a: float, prefix: str) -> str:
    values = {
        "pause": 0.0,
        f"{prefix}A": 40.0 * p_a - 30.0,
        f"{prefix}B": 10.0 - 40.0 * p_a,
    }
    maximum = max(values.values())
    return next(action for action, value in values.items() if maximum - value <= 1e-12)


def report_posterior(p_a: float, accuracy: float, report: str) -> float:
    likelihood_a = accuracy if report == "A" else 1.0 - accuracy
    likelihood_b = 1.0 - accuracy if report == "A" else accuracy
    denominator = p_a * likelihood_a + (1.0 - p_a) * likelihood_b
    if denominator <= 0:
        raise ValueError("zero-probability report")
    return p_a * likelihood_a / denominator


def static_world_values(p_a: float, accuracy: float, cost: float) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for state in ("A", "B"):
        inspect = -cost
        for report in ("A", "B"):
            probability = accuracy if report == state else 1.0 - accuracy
            posterior = report_posterior(p_a, accuracy, report)
            terminal = binary_terminal_action(posterior, "repair_")
            inspect += probability * binary_reward(terminal, state, "repair_")
        values[state] = {
            "inspect": inspect,
            "repair_A": binary_reward("repair_A", state, "repair_"),
            "repair_B": binary_reward("repair_B", state, "repair_"),
        }
    return values


def dynamic_world_values(p_a: float, flip: float, accuracy: float, cost: float) -> dict[str, dict[str, float]]:
    final_prior = (1.0 - flip) * p_a + flip * (1.0 - p_a)
    values: dict[str, dict[str, float]] = {}
    for current in ("A", "B"):
        row = {"refresh": -cost, "execute_A": 0.0, "execute_B": 0.0}
        for final in ("A", "B"):
            p_final = (1.0 - flip) if final == current else flip
            row["execute_A"] += p_final * binary_reward("execute_A", final, "execute_")
            row["execute_B"] += p_final * binary_reward("execute_B", final, "execute_")
            for report in ("A", "B"):
                p_report = accuracy if report == final else 1.0 - accuracy
                posterior = report_posterior(final_prior, accuracy, report)
                terminal = binary_terminal_action(posterior, "execute_")
                row["refresh"] += p_final * p_report * binary_reward(terminal, final, "execute_")
        values[current] = row
    return values


def progress_world_values(cost: float) -> dict[str, dict[str, float]]:
    values = {}
    for state, terminal in PROGRESS_REWARDS.items():
        values[state] = {
            "verify": max(terminal.values()) - cost,
            "retry": terminal["retry"],
            "wait": terminal["wait"],
        }
    return values


def aggregate_values(states: list[tuple[str, float]], per_world: dict[str, dict[str, float]]) -> dict[str, float]:
    actions = tuple(next(iter(per_world.values())).keys())
    return {action: sum(weight * per_world[state][action] for state, weight in states) for action in actions}


def candidate_outcomes(
    mechanism: str, params: dict[str, Any], cost: float, states: list[tuple[str, float]]
) -> dict[str, dict[str, float]]:
    """Exact deployed harm/pause probabilities under the same continuation pi."""
    if mechanism == "static_diagnosis":
        actions = ("inspect", "repair_A", "repair_B")
    elif mechanism == "dynamic_evidence":
        actions = ("refresh", "execute_A", "execute_B")
    else:
        actions = ("verify", "retry", "wait")
    outcomes = {action: {"harm_probability": 0.0, "pause_probability": 0.0} for action in actions}
    for state, state_weight in states:
        if mechanism == "static_diagnosis":
            for action in ("repair_A", "repair_B"):
                outcomes[action]["harm_probability"] += state_weight * (binary_reward(action, state, "repair_") < 0)
            accuracy = params["inspection_accuracy"]
            for report in ("A", "B"):
                probability = state_weight * (accuracy if report == state else 1.0 - accuracy)
                posterior = report_posterior(params["posterior_A"], accuracy, report)
                terminal = binary_terminal_action(posterior, "repair_")
                outcomes["inspect"]["harm_probability"] += probability * (binary_reward(terminal, state, "repair_") < 0)
                outcomes["inspect"]["pause_probability"] += probability * (terminal == "pause")
        elif mechanism == "dynamic_evidence":
            flip = params["pre_execution_flip_probability"]
            accuracy = params["refresh_accuracy"]
            p = params["posterior_current_A"]
            final_prior = (1.0 - flip) * p + flip * (1.0 - p)
            for final in ("A", "B"):
                p_final = state_weight * ((1.0 - flip) if final == state else flip)
                for action in ("execute_A", "execute_B"):
                    outcomes[action]["harm_probability"] += p_final * (binary_reward(action, final, "execute_") < 0)
                for report in ("A", "B"):
                    probability = p_final * (accuracy if report == final else 1.0 - accuracy)
                    posterior = report_posterior(final_prior, accuracy, report)
                    terminal = binary_terminal_action(posterior, "execute_")
                    outcomes["refresh"]["harm_probability"] += probability * (binary_reward(terminal, final, "execute_") < 0)
                    outcomes["refresh"]["pause_probability"] += probability * (terminal == "pause")
        else:
            terminal = PROGRESS_REWARDS[state]
            best = max(terminal.values())
            best_action = next(action for action, value in terminal.items() if abs(value - best) <= 1e-12)
            outcomes["verify"]["harm_probability"] += state_weight * (best < 0)
            outcomes["verify"]["pause_probability"] += state_weight * (best_action == "wait")
            for action in ("retry", "wait"):
                outcomes[action]["harm_probability"] += state_weight * (terminal[action] < 0)
                outcomes[action]["pause_probability"] += state_weight * (action == "wait")
    for action in actions:
        outcomes[action]["harm_probability"] = min(1.0, max(0.0, outcomes[action]["harm_probability"]))
        outcomes[action]["pause_probability"] = min(1.0, max(0.0, outcomes[action]["pause_probability"]))
        outcomes[action]["information_cost"] = cost if action == INFO_ACTION[mechanism] else 0.0
    return outcomes


def base_parameters(mechanism: str, rng: random.Random) -> tuple[dict[str, Any], list[tuple[str, float]], Callable[[float], dict[str, dict[str, float]]]]:
    if mechanism == "static_diagnosis":
        p_a = rng.uniform(0.28, 0.72)
        accuracy = rng.uniform(0.82, 0.99)
        params = {"posterior_A": p_a, "inspection_accuracy": accuracy}
        states = [("A", p_a), ("B", 1.0 - p_a)]
        return params, states, lambda cost: static_world_values(p_a, accuracy, cost)
    if mechanism == "dynamic_evidence":
        p_a = rng.uniform(0.12, 0.88)
        flip = rng.uniform(0.05, 0.35)
        accuracy = rng.uniform(0.82, 0.97)
        params = {"posterior_current_A": p_a, "pre_execution_flip_probability": flip, "refresh_accuracy": accuracy}
        states = [("A", p_a), ("B", 1.0 - p_a)]
        return params, states, lambda cost: dynamic_world_values(p_a, flip, accuracy, cost)
    if mechanism == "progress_uncertainty":
        names = list(PROGRESS_REWARDS)
        raw = [rng.gammavariate(1.2, 1.0) for _ in names]
        total = sum(raw)
        weights = [value / total for value in raw]
        params = {"progress_posterior": dict(zip(names, weights)), "verification_accuracy": 1.0}
        states = list(zip(names, weights))
        return params, states, progress_world_values
    raise ValueError(f"unknown mechanism: {mechanism}")


def desired_gap(stratum: str, gross_gap: float, rng: random.Random) -> float | None:
    if stratum == "valuable":
        if gross_gap <= 0.70:
            return None
        return rng.uniform(0.55, min(3.0, gross_gap - 0.10))
    if stratum == "unhelpful":
        return -rng.uniform(0.55, 2.5)
    if stratum == "boundary":
        return rng.uniform(-0.40, 0.40)
    raise ValueError(stratum)


def descriptions(mechanism: str, params: dict[str, Any], cost: float) -> tuple[str, dict[str, str], str]:
    if mechanism == "static_diagnosis":
        p = params["posterior_A"]
        accuracy = params["inspection_accuracy"]
        history = (
            f"A static device has fault A with posterior {p:.6f} and fault B otherwise. "
            f"A new inspection costs {cost:.6f} and reports the unchanged fault with symmetric accuracy {accuracy:.6f}. "
            "After an inspection, the fixed controller may repair A, repair B, or pause using only the report and this history. "
            "A correct repair pays +10, a wrong repair -30, and pause 0. Choose the next action."
        )
        candidates = {
            "inspect": "Buy the inspection, then let the fixed history-only controller act on its report.",
            "repair_A": "Repair fault A immediately without another observation.",
            "repair_B": "Repair fault B immediately without another observation.",
        }
        continuation = "After inspect, Bayes-update from the public posterior and noisy report, then choose pause/repair_A/repair_B with maximal expected reward."
    elif mechanism == "dynamic_evidence":
        p = params["posterior_current_A"]
        flip = params["pre_execution_flip_probability"]
        accuracy = params["refresh_accuracy"]
        history = (
            f"A time-stamped authorization signal implies current state A with posterior {p:.6f} and B otherwise. "
            f"Before execution the state independently flips A<->B with probability {flip:.6f}. "
            f"A just-in-time refresh costs {cost:.6f} and reports the execution-time state with symmetric accuracy {accuracy:.6f}. "
            "After refresh, the fixed controller may execute A, execute B, or pause using only public history and the new report. "
            "A matching execution pays +10, a mismatch -30, and pause 0. Choose the next action."
        )
        candidates = {
            "refresh": "Refresh at execution time, then let the fixed history-only controller use the new report.",
            "execute_A": "Execute the A-specific operation without refreshing.",
            "execute_B": "Execute the B-specific operation without refreshing.",
        }
        continuation = "Integrate the frozen pre-execution transition; after refresh Bayes-update about the execution-time state and choose pause/execute_A/execute_B."
    else:
        posterior = params["progress_posterior"]
        weights = ", ".join(f"{state}={weight:.6f}" for state, weight in posterior.items())
        history = (
            f"An operation timed out. Public telemetry gives the progress posterior {weights}. "
            f"A status verification costs {cost:.6f} and reveals the actual progress state. "
            "The public terminal reward table is: not_started(retry=+10, wait=-10, compensate=-30); "
            "in_progress(retry=-8, wait=+8, compensate=-20); committed(retry=-30, wait=+6, compensate=+10); "
            "completed(retry=-30, wait=+10, compensate=-20). "
            "After verification, the fixed controller chooses retry, wait, or compensate from that table. "
            "Without verification you may retry or wait now. Choose the next action."
        )
        candidates = {
            "verify": "Verify exact progress, then let the fixed history-only controller choose retry/wait/compensate.",
            "retry": "Retry immediately without checking progress.",
            "wait": "Wait for completion without checking progress.",
        }
        continuation = "After verify, choose the highest-reward terminal action in the revealed progress state; otherwise apply the selected direct action."
    return history, candidates, continuation


def reference_actions(mechanism: str, state: str) -> tuple[list[str], list[str]]:
    if mechanism == "static_diagnosis":
        direct = f"repair_{state}"
    elif mechanism == "dynamic_evidence":
        direct = f"execute_{state}"
    else:
        direct = "retry" if state == "not_started" else "wait"
    info = INFO_ACTION[mechanism]
    return [direct], [direct, info]


def simulate_once(case: dict[str, Any], action: str, rng: random.Random) -> float:
    mechanism = case["mechanism"]
    params = case["public_context"]["parameters"]
    worlds = case["private_context"]["worlds"]
    draw = rng.random()
    cumulative = 0.0
    state = worlds[-1]["state"]
    for world in worlds:
        cumulative += world["posterior"]
        if draw <= cumulative:
            state = world["state"]
            break
    cost = case["public_context"]["information_cost"]
    if mechanism == "static_diagnosis":
        if action != "inspect":
            return binary_reward(action, state, "repair_")
        accuracy = params["inspection_accuracy"]
        report = state if rng.random() < accuracy else ("B" if state == "A" else "A")
        posterior = report_posterior(params["posterior_A"], accuracy, report)
        terminal = binary_terminal_action(posterior, "repair_")
        return binary_reward(terminal, state, "repair_") - cost
    if mechanism == "dynamic_evidence":
        final = state
        if rng.random() < params["pre_execution_flip_probability"]:
            final = "B" if state == "A" else "A"
        if action != "refresh":
            return binary_reward(action, final, "execute_")
        accuracy = params["refresh_accuracy"]
        report = final if rng.random() < accuracy else ("B" if final == "A" else "A")
        p = params["posterior_current_A"]
        flip = params["pre_execution_flip_probability"]
        final_prior = (1.0 - flip) * p + flip * (1.0 - p)
        posterior = report_posterior(final_prior, accuracy, report)
        terminal = binary_terminal_action(posterior, "execute_")
        return binary_reward(terminal, final, "execute_") - cost
    if action == "verify":
        return max(PROGRESS_REWARDS[state].values()) - cost
    return PROGRESS_REWARDS[state][action]


def attach_mc(case: dict[str, Any]) -> None:
    estimates = {}
    for action in case["candidate_order"]:
        rng = random.Random(stable_seed(MC_SEED, case["case_id"], action))
        estimates[action] = sum(simulate_once(case, action, rng) for _ in range(MC_ROLLOUTS_PER_ACTION)) / MC_ROLLOUTS_PER_ACTION
    case["mc_same_pi"] = {
        "seed": MC_SEED,
        "rollouts_per_action": MC_ROLLOUTS_PER_ACTION,
        "estimates": estimates,
        "max_absolute_error_vs_exact": max(abs(estimates[a] - case["gold"]["q_pi"][a]) for a in estimates),
    }


def build_case(mechanism: str, stratum: str, index: int) -> dict[str, Any]:
    rng = random.Random(stable_seed("case", mechanism, stratum, index))
    for attempt in range(20_000):
        params, states, builder = base_parameters(mechanism, rng)
        no_cost = builder(0.0)
        aggregated_zero = aggregate_values(states, no_cost)
        info = INFO_ACTION[mechanism]
        other = max(value for action, value in aggregated_zero.items() if action != info)
        gross_gap = aggregated_zero[info] - other
        target = desired_gap(stratum, gross_gap, rng)
        if target is None:
            continue
        cost = gross_gap - target
        if not 0.05 <= cost <= 8.0:
            continue
        cost = round(cost, 6)
        per_world = builder(cost)
        aggregated = aggregate_values(states, per_world)
        gap = aggregated[info] - max(value for action, value in aggregated.items() if action != info)
        if stratum == "valuable" and gap < 0.5:
            continue
        if stratum == "unhelpful" and gap > -0.5:
            continue
        if stratum == "boundary" and abs(gap) >= 0.5:
            continue
        config_for_id = {"mechanism": mechanism, "stratum": stratum, "index": index, "parameters": params, "cost": cost}
        case_id = f"{mechanism[:3]}-{sha256_bytes(canonical_json(config_for_id).encode())[:16]}"
        history, candidate_descriptions, continuation = descriptions(mechanism, params, cost)
        candidate_order = list(aggregated)
        random.Random(stable_seed("candidate-order", case_id)).shuffle(candidate_order)
        worlds = []
        for state, posterior in states:
            refs, full_refs = reference_actions(mechanism, state)
            worlds.append({
                "state": state,
                "posterior": posterior,
                "q_pi": per_world[state],
                "reference_actions": refs,
                "full_reference_actions": full_refs,
            })
        optimum = max(aggregated.values())
        outcomes = candidate_outcomes(mechanism, params, cost, states)
        case = {
            "case_id": case_id,
            "mechanism": mechanism,
            "stratum": stratum,
            "split": None,
            "full_reference_control": False,
            "public_context": {
                "history": history,
                "parameters": params,
                "information_cost": cost,
                "reward_rule": {"correct": REWARD_CORRECT, "wrong": REWARD_WRONG, "pause": 0.0},
                "progress_reward_table": PROGRESS_REWARDS if mechanism == "progress_uncertainty" else None,
                "continuation_pi": continuation,
            },
            "candidate_order": candidate_order,
            "candidates": {action: candidate_descriptions[action] for action in candidate_order},
            "private_context": {"worlds": worlds},
            "gold": {
                "q_pi": aggregated,
                "optimal_actions": [action for action, value in aggregated.items() if abs(value - optimum) <= 1e-10],
                "information_value_gap": gap,
                "reward_span": REWARD_SPAN,
                "candidate_outcomes_under_pi": outcomes,
                "normalization": "Fixed +10 minus -30 task reward span; frozen before teacher outputs.",
            },
            "reference_rule": {
                "incomplete": "Privileged minimal direct action selected from the hidden state; it skips verification because the reference generator already sees that state.",
                "full": "The same hidden-state success rule plus the explicitly available verification workflow (public sensor reliability >=0.8, or exact progress verification). No history-only oracle answer is added.",
            },
        }
        attach_mc(case)
        return case
    raise RuntimeError(f"could not construct {mechanism}/{stratum}/{index}")


def assign_splits_and_controls(cases: list[dict[str, Any]]) -> None:
    for mechanism in MECHANISMS:
        for stratum in STRATA:
            group = [case for case in cases if case["mechanism"] == mechanism and case["stratum"] == stratum]
            if len(group) != 40:
                raise AssertionError((mechanism, stratum, len(group)))
            rng = random.Random(stable_seed(SPLIT_SEED, "split", mechanism, stratum))
            rng.shuffle(group)
            for position, case in enumerate(group):
                case["split"] = "dev" if position < 8 else "audit"
            audit = group[8:]
            control_rng = random.Random(stable_seed(SPLIT_SEED, "full-reference", mechanism, stratum))
            control_rng.shuffle(audit)
            for case in audit[:8]:
                case["full_reference_control"] = True


def validate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    counts = Counter()
    ids = set()
    mc_errors = []
    mc_signed_errors = []
    for case in cases:
        case_id = case.get("case_id")
        if case_id in ids:
            errors.append(f"duplicate case_id {case_id}")
        ids.add(case_id)
        mechanism, stratum, split = case["mechanism"], case["stratum"], case["split"]
        counts[f"mechanism:{mechanism}"] += 1
        counts[f"stratum:{mechanism}:{stratum}"] += 1
        counts[f"split:{split}"] += 1
        counts[f"cell:{mechanism}:{stratum}:{split}"] += 1
        if case["full_reference_control"]:
            counts[f"full:{mechanism}"] += 1
            counts[f"full:{mechanism}:{stratum}"] += 1
            if split != "audit":
                errors.append(f"dev case marked full control: {case_id}")
        worlds = case["private_context"]["worlds"]
        if len(worlds) != STATE_COUNTS[mechanism]:
            errors.append(f"wrong state count: {case_id}")
        if abs(sum(world["posterior"] for world in worlds) - 1.0) > 1e-10:
            errors.append(f"posterior does not sum to one: {case_id}")
        if set(case["candidate_order"]) != set(case["candidates"]) or len(case["candidate_order"]) != 3:
            errors.append(f"candidate mismatch: {case_id}")
        q_pi = case["gold"]["q_pi"]
        for action in case["candidate_order"]:
            recomputed = sum(world["posterior"] * world["q_pi"][action] for world in worlds)
            if abs(recomputed - q_pi[action]) > 1e-9:
                errors.append(f"posterior aggregation mismatch: {case_id}/{action}")
        gap = case["gold"]["information_value_gap"]
        if stratum == "valuable" and gap < 0.5 - 1e-9:
            errors.append(f"valuable threshold violation: {case_id}")
        if stratum == "unhelpful" and gap > -0.5 + 1e-9:
            errors.append(f"unhelpful threshold violation: {case_id}")
        if stratum == "boundary" and abs(gap) >= 0.5:
            errors.append(f"boundary threshold violation: {case_id}")
        if case["gold"]["reward_span"] != REWARD_SPAN:
            errors.append(f"reward span mismatch: {case_id}")
        outcomes = case["gold"].get("candidate_outcomes_under_pi", {})
        if set(outcomes) != set(case["candidate_order"]):
            errors.append(f"candidate outcome mismatch: {case_id}")
        for action, outcome in outcomes.items():
            if not 0.0 <= outcome["harm_probability"] <= 1.0 or not 0.0 <= outcome["pause_probability"] <= 1.0:
                errors.append(f"invalid outcome probability: {case_id}/{action}")
        mc_errors.append(case["mc_same_pi"]["max_absolute_error_vs_exact"])
        mc_signed_errors.extend(
            case["mc_same_pi"]["estimates"][action] - q_pi[action] for action in case["candidate_order"]
        )
    expected = {
        "split:dev": 72,
        "split:audit": 288,
        **{f"mechanism:{m}": 120 for m in MECHANISMS},
        **{f"stratum:{m}:{s}": 40 for m in MECHANISMS for s in STRATA},
        **{f"cell:{m}:{s}:dev": 8 for m in MECHANISMS for s in STRATA},
        **{f"cell:{m}:{s}:audit": 32 for m in MECHANISMS for s in STRATA},
        **{f"full:{m}": 24 for m in MECHANISMS},
        **{f"full:{m}:{s}": 8 for m in MECHANISMS for s in STRATA},
    }
    for key, value in expected.items():
        if counts[key] != value:
            errors.append(f"count {key}={counts[key]} expected {value}")
    if len(cases) != 360:
        errors.append(f"total cases={len(cases)} expected 360")
    if errors:
        raise ValueError("case validation failed:\n" + "\n".join(errors[:50]))
    return {
        "valid": True,
        "independent_root_cases": len(cases),
        "counts": dict(sorted(counts.items())),
        "same_pi_mc": {
            "seed": MC_SEED,
            "rollouts_per_case_action": MC_ROLLOUTS_PER_ACTION,
            "num_nested_rollouts": len(cases) * 3 * MC_ROLLOUTS_PER_ACTION,
            "mean_case_max_absolute_error": statistics.fmean(mc_errors),
            "max_case_max_absolute_error": max(mc_errors),
            "mean_signed_action_error": statistics.fmean(mc_signed_errors),
            "mean_absolute_action_error": statistics.fmean(abs(error) for error in mc_signed_errors),
            "root_mean_squared_action_error": math.sqrt(statistics.fmean(error * error for error in mc_signed_errors)),
            "independent_sample_increment": 0,
        },
    }


def generate_cases() -> list[dict[str, Any]]:
    cases = [build_case(mechanism, stratum, index) for mechanism in MECHANISMS for stratum in STRATA for index in range(40)]
    assign_splits_and_controls(cases)
    cases.sort(key=lambda item: (MECHANISMS.index(item["mechanism"]), STRATA.index(item["stratum"]), item["case_id"]))
    validate_cases(cases)
    return cases


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_dataset(output_dir: Path, model_commit: str | None = None) -> dict[str, Any]:
    cases = generate_cases()
    validation = validate_cases(cases)
    cases_payload = "".join(canonical_json(case) + "\n" for case in cases).encode("utf-8")
    cases_path = output_dir / "cases.jsonl"
    manifest_path = output_dir / "manifest.json"
    project = Path(__file__).resolve().parent
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "G1 finite-state gold and frozen root cases; no teacher inference or model training",
        "data_frozen": True,
        "audit_started": False,
        "split_seed": SPLIT_SEED,
        "bootstrap": {"replicates": 2000, "seed": 23, "unit": "root_case", "stratify_by": "mechanism"},
        "primary_metric": "case-level normalized regret using fixed reward span 40",
        "primary_contrast": "mean(r_PR - r_PR-IC) on 288 locked audit root cases",
        "invalid_output_fallback": "all three candidates tied uniformly; invalid rate also reported separately",
        "tie_rule": "uniform over all maximum-score candidate actions",
        "gate": {
            "valuable_PR_rate_at_regret_ge_0.05": 0.15,
            "minimum_mechanisms_with_problem": 2,
            "minimum_mean_paired_improvement": 0.01,
            "confidence_interval_must_support_positive": True,
            "full_reference_must_not_explain_all": True,
        },
        "teacher": {
            "target_model": "Qwen/Qwen3-4B-Instruct-2507",
            "requested_revision": model_commit or "UNRESOLVED",
            "resolved_commit": model_commit,
            "prompt_version": "v1-initial-unfrozen",
            "max_new_tokens": 128,
            "method_label": "VPR-style adaptation; not a claim of original-model reproduction",
        },
        "reference_completeness": {
            "selected_cases": 72,
            "selection": "8 locked audit cases per mechanism x value stratum, fixed with split seed 17",
            "rule": "Complete only the same hidden-state reference workflow with qualifying verification actions; never insert the history-only gold answer.",
        },
        "continuation": {
            "name": "fixed analytic history-only Bayes controller pi",
            "gold": "exact complete hidden-state enumeration with true posterior weights",
            "mc": "seeded CPU rollout of exactly the same pi; nested rollouts are not independent cases",
        },
        "call_budget": {"teacher_hard_limit": 4272, "frozen_first_pass_plus_full_reference": 3624, "optional_llm_mc_limit": 2592},
        "validation": validation,
        "hashes": {
            "cases_sha256": sha256_bytes(cases_payload),
            "audit_cases_py_sha256": sha256_file(project / "audit_cases.py"),
            "audit_teacher_py_sha256": sha256_file(project / "audit_teacher.py"),
            "core_py_sha256": sha256_file(project / "core.py"),
        },
        "files": {"cases": "data/audit/cases.jsonl", "manifest": "data/audit/manifest.json"},
    }
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(cases_payload) + len(manifest_payload) > 128 * 1024 * 1024:
        raise RuntimeError("new audit data would exceed the 128 MiB budget")
    atomic_write(cases_path, cases_payload)
    atomic_write(manifest_path, manifest_payload)
    return manifest


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data" / "audit")
    generate.add_argument("--model-commit", default=None)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--cases", type=Path, default=Path(__file__).parent / "data" / "audit" / "cases.jsonl")
    args = parser.parse_args()
    if args.command == "generate":
        result = write_dataset(args.output_dir, args.model_commit)
    else:
        result = validate_cases(load_cases(args.cases))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
