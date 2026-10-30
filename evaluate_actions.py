"""Evaluate a model's FIRST inspection decision on the finite toy problem.

The LLM selects only skip/quick/precise.  ``core.information_value`` then
integrates inspection reports and state changes exactly, with the SAME
Bayes-optimal, observation-limited terminal controller for every model.
This is NOT an end-to-end LLM rollout, an empirical safety rate, or evidence
that the LLM can perform the terminal reasoning itself.

Requires a CUDA GPU and separately installed torch, transformers, and
optionally peft.  Loading happens only when main() is run; importing this
file or compiling it does not download weights or initialize a GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any

from core import Inspection, Scenario, TIE_TOLERANCE, first_actions, information_value
from generate_pairs import TEMPLATES, action_text, format_conditional_prompt, format_prompt

IMPLICIT_REWARD_BETA = 0.1


SCOPE = "LLM first decision; exact observation-limited Bayesian terminal controller"
INVALID_OUTPUT_UTILITY = 0.0  # Explicit safe-abort convention, not a parsed tool call.


def load_scenarios(
    manifest: Path, split: str, regime: str | None, maximum: int | None, seed: int
) -> tuple[list[tuple[dict[str, Any], Scenario]], int]:
    """Read unique scenarios; subsample scenarios, never preference pairs."""
    selected: list[tuple[dict[str, Any], Scenario]] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != split or (regime is not None and row["regime"] != regime):
                continue
            config = dict(row["config"])
            config["inspections"] = tuple(Inspection(**item) for item in config["inspections"])
            scenario = Scenario(**config)
            if row["scenario_id"] != scenario.scenario_id:
                raise ValueError(f"Scenario ID mismatch at {manifest}:{line_number}")
            if scenario.scenario_id in seen:
                raise ValueError(f"Duplicate scenario ID at {manifest}:{line_number}")
            if scenario.terminal_visible:
                raise ValueError("This prompt/evaluator targets partial-observation scenarios only")
            if not set(first_actions(scenario)).issubset({"skip", "quick", "precise"}):
                raise ValueError("This evaluator supports only skip, quick, and precise")
            seen.add(scenario.scenario_id)
            selected.append((row, scenario))
    eligible_count = len(selected)
    if eligible_count == 0:
        raise ValueError(f"No scenarios match split={split!r}, regime={regime!r}")
    # A fixed sample is reused across base/adapted models. The unit is a
    # scenario, so paraphrases cannot inflate sample size.
    selected.sort(key=lambda item: item[1].scenario_id)
    if maximum is not None and maximum < eligible_count:
        selected = random.Random(seed).sample(selected, maximum)
        selected.sort(key=lambda item: item[1].scenario_id)
    return selected, eligible_count


def parse_action(text: str, allowed: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Require exactly one JSON call; do not silently repair or guess outputs."""
    try:
        call = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        return None, "expected_exact_tool_and_arguments_keys"
    arguments = call["arguments"]
    if not isinstance(arguments, dict):
        return None, "arguments_must_be_object"
    if call["tool"] == "skip_inspection" and arguments == {}:
        return ("skip", None) if "skip" in allowed else (None, "unavailable_action")
    if call["tool"] == "inspect" and set(arguments) == {"test"}:
        action = arguments["test"]
        if isinstance(action, str) and action in allowed and action in {"quick", "precise"}:
            return action, None
        return None, "unavailable_inspection"
    return None, "unknown_tool_or_invalid_arguments"


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["format_valid"]]
    return {
        "n_scenarios": len(records),
        "n_format_valid": len(valid),
        "format_valid_rate": len(valid) / len(records) if records else None,
        "action_counts": dict(Counter(record["action"] if record["format_valid"] else "INVALID" for record in records)),
        "mean_oracle_expected_return": average([record["oracle_expected_return"] for record in records]),
        "mean_expected_return_valid_only": average([record["expected_return"] for record in valid]),
        "mean_decision_regret_valid_only": average([record["decision_regret"] for record in valid]),
        "optimal_action_rate_valid_only": average([float(record["optimal_action"]) for record in valid]),
        "mean_utility_invalid_as_abort": average([record["utility_invalid_as_abort"] for record in records]),
        "mean_regret_invalid_as_abort": average([record["regret_invalid_as_abort"] for record in records]),
        "mean_generation_seconds": average([record["generation_seconds"] for record in records]),
        "mean_generated_tokens": average([float(record["generated_tokens"]) for record in records]),
        "mean_prompt_tokens": average([float(record["prompt_tokens"]) for record in records]),
    }


def tokenize_boundary(tokenizer: Any, prompt: str, action: str) -> tuple[list[int], list[int]]:
    """Exactly train_dpo.py's tokenize_action_pair boundary: no EOS/template suffix is scored."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("Empty tokenized prompt")
    full_ids = tokenizer(prompt + action, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer merges across the prompt/action boundary; change the prompt delimiter.")
    completion = full_ids[len(prompt_ids) :]
    if not completion:
        raise ValueError("Empty tokenized action")
    return prompt_ids, completion


def sequence_logprob(model: Any, device: Any, prompt_ids: list[int], completion_ids: list[int]) -> float:
    """Sum of log p(token_t | prefix) over action-content tokens only, FP32, no length normalization."""
    import torch

    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids).logits.float()
    start = len(prompt_ids)
    total = 0.0
    for offset, token_id in enumerate(completion_ids):
        position = start + offset - 1
        log_probs = torch.log_softmax(logits[0, position], dim=-1)
        total += log_probs[token_id].item()
    return total


def tie_aware_readout(criterion: dict[str, float], oracle: dict[str, float], tol: float = TIE_TOLERANCE) -> dict[str, Any]:
    """A degenerate tie (e.g. BASE's identically-zero implicit reward) is a genuine
    tie, not an insertion-order default: realized return is the expectation under a
    uniform distribution over the full tied-argmax set."""
    best_criterion = max(criterion.values())
    tied = sorted(action for action, value in criterion.items() if best_criterion - value <= tol)
    best_oracle = max(oracle.values())
    sorted_criterion = sorted(criterion.values(), reverse=True)
    margin = sorted_criterion[0] - sorted_criterion[1] if len(sorted_criterion) > 1 else None
    expected_return = statistics.fmean(oracle[action] for action in tied)
    return {
        "tied_actions": tied,
        "margin": margin,
        "expected_return": expected_return,
        "regret": max(0.0, best_oracle - expected_return),
        "optimal_action_rate": statistics.fmean(1.0 if best_oracle - oracle[action] <= tol else 0.0 for action in tied),
    }


def strict_pair_accuracy(criterion: dict[str, float], oracle: dict[str, float], tol: float = TIE_TOLERANCE) -> tuple[int, int]:
    """Count oracle pairs with a strict preference the criterion ranks the same way."""
    correct = 0
    total = 0
    for left, right in itertools.combinations(sorted(oracle), 2):
        oracle_diff = oracle[left] - oracle[right]
        if abs(oracle_diff) <= tol:
            continue
        total += 1
        if (criterion[left] - criterion[right] > 0) == (oracle_diff > 0):
            correct += 1
    return correct, total


def load_reused_greedy(path: Path) -> dict[str, dict[str, Any]]:
    reused = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            reused[row["scenario_id"]] = row
    return reused


def summarize_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_scenarios": len(records)}
    greedy_valid = [record for record in records if record["greedy"]["action"] is not None]
    summary["greedy"] = {
        "format_valid_rate": len(greedy_valid) / len(records) if records else None,
        "action_counts": dict(Counter(record["greedy"]["action"] or "INVALID" for record in records)),
        "mean_regret_valid_only": average([record["greedy"]["regret"] for record in greedy_valid]),
        "optimal_action_rate_valid_only": average([float(record["greedy"]["optimal_action"]) for record in greedy_valid]),
    }
    for key in ("policy", "implicit"):
        readout_key = f"{key}_readout"
        summary[key] = {
            "action_counts": dict(Counter(",".join(record[readout_key]["tied_actions"]) for record in records)),
            "mean_regret": average([record[readout_key]["regret"] for record in records]),
            "optimal_action_rate": average([record[readout_key]["optimal_action_rate"] for record in records]),
            "mean_margin": average([record[readout_key]["margin"] for record in records if record[readout_key]["margin"] is not None]),
            "strict_pair_accuracy": (
                sum(record[f"{key}_pair_accuracy"]["correct"] for record in records)
                / sum(record[f"{key}_pair_accuracy"]["total"] for record in records)
                if sum(record[f"{key}_pair_accuracy"]["total"] for record in records) > 0
                else None
            ),
        }
    summary["mean_oracle_margin"] = average([record["oracle_margin"] for record in records if record["oracle_margin"] is not None])
    return summary


def run_score_mode(args: argparse.Namespace) -> None:
    """Step A / G3 candidate-level scoring: L_policy, L_base, R_implicit over the
    exact train_dpo.py tokenization boundary, for all 3 candidates per scenario,
    across train+validation+test. Test-split greedy actions are reused from an
    existing G2R run (never regenerated); train/validation greedy actions are
    freshly generated with the identical greedy decoding convention."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and any(path.exists() for path in (predictions_path, summary_path)):
        raise SystemExit("Output files already exist; choose another --output-dir or pass --overwrite")

    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise SystemExit(f"Requested CUDA device is unavailable: {args.device}")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The requested GPU must support BF16; this entry point does not silently change precision")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=False, local_files_only=True,
    ).to(device)
    peft_version = None
    has_adapter = bool(args.adapter)
    if has_adapter:
        import peft
        from peft import PeftModel

        peft_version = peft.__version__
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).to(device)
    model.eval()

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos_id[0] if isinstance(eos_id, list) else eos_id)
    generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, eos_token_id=eos_id, pad_token_id=pad_id)

    reused_greedy = load_reused_greedy(args.reuse_greedy_dir / "predictions.jsonl") if args.reuse_greedy_dir else {}

    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and summary_path.exists():
        summary_path.unlink()
    all_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8") as stream:
        for split in ("train", "validation", "test"):
            scenarios, _ = load_scenarios(args.manifest, split, None, None, args.seed)
            for index, (row, scenario) in enumerate(scenarios, start=1):
                messages = format_prompt(scenario, args.variant)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                if split == "test" and scenario.scenario_id in reused_greedy:
                    reused = reused_greedy[scenario.scenario_id]
                    greedy = {
                        "action": reused["action"], "format_valid": reused["format_valid"],
                        "format_error": reused["format_error"], "raw_output": reused["raw_output"],
                        "source": str(args.reuse_greedy_dir),
                    }
                else:
                    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
                    prompt_length = int(inputs["input_ids"].shape[1])
                    with torch.inference_mode():
                        output = model.generate(**inputs, generation_config=generation_config, use_model_defaults=False)
                    response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
                    action, error = parse_action(response, first_actions(scenario))
                    greedy = {"action": action, "format_valid": action is not None, "format_error": error, "raw_output": response, "source": "fresh_generate"}

                oracle_values = {candidate: information_value(scenario, candidate) for candidate in first_actions(scenario)}
                best_oracle = max(oracle_values.values())
                sorted_oracle = sorted(oracle_values.values(), reverse=True)
                oracle_margin = sorted_oracle[0] - sorted_oracle[1] if len(sorted_oracle) > 1 else None
                greedy_return = oracle_values[greedy["action"]] if greedy["action"] is not None else None
                greedy["expected_return"] = greedy_return
                greedy["regret"] = max(0.0, best_oracle - greedy_return) if greedy_return is not None else None
                greedy["optimal_action"] = (best_oracle - greedy_return <= TIE_TOLERANCE) if greedy_return is not None else None

                policy_logp: dict[str, float] = {}
                base_logp: dict[str, float] = {}
                implicit_reward: dict[str, float] = {}
                for candidate in first_actions(scenario):
                    prompt_ids, completion_ids = tokenize_boundary(tokenizer, prompt, action_text(candidate))
                    policy_logp[candidate] = sequence_logprob(model, device, prompt_ids, completion_ids)
                    if has_adapter:
                        with model.disable_adapter():
                            base_logp[candidate] = sequence_logprob(model, device, prompt_ids, completion_ids)
                    else:
                        base_logp[candidate] = policy_logp[candidate]
                    implicit_reward[candidate] = IMPLICIT_REWARD_BETA * (policy_logp[candidate] - base_logp[candidate])

                policy_readout = tie_aware_readout(policy_logp, oracle_values)
                implicit_readout = tie_aware_readout(implicit_reward, oracle_values)
                policy_correct, policy_total = strict_pair_accuracy(policy_logp, oracle_values)
                implicit_correct, implicit_total = strict_pair_accuracy(implicit_reward, oracle_values)

                record = {
                    "scenario_id": scenario.scenario_id, "split": row["split"], "regime": row["regime"], "variant": args.variant,
                    "oracle_values": oracle_values, "oracle_margin": oracle_margin,
                    "greedy": greedy,
                    "policy_logp": policy_logp, "base_logp": base_logp, "implicit_reward": implicit_reward,
                    "policy_readout": policy_readout, "implicit_readout": implicit_readout,
                    "policy_pair_accuracy": {"correct": policy_correct, "total": policy_total},
                    "implicit_pair_accuracy": {"correct": implicit_correct, "total": implicit_total},
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                all_records.append(record)
                if index == 1 or index % 100 == 0 or index == len(scenarios):
                    print(json.dumps({"split": split, "completed": index, "total": len(scenarios)}), flush=True)

    regimes = sorted({record["regime"] for record in all_records})
    splits = sorted({record["split"] for record in all_records})
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "G3 Step A: 3-candidate log-probability scoring + implicit DPO reward readout; test greedy reused, train/validation greedy freshly generated, all with identical greedy decoding",
        "model": args.model, "adapter": args.adapter, "requested_revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "manifest": str(args.manifest.resolve()), "prompt_variant": args.variant,
        "implicit_reward_beta": IMPLICIT_REWARD_BETA,
        "reuse_greedy_dir": str(args.reuse_greedy_dir) if args.reuse_greedy_dir else None,
        "device": args.device, "gpu_name": torch.cuda.get_device_name(device), "precision": "bfloat16",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_version},
        "elapsed_evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "overall": summarize_score(all_records),
        "by_split": {split: summarize_score([record for record in all_records if record["split"] == split]) for split in splits},
        "by_regime": {regime: summarize_score([record for record in all_records if record["regime"] == regime]) for regime in regimes},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(predictions_path), "summary": str(summary_path), "overall": summary["overall"]}, indent=2))


def load_conditional_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the G3 conditional scenario manifest (family-level splits, true ranking).

    The conditional manifest carries no legacy ``regime`` field; the split and
    family_id replace it for reporting.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            config = dict(row["config"])
            config["inspections"] = tuple(Inspection(**item) for item in config["inspections"])
            scenario = Scenario(**config)
            if row["scenario_id"] != scenario.scenario_id:
                raise ValueError(f"Scenario ID mismatch at {path}:{line_number}")
            if scenario.scenario_id in seen:
                raise ValueError(f"Duplicate scenario ID at {path}:{line_number}")
            if scenario.terminal_visible:
                raise ValueError("This evaluator targets partial-observation scenarios only")
            if not set(first_actions(scenario)).issubset({"skip", "quick", "precise"}):
                raise ValueError("This evaluator supports only skip, quick, and precise")
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError(f"Unexpected conditional split at {path}:{line_number}")
            seen.add(scenario.scenario_id)
            rows.append({"row": row, "scenario": scenario})
    if not rows:
        raise ValueError(f"No conditional scenarios found in {path}")
    rows.sort(key=lambda item: item["scenario"].scenario_id)
    return rows


def load_probe_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load both endpoints of each fixed-cost counterfactual probe pair."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            for side in ("left", "right"):
                item = pair[side]
                config = dict(item["config"])
                config["inspections"] = tuple(Inspection(**entry) for entry in config["inspections"])
                scenario = Scenario(**config)
                if item["scenario_id"] != scenario.scenario_id:
                    raise ValueError(f"Probe scenario ID mismatch at {path}:{line_number}")
                if scenario.scenario_id in seen:
                    raise ValueError(f"Duplicate probe scenario at {path}:{line_number}")
                seen.add(scenario.scenario_id)
                entries.append({
                    "pair_id": pair["pair_id"], "group": pair["group"], "varied_field": pair["varied_field"],
                    "side": side, "scenario": scenario,
                    "file_optimal_action": item["optimal_action"], "file_margin": item["margin"],
                })
    if not entries:
        raise ValueError(f"No probe scenarios found in {path}")
    return entries


def run_conditional_mode(args: argparse.Namespace) -> None:
    """Evaluate BASE/TRUE/SHUFFLE on the G3 conditional splits and fixed-cost probe.

    Three readouts per scenario: deployed greedy generation, candidate-sequence
    MAP over L_policy (the 3 canonical action texts), and implicit DPO reward
    R_implicit = beta * (L_policy - L_base) with L_base from disable_adapter().
    All candidate scoring uses the same train_dpo.py token boundary as Step A.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and any(path.exists() for path in (predictions_path, summary_path)):
        raise SystemExit("Output files already exist; choose another --output-dir or pass --overwrite")

    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise SystemExit(f"Requested CUDA device is unavailable: {args.device}")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The requested GPU must support BF16; this entry point does not silently change precision")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=False, local_files_only=True,
    ).to(device)
    peft_version = None
    has_adapter = bool(args.adapter)
    if has_adapter:
        import peft
        from peft import PeftModel

        peft_version = peft.__version__
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).to(device)
    model.eval()

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos_id[0] if isinstance(eos_id, list) else eos_id)
    generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, eos_token_id=eos_id, pad_token_id=pad_id)

    conditional_rows = load_conditional_manifest(args.conditional_manifest)
    probe_entries = load_probe_scenarios(args.probe_file)

    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and summary_path.exists():
        summary_path.unlink()
    all_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    def evaluate_entry(scenario: Scenario, split: str, extra: dict[str, Any]) -> dict[str, Any]:
        messages = format_conditional_prompt(scenario, args.variant)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        prompt_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation_config, use_model_defaults=False)
        response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        action, error = parse_action(response, first_actions(scenario))

        oracle_values = {candidate: information_value(scenario, candidate) for candidate in first_actions(scenario)}
        best_oracle = max(oracle_values.values())
        sorted_oracle = sorted(oracle_values.values(), reverse=True)
        oracle_margin = sorted_oracle[0] - sorted_oracle[1] if len(sorted_oracle) > 1 else None
        greedy_return = oracle_values[action] if action is not None else None
        greedy = {
            "action": action, "format_valid": action is not None, "format_error": error, "raw_output": response,
            "expected_return": greedy_return,
            "regret": max(0.0, best_oracle - greedy_return) if greedy_return is not None else None,
            "optimal_action": (best_oracle - greedy_return <= TIE_TOLERANCE) if greedy_return is not None else None,
        }

        policy_logp: dict[str, float] = {}
        base_logp: dict[str, float] = {}
        implicit_reward: dict[str, float] = {}
        for candidate in first_actions(scenario):
            prompt_ids, completion_ids = tokenize_boundary(tokenizer, prompt, action_text(candidate))
            policy_logp[candidate] = sequence_logprob(model, device, prompt_ids, completion_ids)
            if has_adapter:
                with model.disable_adapter():
                    base_logp[candidate] = sequence_logprob(model, device, prompt_ids, completion_ids)
            else:
                base_logp[candidate] = policy_logp[candidate]
            implicit_reward[candidate] = IMPLICIT_REWARD_BETA * (policy_logp[candidate] - base_logp[candidate])

        policy_readout = tie_aware_readout(policy_logp, oracle_values)
        implicit_readout = tie_aware_readout(implicit_reward, oracle_values)
        policy_correct, policy_total = strict_pair_accuracy(policy_logp, oracle_values)
        implicit_correct, implicit_total = strict_pair_accuracy(implicit_reward, oracle_values)
        record = {
            "scenario_id": scenario.scenario_id, "split": split, "variant": args.variant,
            **extra,
            "oracle_values": oracle_values, "oracle_margin": oracle_margin,
            "greedy": greedy,
            "policy_logp": policy_logp, "base_logp": base_logp, "implicit_reward": implicit_reward,
            "policy_readout": policy_readout, "implicit_readout": implicit_readout,
            "policy_pair_accuracy": {"correct": policy_correct, "total": policy_total},
            "implicit_pair_accuracy": {"correct": implicit_correct, "total": implicit_total},
        }
        return record

    with predictions_path.open("w", encoding="utf-8") as stream:
        for index, item in enumerate(conditional_rows, start=1):
            row = item["row"]
            record = evaluate_entry(item["scenario"], row["split"], {"family_id": row["family_id"], "true_ranking": row["true_ranking"]})
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            all_records.append(record)
            if index == 1 or index % 100 == 0 or index == len(conditional_rows):
                print(json.dumps({"split": row["split"], "completed": index, "total": len(conditional_rows)}), flush=True)
        for index, entry in enumerate(probe_entries, start=1):
            record = evaluate_entry(entry["scenario"], "probe", {
                "pair_id": entry["pair_id"], "group": entry["group"], "varied_field": entry["varied_field"],
                "side": entry["side"], "file_optimal_action": entry["file_optimal_action"], "file_margin": entry["file_margin"],
            })
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            all_records.append(record)
            if index == 1 or index % 32 == 0 or index == len(probe_entries):
                print(json.dumps({"split": "probe", "completed": index, "total": len(probe_entries)}), flush=True)

    splits = ["train", "validation", "test", "probe"]
    groups = sorted({record["group"] for record in all_records if record["split"] == "probe"})
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "G3 conditional evaluation: deployed greedy generation, candidate-sequence MAP over L_policy, and implicit DPO rerank over R=beta*(L_policy-L_base); identical prompt/controller rules for train and evaluation",
        "model": args.model, "adapter": args.adapter, "requested_revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "conditional_manifest": str(args.conditional_manifest.resolve()),
        "probe_file": str(args.probe_file.resolve()),
        "prompt_variant": args.variant, "implicit_reward_beta": IMPLICIT_REWARD_BETA,
        "device": args.device, "gpu_name": torch.cuda.get_device_name(device), "precision": "bfloat16",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_version},
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens, "use_model_defaults": False},
        "elapsed_evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "overall": summarize_score(all_records),
        "by_split": {split: summarize_score([record for record in all_records if record["split"] == split]) for split in splits},
        "by_probe_group": {group: summarize_score([record for record in all_records if record.get("group") == group]) for group in groups},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(predictions_path), "summary": str(summary_path), "overall": summary["overall"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("generate", "score", "conditional"), default="generate", help="generate: original greedy-only eval. score: G3 Step A candidate scoring. conditional: G3 new-split + probe evaluation.")
    parser.add_argument("--reuse-greedy-dir", type=Path, default=None, help="score mode: existing run dir whose predictions.jsonl supplies test-split greedy actions")
    parser.add_argument("--conditional-manifest", type=Path, default=Path(__file__).parent / "data" / "g3_conditional" / "scenario_manifest.jsonl", help="conditional mode: G3 scenario manifest")
    parser.add_argument("--probe-file", type=Path, default=Path(__file__).parent / "data" / "g3_conditional" / "fixed_cost_probes.jsonl", help="conditional mode: fixed-cost counterfactual probe pairs")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter directory or Hugging Face ID")
    parser.add_argument("--revision", default=None, help="Optional base model revision/commit for reproducibility")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data" / "scenario_manifest.jsonl")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--regime", default=None, help="Optional exact regime, e.g. ood_drift")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Deterministic scenario subsample for a small GPU profile")
    parser.add_argument("--variant", type=int, choices=range(len(TEMPLATES)), default=0, help="Exactly one prompt variant per scenario")
    parser.add_argument("--seed", type=int, default=17, help="Scenario subsampling seed; decoding is greedy")
    parser.add_argument("--device", default="cuda:0", help="One explicit CUDA device, e.g. cuda:0; no automatic device mapping")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "action_eval")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace predictions.jsonl and summary.json in output-dir")
    args = parser.parse_args()
    if args.max_scenarios is not None and args.max_scenarios < 1:
        parser.error("--max-scenarios must be positive")
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("--max-new-tokens must be in [1, 64]")
    if re.fullmatch(r"cuda:\d+", args.device) is None:
        parser.error("--device must identify one CUDA GPU explicitly, e.g. cuda:0")
    if args.mode == "score":
        run_score_mode(args)
        return
    if args.mode == "conditional":
        run_conditional_mode(args)
        return
    scenarios, eligible_count = load_scenarios(args.manifest, args.split, args.regime, args.max_scenarios, args.seed)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and any(path.exists() for path in (predictions_path, summary_path)):
        parser.error("Output files already exist; choose another --output-dir or pass --overwrite")

    # Lazy imports keep the CPU oracle, parser, and syntax checks independent
    # of GPU libraries. No model is loaded until the script is executed.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        parser.error(f"Requested CUDA device is unavailable: {args.device}")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        parser.error("The requested GPU must support BF16; this entry point does not silently change precision")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    # Explicit placement on one device; do not use device_map='auto'.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to(device)
    peft_version = None
    if args.adapter:
        import peft
        from peft import PeftModel

        peft_version = peft.__version__
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).to(device)
    model.eval()
    eos_id = model.generation_config.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Remove an old summary before an explicitly authorized overwrite. If a
    # run fails midway, it must not appear paired with a previous run's summary.
    if args.overwrite and summary_path.exists():
        summary_path.unlink()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8") as stream:
        for index, (row, scenario) in enumerate(scenarios, start=1):
            messages = format_prompt(scenario, args.variant)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            prompt_length = int(inputs["input_ids"].shape[1])
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            with torch.inference_mode():
                # use_model_defaults=False must be passed to generate() itself, not
                # the GenerationConfig constructor. Without it, transformers treats
                # do_sample=False as an untouched dataclass default (False already
                # equals GenerationConfig's own default) and silently re-merges the
                # model's own generation_config.json (do_sample=True, temperature=0.7,
                # ...) back in. Confirmed via the "modified to match model-specific
                # defaults" warning present in every prior eval run's log.
                output = model.generate(
                    **inputs, generation_config=generation_config, use_model_defaults=False
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - before
            new_tokens = output[0, prompt_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            action, error = parse_action(response, first_actions(scenario))
            values = {candidate: information_value(scenario, candidate) for candidate in first_actions(scenario)}
            optimum = max(values.values())
            expected_return = information_value(scenario, action) if action is not None else None
            regret = max(0.0, optimum - expected_return) if expected_return is not None else None
            abort_adjusted_utility = expected_return if expected_return is not None else INVALID_OUTPUT_UTILITY
            record = {
                "scenario_id": scenario.scenario_id,
                "split": row["split"],
                "regime": row["regime"],
                "variant": args.variant,
                "scenario": scenario.to_dict(),
                "raw_output": response,
                "format_valid": action is not None,
                "format_error": error,
                "action": action,
                "expected_return": expected_return,
                "decision_regret": regret,
                "optimal_action": regret <= TIE_TOLERANCE if regret is not None else None,
                "oracle_expected_return": optimum,
                "information_values": values,
                "utility_invalid_as_abort": abort_adjusted_utility,
                "regret_invalid_as_abort": max(0.0, optimum - abort_adjusted_utility),
                "prompt_tokens": prompt_length,
                "generated_tokens": int(new_tokens.numel()),
                "generation_seconds": elapsed,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            if index == 1 or index % 50 == 0 or index == len(scenarios):
                print(json.dumps({"completed": index, "total": len(scenarios), "format_valid": sum(item["format_valid"] for item in records)}), flush=True)

    regimes = sorted({record["regime"] for record in records})
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": SCOPE,
        "limitations": [
            "Only the first inspection decision is generated by an LLM; the final controller is an exact CPU oracle using lawful observations.",
            "Returns and regrets are exact expectations in the synthetic model, not sampled real-world outcomes or full-agent safety measurements.",
            "One fixed prompt variant is evaluated per unique scenario. A profile subset is not a complete benchmark result.",
        ],
        "invalid_output_convention": "Malformed outputs have null conditional return/regret and are counted in format failures. The separately named invalid_as_abort aggregates assign them utility 0, corresponding to safe abort with no inspection or repair.",
        "model": args.model,
        "adapter": args.adapter,
        "requested_revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "regime_filter": args.regime,
        "eligible_scenarios": eligible_count,
        "evaluated_scenarios": len(records),
        "profile_subset": len(records) < eligible_count,
        "scenario_selection_seed": args.seed,
        "prompt_variant": args.variant,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(device),
        "precision": "bfloat16",
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_version},
        "elapsed_evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "overall": summarize(records),
        "by_regime": {regime: summarize([record for record in records if record["regime"] == regime]) for regime in regimes},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(predictions_path), "summary": str(summary_path), "overall": summary["overall"]}, indent=2))


if __name__ == "__main__":
    main()
