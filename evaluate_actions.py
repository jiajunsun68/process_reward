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
import json
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any

from core import Inspection, Scenario, TIE_TOLERANCE, first_actions, information_value
from generate_pairs import TEMPLATES, format_prompt


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
                output = model.generate(**inputs, generation_config=generation_config)
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
