"""Conditional SFT positive control: one global-optimal first action per train scenario.

Created only for the G3 conditional round's SFT trigger (see status.md section G).
This is NOT a new framework: it reuses train_dpo.py's action parsing, LoRA export,
CUDA checks, and action-only token boundary, and trains with the standard
transformers.Trainer.  Only action-content tokens contribute to the cross-entropy;
the prompt, assistant header, padding, EOS, and chat suffix are masked (-100).

Data are built in memory from the shared conditional scenario manifest: each of
the 720 train scenarios contributes exactly one example, whose completion is the
scenario's single globally optimal action (rank 0 of its TRUE ranking).  DPO
pair "chosen" rows are never all reused as SFT labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

from core import Inspection, Scenario, TIE_TOLERANCE, first_actions, information_value
from generate_pairs import action_text as canonical_action_text
from generate_pairs import format_conditional_prompt
from train_dpo import (
    action_text,
    check_cuda,
    file_sha256,
    gpu_memory,
    save_final_adapter,
    tokenize_action_pair,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data" / "g3_conditional" / "scenario_manifest.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-targets", choices=("qv", "all-linear"), default="qv")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--action-format", choices=("tool-json",), default="tool-json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-final-adapter", action="store_true")
    parser.add_argument("--max-save-mib", type=int, default=512)
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if not 1 <= args.max_action_tokens <= args.max_length:
        parser.error("Require 1 <= --max-action-tokens <= --max-length")
    return args


def load_sft_rows(path: Path, tokenizer: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per train scenario: prompt + the scenario's globally optimal action."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            config = dict(row["config"])
            config["inspections"] = tuple(Inspection(**item) for item in config["inspections"])
            scenario = Scenario(**config)
            if row["scenario_id"] != scenario.scenario_id:
                raise ValueError(f"Scenario ID mismatch at {path}:{line_number}")
            if scenario.scenario_id in seen:
                raise ValueError(f"Duplicate train scenario at {path}:{line_number}")
            seen.add(scenario.scenario_id)
            values = {candidate: information_value(scenario, candidate) for candidate in first_actions(scenario)}
            best = max(values.values())
            optimal = sorted(action for action, value in values.items() if best - value <= TIE_TOLERANCE)
            if len(optimal) != 1:
                raise ValueError(f"{path}:{line_number}: global optimum is not unique")
            completion = canonical_action_text(optimal[0])
            if action_text([{"role": "assistant", "content": completion}], f"{path}:{line_number}", args.action_format) != completion:
                raise ValueError(f"{path}:{line_number}: action does not round-trip through tool-json")
            prompt = tokenizer.apply_chat_template(
                format_conditional_prompt(scenario, variant=0), tokenize=False, add_generation_prompt=True
            )
            # Rejected is supplied only because the shared boundary helper tokenizes both
            # labels; SFT uses the chosen path only.
            tokens = tokenize_action_pair({"prompt": prompt, "chosen": completion, "rejected": completion}, tokenizer)
            prompt_ids = tokens["prompt_input_ids"]
            completion_ids = tokens["chosen_input_ids"]
            if len(completion_ids) > args.max_action_tokens:
                raise ValueError(f"{path}:{line_number}: action exceeds --max-action-tokens; no truncation is performed")
            if len(prompt_ids) + len(completion_ids) > args.max_length:
                raise ValueError(f"{path}:{line_number}: sequence exceeds --max-length; no truncation is performed")
            rows.append({
                "scenario_id": scenario.scenario_id, "family_id": row["family_id"],
                "completion": completion, "optimal_action": optimal[0],
                "input_ids": prompt_ids + completion_ids,
                "labels": [-100] * len(prompt_ids) + completion_ids,
            })
    if not rows:
        raise ValueError(f"No train scenarios in {path}")
    if args.max_train_samples is not None and args.max_train_samples < len(rows):
        selected = random.Random(args.seed).sample(range(len(rows)), args.max_train_samples)
        rows = [rows[index] for index in sorted(selected)]
    stats = {
        "source": str(path.resolve()), "sha256": file_sha256(path),
        "train_scenarios": len(rows),
        "max_prompt_tokens": max(len(row["input_ids"]) - len([x for x in row["labels"] if x != -100]) for row in rows),
        "max_action_tokens": max(len([x for x in row["labels"] if x != -100]) for row in rows),
        "mean_prompt_tokens": sum(len(row["input_ids"]) - len([x for x in row["labels"] if x != -100]) for row in rows) / len(rows),
        "label_actions": dict(Counter(row["optimal_action"] for row in rows)),
        "scored_tokens": "action content only; prompt, assistant header, EOS, padding, and chat suffix masked (-100)",
    }
    return rows, stats


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}. Choose a new directory for this run.")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit("Missing GPU dependencies. Install requirements-gpu.txt in a dedicated environment.") from exc

    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")}
    hardware = check_cuda(torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    installed = sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions() if dist.metadata.get("Name")})
    environment = {"python": sys.version, "platform": __import__("platform").platform(), "hardware": hardware, "process_id": os.getpid()}
    print(json.dumps(environment, indent=2), flush=True)
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    metrics: dict[str, Any] = {"status": "starting", "gpu_validation": "performed only on the machine executing this script", "completed_optimizer_steps": 0}
    run = {
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "environment": environment,
        "versions": {"packages": versions, "installed": installed},
        "script_sha256": file_sha256(Path(__file__)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"name_or_path": args.model, "requested_revision": args.revision, "resolved_commit": None},
        "data": {"train": {"source": str(args.manifest.resolve()), "sha256": file_sha256(args.manifest)}},
        "metrics": metrics,
        "storage_policy": {"automatic_checkpoints": False, "final_adapter_requested": args.save_final_adapter,
                           "max_adapter_mib": args.max_save_mib, "training_log_interval_steps": 10},
    }
    run_path = args.output_dir / "run.json"
    metrics_path = args.output_dir / "metrics.jsonl"
    with run_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(run, indent=2, ensure_ascii=False, default=str) + "\n")
    metrics_path.touch(exist_ok=False)

    def record_event(event: str, step: int | None = None, values: dict | None = None) -> None:
        memory = gpu_memory(torch)
        for kind in ("allocated", "reserved"):
            key = f"run_peak_{kind}_gib"
            metrics[key] = max(metrics.get(key, 0.0), memory[f"peak_{kind}_gib"])
        if step is not None:
            metrics["completed_optimizer_steps"] = step
        record = {"event": event, "step": metrics["completed_optimizer_steps"],
                  "elapsed_seconds": time.perf_counter() - started, "gpu": memory, "metrics": values or {}}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        write_json(run_path, run)

    class ActionOnlySFTTrainer(Trainer):
        def _save_checkpoint(self, *unused_args, **unused_kwargs):
            raise RuntimeError("Automatic checkpoints are disabled for this entry point")

        def save_model(self, *unused_args, **unused_kwargs):
            raise RuntimeError("Trainer saves are disabled; use --save-final-adapter")

        def save_state(self, *unused_args, **unused_kwargs):
            raise RuntimeError("Trainer state saving is disabled for this entry point")

    class MemoryLogCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            logs = logs or {}
            phase_summary = "train_runtime" in logs or any(key.startswith("eval_") for key in logs)
            if state.global_step % 10 == 0 or phase_summary:
                record_event("phase_metrics" if phase_summary else "training_metrics", state.global_step, logs)

    try:
        record_event("starting")
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False)
        if tokenizer.chat_template is None:
            raise ValueError("The tokenizer must provide its model's chat template")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a padding token nor an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        train_rows, train_stats = load_sft_rows(args.manifest, tokenizer, args)
        run["data"] = {"train": train_stats}
        metrics["data"] = train_stats
        record_event("data_prepared")
        print(json.dumps({"train": train_stats}, indent=2), flush=True)

        expected_steps = int(len(train_rows) * args.epochs + 1e-9) // (args.microbatch * args.grad_accum) if args.max_steps == -1 else args.max_steps
        metrics["expected_optimizer_steps"] = expected_steps
        metrics["effective_train_batch"] = args.microbatch * args.grad_accum

        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None,
            low_cpu_mem_usage=True, trust_remote_code=False, use_safetensors=True,
        )
        model.config.use_cache = False
        model.to(torch.device("cuda:0"))
        metrics["resolved_model_commit"] = getattr(model.config, "_commit_hash", None)
        run["model"]["resolved_commit"] = metrics["resolved_model_commit"]
        target_modules = ["q_proj", "v_proj"] if args.lora_targets == "qv" else "all-linear"
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=target_modules, bias="none",
        )
        model = get_peft_model(model, peft_config)
        record_event("model_loaded")
        model.print_trainable_parameters()
        metrics["parameter_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
        metrics["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        dataset_rows = [{"input_ids": row["input_ids"], "labels": row["labels"]} for row in train_rows]

        def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
            # microbatch=1: no padding is ever needed or silently introduced.
            if len(features) != 1:
                raise ValueError("Action-only SFT collator expects exactly one example per microbatch")
            feature = features[0]
            return {
                "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long),
                "labels": torch.tensor([feature["labels"]], dtype=torch.long),
            }

        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.microbatch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            bf16=True, fp16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=0.05,
            logging_strategy="steps", logging_steps=10, logging_first_step=False,
            logging_dir=str(args.output_dir), save_strategy="no", save_only_model=True,
            load_best_model_at_end=False, eval_strategy="no",
            report_to="none", dataloader_num_workers=0, remove_unused_columns=False,
            seed=args.seed, data_seed=args.seed,
        )
        trainer = ActionOnlySFTTrainer(
            model=model, args=training_args,
            train_dataset=Dataset.from_list(dataset_rows),
            data_collator=collate, callbacks=[MemoryLogCallback()],
        )
        metrics["status"] = "training"
        record_event("training_start")
        train_start = time.perf_counter()
        training_result = trainer.train(resume_from_checkpoint=False)
        torch.cuda.synchronize()
        metrics["training_wall_seconds"] = time.perf_counter() - train_start
        metrics["training_memory"] = gpu_memory(torch)
        metrics["train"] = training_result.metrics
        metrics["completed_optimizer_steps"] = trainer.state.global_step
        if trainer.state.global_step != expected_steps:
            raise RuntimeError(f"Expected {expected_steps} optimizer steps, got {trainer.state.global_step}")
        record_event("training_complete", trainer.state.global_step, training_result.metrics)

        metrics["status"] = "fit_metrics"
        record_event("fit_metrics_start")
        model.eval()
        fit: dict[str, Any] = {}
        losses: list[float] = []
        token_correct = token_total = 0
        exact = 0
        with torch.inference_mode():
            for row in train_rows:
                input_ids = torch.tensor([row["input_ids"]], device="cuda:0")
                labels = torch.tensor([row["labels"]], device="cuda:0")
                output = model(input_ids=input_ids, labels=labels)
                losses.append(float(output.loss))
                predictions = output.logits[0, :-1].argmax(dim=-1)
                targets = labels[0, 1:]
                mask = (labels[0] != -100)[1:]
                token_correct += int((predictions[mask] == targets[mask]).sum())
                token_total += int(mask.sum())
                exact += int(bool((predictions[mask] == targets[mask]).all()))
        fit["train"] = {
            "n": len(train_rows), "mean_loss": sum(losses) / len(losses),
            "completion_token_accuracy": token_correct / token_total if token_total else None,
            "exact_action_accuracy": exact / len(train_rows),
        }
        metrics["fit"] = fit
        metrics["fit_note"] = "Held-out generalization is reported by the separate G3-SFT-eval-a01 conditional evaluation, not by this train-only fit pass."
        record_event("fit_metrics_complete", values=fit)

        if args.save_final_adapter:
            metrics["status"] = "saving_final_adapter"
            record_event("adapter_export_start")
            metrics["saved_adapter"] = save_final_adapter(trainer.model, args, metrics["resolved_model_commit"])
            record_event("adapter_export_complete", values=metrics["saved_adapter"])
        metrics["status"] = "completed"
    except BaseException as exc:
        metrics["status"] = "failed"
        metrics["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metrics["total_wall_seconds"] = time.perf_counter() - started
        metrics["last_phase_memory"] = gpu_memory(torch)
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record_event(metrics["status"])
        print(json.dumps(metrics, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
