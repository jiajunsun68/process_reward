#!/usr/bin/env python3
"""Single-GPU, action-only DPO pilot; no claim of completed GPU validation.

Example (run on the machine that actually has the GPU):
  CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
    --train-file data/information/train.jsonl --output-dir runs/info-profile \
    --max-steps 20 --max-train-samples 256

Input: conversational prompt/chosen/rejected JSONL. Each completion must contain
exactly one assistant action. Metadata is never passed to the model.

Each run writes only run.json and metrics.jsonl by default. No checkpoints are
saved. Add --save-final-adapter to export one final, size-limited adapter with
only adapter_model.safetensors and adapter_config.json. An output directory
must be empty; previous runs are never overwritten or resumed.

The implementation deliberately targets the released TRL 0.21.0 API, not main:
https://huggingface.co/docs/trl/v0.21.0/dpo_trainer
https://github.com/huggingface/trl/blob/v0.21.0/trl/trainer/dpo_trainer.py
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, help="Optional held-out validation JSONL; do not use the test split.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-final-adapter", action="store_true", help="Opt in to one final two-file adapter; profiling saves no model by default.")
    parser.add_argument("--max-save-mib", type=int, default=512, help="Maximum combined final adapter file size in MiB (default: 512).")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="main", help="Use a model commit hash for a frozen reproduction.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="20 for profiling; -1 uses --epochs.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-action-tokens", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=1, help="Preference pairs per microbatch; internally two sequences.")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-targets", choices=("qv", "all-linear"), default="qv")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--action-format", choices=("tool-json", "text"), default="tool-json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("max_length", "max_action_tokens", "microbatch", "grad_accum", "lora_rank", "lora_alpha", "max_save_mib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_samples", "max_eval_samples"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.beta <= 0:
        parser.error("epochs, learning rate, and beta must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be in [0, 1)")
    return args


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_final_adapter(model: Any, args: argparse.Namespace, resolved_commit: str | None) -> dict[str, Any]:
    """Export only LoRA tensors/config, without Trainer or PEFT model-card saves.

    Serialize in memory and check the exact combined size before creating any
    adapter files. The current trainer supports one standard LoRA adapter only.
    """
    from peft import get_peft_model_state_dict
    from safetensors.torch import save as serialize_safetensors

    if set(model.peft_config) != {"default"}:
        raise RuntimeError("Final export supports exactly one adapter named 'default'")
    config = model.peft_config["default"].to_dict()
    config["inference_mode"] = True
    config["base_model_name_or_path"] = config.get("base_model_name_or_path") or args.model
    config["revision"] = resolved_commit or args.revision

    def config_json(value: Any) -> Any:
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, Enum):
            return value.value
        raise TypeError(f"Unsupported adapter config value: {type(value).__name__}")

    config_bytes = (json.dumps(config, indent=2, sort_keys=True, default=config_json) + "\n").encode("utf-8")
    state = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if not state or any("lora_" not in key for key in state):
        raise RuntimeError("Refusing to export empty or non-LoRA weights")
    limit_bytes = args.max_save_mib * 1024**2
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
    if tensor_bytes + len(config_bytes) > limit_bytes:
        raise ValueError(f"Adapter tensors exceed --max-save-mib={args.max_save_mib}; no adapter files were written")
    # A separate contiguous CPU allocation per tensor also avoids shared-storage
    # aliases, which the safetensors format deliberately rejects.
    cpu_state = {key: tensor.detach().to("cpu").contiguous().clone() for key, tensor in state.items()}
    weights_bytes = serialize_safetensors(cpu_state, metadata={"format": "pt"})
    total_bytes = len(weights_bytes) + len(config_bytes)
    if total_bytes > limit_bytes:
        raise ValueError(f"Serialized adapter exceeds --max-save-mib={args.max_save_mib}; no adapter files were written")
    final_path = args.output_dir / "adapter"
    final_path.mkdir(exist_ok=False)
    written = []
    try:
        for name, payload in (("adapter_model.safetensors", weights_bytes), ("adapter_config.json", config_bytes)):
            path = final_path / name
            with path.open("xb") as stream:
                written.append(path)
                stream.write(payload)
    except BaseException:
        # Clean only this export's own partial files, never pre-existing paths.
        for path in written:
            path.unlink(missing_ok=True)
        if not any(final_path.iterdir()):
            final_path.rmdir()
        raise
    return {
        "path": str(final_path.resolve()), "files": [path.name for path in written],
        "total_bytes": total_bytes, "limit_bytes": limit_bytes,
        "sha256": {"adapter_model.safetensors": hashlib.sha256(weights_bytes).hexdigest(),
                   "adapter_config.json": hashlib.sha256(config_bytes).hexdigest()},
    }


def require_parameters(callable_object: Any, names: set[str]) -> None:
    available = set(inspect.signature(callable_object).parameters)
    missing = names - available
    if missing:
        raise RuntimeError(f"Incompatible API {callable_object}: missing {sorted(missing)}. Install requirements-gpu.txt.")


def check_cuda(torch: Any) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This entry point is single-process. Use python, not a multi-process torchrun/accelerate launch.")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A working CUDA PyTorch installation is required. Run gpu_smoke.py on the GPU server.")
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            f"Found {count} visible GPUs. Expose exactly one: CUDA_VISIBLE_DEVICES=0 python train_dpo.py ... . "
            "Use the other physical card in a separate evaluation process."
        )
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    bf16_kwargs = {}
    if "including_emulation" in inspect.signature(torch.cuda.is_bf16_supported).parameters:
        bf16_kwargs["including_emulation"] = False
    supported = bool(torch.cuda.is_bf16_supported(**bf16_kwargs))
    if not supported or props.major < 8:
        raise RuntimeError("This configuration requires native CUDA BF16 support (Ampere or newer).")
    return {
        "visible_gpu_count": count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "logical_device": "cuda:0",
        "name": props.name,
        "compute_capability": [props.major, props.minor],
        "total_memory_gib": props.total_memory / (1024**3),
        "torch_cuda_version": torch.version.cuda,
        "bf16_supported": supported,
    }


def gpu_memory(torch: Any) -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated(0) / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved(0) / (1024**3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / (1024**3),
    }


def action_text(value: Any, location: str, action_format: str) -> str:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"{location}: expected exactly one assistant message")
    message = value[0]
    if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        raise ValueError(f"{location}: expected assistant role and string content")
    action = message["content"]
    if not action.strip() or "<think>" in action or "</think>" in action:
        raise ValueError(f"{location}: completion must be a nonempty action without reasoning")
    if action_format == "tool-json":
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: invalid JSON tool action") from exc
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"tool", "arguments"}
            or not isinstance(parsed["tool"], str)
            or not parsed["tool"]
            or not isinstance(parsed["arguments"], dict)
        ):
            raise ValueError(f"{location}: action must have only string 'tool' and object 'arguments' fields")
    return action


def tokenize_action_pair(features: dict[str, str], tokenizer: Any) -> dict[str, list[int]]:
    """No EOS/template suffix is scored; only the literal action is a completion."""
    prompt_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("Empty tokenized prompt")
    result = {"prompt_input_ids": prompt_ids}
    for label in ("chosen", "rejected"):
        full_ids = tokenizer(features["prompt"] + features[label], add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenizer merges across the prompt/action boundary; change the prompt delimiter.")
        completion = full_ids[len(prompt_ids) :]
        if not completion:
            raise ValueError("Empty tokenized action")
        result[f"{label}_input_ids"] = completion
    return result


def load_action_data(path: Path, tokenizer: Any, args: argparse.Namespace, limit: int | None) -> tuple[list[dict], dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            prefix = item.get("prompt")
            location = f"{path}:{line_number}"
            if not isinstance(prefix, list) or not prefix:
                raise ValueError(f"{location}: prompt must be a nonempty conversation")
            for message in prefix:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant", "tool"}
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(f"{location}: invalid text-only prompt message")
            if prefix[-1]["role"] not in {"user", "tool"}:
                raise ValueError(f"{location}: prompt must end before a new assistant action")
            chosen = action_text(item.get("chosen"), location + ":chosen", args.action_format)
            rejected = action_text(item.get("rejected"), location + ":rejected", args.action_format)
            if chosen == rejected:
                raise ValueError(f"{location}: chosen and rejected actions are identical")
            prompt = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
            row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            tokens = tokenize_action_pair(row, tokenizer)
            action_lengths = [len(tokens[f"{label}_input_ids"]) for label in ("chosen", "rejected")]
            if max(action_lengths) > args.max_action_tokens:
                raise ValueError(f"{location}: action exceeds --max-action-tokens; no truncation is performed")
            if len(tokens["prompt_input_ids"]) + max(action_lengths) > args.max_length:
                raise ValueError(f"{location}: sequence exceeds --max-length; no history or action is silently truncated")
            row["_prompt_length"] = len(tokens["prompt_input_ids"])
            row["_action_length"] = max(action_lengths)
            rows.append(row)
    if not rows:
        raise ValueError(f"No preference pairs in {path}")
    original_count = len(rows)
    if limit is not None and limit < len(rows):
        selected = random.Random(args.seed).sample(range(len(rows)), limit)
        rows = [rows[index] for index in sorted(selected)]
    stats = {
        "source": str(path.resolve()),
        "sha256": file_sha256(path),
        "source_pairs": original_count,
        "selected_pairs": len(rows),
        "max_prompt_tokens": max(row["_prompt_length"] for row in rows),
        "max_action_tokens": max(row["_action_length"] for row in rows),
        "mean_prompt_tokens": sum(row["_prompt_length"] for row in rows) / len(rows),
        "scored_tokens": "action content only; prompt, assistant header, EOS, and chat suffix excluded",
    }
    for row in rows:
        row.pop("_prompt_length")
        row.pop("_action_length")
    return rows, stats


def main() -> None:
    args = parse_args()
    for path in (args.train_file, args.eval_file):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}. Choose a new directory for this run.")
    # Imports happen after argument parsing, so --help requires no GPU packages.
    try:
        import torch
        from datasets import Dataset
        from packaging.version import Version
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit("Missing GPU dependencies. Install requirements-gpu.txt in a dedicated environment.") from exc

    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "safetensors", "packaging")}
    if versions["trl"] != "0.21.0" or not Version("4.55.0") <= Version(versions["transformers"]) < Version("4.56.0"):
        raise RuntimeError(f"This entry point targets TRL 0.21.0 and Transformers 4.55.x; got {versions}. Install requirements-gpu.txt.")
    require_parameters(DPOConfig, {
        "precompute_ref_log_probs", "precompute_ref_batch_size", "max_length", "max_prompt_length",
        "max_completion_length", "use_logits_to_keep", "gradient_checkpointing_kwargs", "disable_dropout",
    })
    require_parameters(DPOTrainer.__init__, {"processing_class", "peft_config", "ref_model"})
    require_parameters(DPOTrainer.tokenize_row, {"features", "processing_class", "max_prompt_length", "max_completion_length", "add_special_tokens"})
    hardware = check_cuda(torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    installed = sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions() if dist.metadata.get("Name")})
    environment = {"python": sys.version, "platform": platform.platform(), "hardware": hardware, "process_id": os.getpid()}
    print(json.dumps(environment, indent=2), flush=True)
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    metrics: dict[str, Any] = {"status": "starting", "gpu_validation": "performed only on the machine executing this script", "completed_optimizer_steps": 0}
    run = {
        "args": vars(args), "environment": environment,
        "versions": {"packages": versions, "installed": installed},
        "script_sha256": file_sha256(Path(__file__)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"name_or_path": args.model, "requested_revision": args.revision, "resolved_commit": None},
        "data": {name: {"source": str(path.resolve()), "sha256": file_sha256(path)} if path else None
                 for name, path in (("train", args.train_file), ("validation", args.eval_file))},
        "metrics": metrics,
        "storage_policy": {"automatic_checkpoints": False, "final_adapter_requested": args.save_final_adapter,
                           "max_adapter_mib": args.max_save_mib, "training_log_interval_steps": 10},
    }
    run_path = args.output_dir / "run.json"
    metrics_path = args.output_dir / "metrics.jsonl"
    # Exclusive creation also prevents two processes from claiming the same run.
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

    class ActionOnlyDPOTrainer(DPOTrainer):
        @staticmethod
        def tokenize_row(features, processing_class, max_prompt_length=None, max_completion_length=None, add_special_tokens=False):
            # TRL 0.21's default appends EOS. We intentionally score no EOS.
            # Its normal concatenated_forward masks the whole prompt and padding.
            if max_prompt_length is not None or max_completion_length is not None or add_special_tokens:
                raise ValueError("Action-only tokenization requires no implicit special tokens or truncation")
            return tokenize_action_pair(features, processing_class)

        # save_strategy='no' prevents normal step/epoch/end checkpoints in
        # Transformers 4.55. These guards fail before any checkpoint/model card
        # is created if a callback or changed internal path requests a save.
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
        train_rows, train_stats = load_action_data(args.train_file, tokenizer, args, args.max_train_samples)
        eval_rows, eval_stats = (None, None)
        if args.eval_file:
            eval_rows, eval_stats = load_action_data(args.eval_file, tokenizer, args, args.max_eval_samples)
        run["data"] = {"train": train_stats, "validation": eval_stats}
        record_event("data_prepared")
        print(json.dumps({"train": train_stats, "validation": eval_stats}, indent=2), flush=True)

        # No device_map='auto', sharding, offload, or second reference model.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None,
            low_cpu_mem_usage=True, trust_remote_code=False, use_safetensors=True,
        )
        model.config.use_cache = False
        require_parameters(model.forward, {"logits_to_keep"})
        model.to(torch.device("cuda:0"))
        metrics["resolved_model_commit"] = getattr(model.config, "_commit_hash", None)
        run["model"]["resolved_commit"] = metrics["resolved_model_commit"]
        record_event("model_loaded")
        target_modules = ["q_proj", "v_proj"] if args.lora_targets == "qv" else "all-linear"
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=target_modules, bias="none",
        )
        config = DPOConfig(
            output_dir=str(args.output_dir), per_device_train_batch_size=args.microbatch,
            per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate, num_train_epochs=args.epochs, max_steps=args.max_steps,
            max_length=args.max_length, max_prompt_length=None, max_completion_length=None,
            beta=args.beta, loss_type="sigmoid", bf16=True, fp16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            precompute_ref_log_probs=True, precompute_ref_batch_size=1,
            # TRL's default True would silently zero LoRA's requested 0.05 dropout.
            disable_dropout=False, use_logits_to_keep=True,
            optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=0.05,
            logging_strategy="steps", logging_steps=10, logging_first_step=False,
            logging_dir=str(args.output_dir), save_strategy="no", save_only_model=True,
            load_best_model_at_end=False, eval_strategy="no",
            report_to="none", push_to_hub=False, dataloader_num_workers=0,
            seed=args.seed, data_seed=args.seed, remove_unused_columns=True,
        )
        trainer = ActionOnlyDPOTrainer(
            model=model, ref_model=None, args=config, processing_class=tokenizer,
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(eval_rows) if eval_rows is not None else None,
            peft_config=peft_config, callbacks=[MemoryLogCallback()],
        )
        if trainer.ref_model is not None or not trainer.is_peft_model:
            raise RuntimeError("Expected a LoRA model using its adapter-disabled base as the reference")
        trainer.model.print_trainable_parameters()
        metrics["parameter_dtypes"] = dict(Counter(str(p.dtype) for p in trainer.model.parameters()))
        metrics["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        metrics["effective_preference_batch"] = args.microbatch * args.grad_accum
        metrics["reference"] = "frozen base model, adapter disabled, eval mode, precomputed before updates"
        metrics["status"] = "precomputing_reference"
        record_event("reference_precompute_start")
        # Precompute in eval mode so base/adapter dropout cannot perturb reference scores.
        trainer.model.eval()
        torch.cuda.synchronize()
        ref_start = time.perf_counter()
        loader = trainer.get_train_dataloader()
        del loader
        if eval_rows is not None:
            loader = trainer.get_eval_dataloader()
            del loader
        for split_name, dataset in (("train", trainer.train_dataset), ("validation", trainer.eval_dataset)):
            if dataset is None:
                continue
            for column in ("ref_chosen_logps", "ref_rejected_logps"):
                if column not in dataset.column_names:
                    raise RuntimeError(f"Reference precomputation did not populate {split_name}/{column}")
                if not torch.isfinite(torch.tensor(dataset[column], dtype=torch.float32)).all().item():
                    raise RuntimeError(f"Nonfinite reference scores in {split_name}/{column}")
        torch.cuda.synchronize()
        metrics["reference_precompute_seconds"] = time.perf_counter() - ref_start
        metrics["setup_and_reference_memory"] = gpu_memory(torch)
        metrics["setup_seconds"] = time.perf_counter() - started
        record_event("reference_precompute_complete")
        print(json.dumps({"precompute": metrics}, indent=2), flush=True)
        torch.cuda.reset_peak_memory_stats(0)
        train_start = time.perf_counter()
        metrics["status"] = "training"
        record_event("training_start")
        training_result = trainer.train(resume_from_checkpoint=False)
        torch.cuda.synchronize()
        metrics["training_wall_seconds"] = time.perf_counter() - train_start
        metrics["training_memory"] = gpu_memory(torch)
        metrics["train"] = training_result.metrics
        metrics["completed_optimizer_steps"] = trainer.state.global_step
        if trainer.state.global_step > 0:
            metrics["wall_seconds_per_optimizer_step"] = metrics["training_wall_seconds"] / trainer.state.global_step
        record_event("training_complete", trainer.state.global_step, training_result.metrics)
        if eval_rows is not None:
            metrics["status"] = "validating"
            record_event("validation_start")
            metrics["validation"] = trainer.evaluate()
            record_event("validation_complete", values=metrics["validation"])
        if args.save_final_adapter:
            metrics["status"] = "saving_final_adapter"
            record_event("adapter_export_start")
            metrics["saved_adapter"] = save_final_adapter(
                trainer.accelerator.unwrap_model(trainer.model), args, metrics["resolved_model_commit"],
            )
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
