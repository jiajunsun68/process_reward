"""G7 pointer-action policy training and evaluation.

This is the unified training/evaluation entry.  The model never emits a tool
argument: it selects one fixed source label (A..J) or Z=UNKNOWN.  The program
copies the public source value and performs the native ToolSandbox operation.
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
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "g7_process"
RUNS_DIR = ROOT / "runs"
STATUS_PATH = ROOT / "status.md"
QWEN_VENV = Path("/data/sjj/process_reward_venv/bin/python")
QWEN_SNAPSHOT = Path("/data4/sjj/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/"
                     "snapshots/2de2439ea21be1dc5cb21f22f88af07e43393cbb")
LABELS = "ABCDEFGHIJ"
UNKNOWN_LABEL = "Z"
MAX_CONTEXT_TOKENS = 2048
BOOTSTRAP_SEED = 23
BOOTSTRAP_REPS = 2000
TRAIN_SEEDS = (11, 23)
TRAIN_UPDATES = 128
GROUP_SIZE = 4
FAMILIES_PER_UPDATE = 4
SFT_LR = 1e-4
RL_LR = 1e-5
KL_COEF = 0.02
CLIP_EPS = 0.2
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
# The frozen instruct base is close to deterministic on the pointer prompt:
# with zero LoRA-B initialization most four-sample groups have zero reward
# variance.  A seeded nonzero LoRA-B initialization keeps T=1.0 categorical
# sampling as the exploration mechanism while leaving the base and LoRA budget
# unchanged; the same init is used for both reward arms at a given seed.
INIT_STD = 0.7
SFT_MICRO_BATCH = 1
SFT_EXAMPLES_PER_UPDATE = 24
TASK_PROMPT_VERSION = "g7-pointer-prompt-v1"
PROTOCOL_ID = "g7-pointer-policy-v1"


def now_cst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases() -> list[dict[str, Any]]:
    path = DATA_DIR / "cases.jsonl"
    if not path.exists():
        raise SystemExit(f"missing G7 data: {path}; run g7_data.py --build")
    return read_jsonl(path)


def cases_by_split(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [case for case in cases if case["split"] == split]


def case_final_prefix(case: dict[str, Any]) -> dict[str, Any]:
    return case["prefixes"][-1]


def allowed_labels(prefix: dict[str, Any]) -> list[str]:
    labels = [candidate["label"] for candidate in prefix["candidates"]]
    return labels + [UNKNOWN_LABEL]


# ---------------------------------------------------------------------------
# Prompt: only public history, fixed candidate labels, and prior labels.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a policy that maintains which public-source value is currently supported. "
    "You reply with one fixed candidate label only. A separate program copies the chosen "
    "source value and performs the final tool operation; you never write a phone number, "
    "timestamp, value, or tool argument."
)


def task_text(case: dict[str, Any]) -> str:
    if case["workflow"] == "contact":
        operation = ("At the terminal event the program will send the task message to the target "
                     "using the current supported phone number.")
        field_name = "phone_number"
    else:
        operation = ("At the terminal event the program will modify the target reminder by setting "
                     "its timestamp to the current supported reminder_timestamp plus 1800 seconds.")
        field_name = "reminder_timestamp"
    return (f"Target object: {case['target_id']}\n"
            f"Field being maintained: {field_name}\n"
            f"Operation: {operation}\n"
            "At every public observation, choose the candidate source for the value that is currently "
            "in force for the target field, or Z when no public source currently supports a value. "
            "Candidates may belong to another object; selecting one copies that source's value.")


def build_action_prompt(case: dict[str, Any], prefix: dict[str, Any], representation: str,
                        previous: list[str]) -> list[dict[str, str]]:
    history = prefix["history_j"] if representation == "J" else prefix["history_n"]
    candidate_lines = []
    for candidate in prefix["candidates"]:
        candidate_lines.append(
            f"{candidate['label']} = event {candidate['source_event']}, object "
            f"{candidate['source_object']}, field {candidate['source_field']}, value "
            f"{json.dumps(candidate['source_value'], ensure_ascii=False)}")
    if candidate_lines:
        candidates_text = "\n".join(candidate_lines)
    else:
        candidates_text = "(no candidate source values have been observed yet)"
    allowed = allowed_labels(prefix)
    prior = "\n".join(f"step {index + 1}: {label}" for index, label in enumerate(previous))
    if not prior:
        prior = "(none)"
    user = (
        f"{task_text(case)}\n\n"
        "Public observations (oldest first):\n"
        f"{history}\n\n"
        "Candidate sources with fixed labels:\n"
        f"{candidates_text}\n"
        f"{UNKNOWN_LABEL} = UNKNOWN (no currently supported public source value)\n\n"
        "Prior choices in this episode (labels only, not necessarily correct):\n"
        f"{prior}\n\n"
        f"Allowed labels: {', '.join(allowed)}\n"
        "Reply with exactly one label character from the allowed labels and nothing else."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def prompt_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Model / token-policy.
# ---------------------------------------------------------------------------

class Policy:
    def __init__(self, device: str = "cuda:0", adapter: str | Path | None = None,
                 trainable: bool = False, init_seed: int | None = None,
                 init_std: float = INIT_STD) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(QWEN_SNAPSHOT), local_files_only=True,
                                                       trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(
            str(QWEN_SNAPSHOT), local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=False,
        ).to(self.device)
        base.config.use_cache = False
        self.model = base
        if adapter is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
        elif trainable:
            config = LoraConfig(
                task_type="CAUSAL_LM", r=LORA_R, lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT, bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(base, config)
            if init_std:
                if init_seed is not None:
                    torch.manual_seed(int(init_seed))
                with torch.no_grad():
                    for name, module in self.model.named_modules():
                        if name.endswith("lora_B"):
                            target = getattr(module, "default", module)
                            target.weight.data.normal_(0.0, init_std)
        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            try:
                self.model.gradient_checkpointing_disable()
            except Exception:
                pass
        self.trainable = trainable
        self.label_ids: dict[str, int] = {}
        for label in LABELS + UNKNOWN_LABEL:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"label {label!r} is not a single token: {ids}")
            self.label_ids[label] = int(ids[0])
        ids = list(self.label_ids.values())
        if len(set(ids)) != len(ids):
            raise RuntimeError("label token IDs collide")
        self.label_to_id = self.label_ids
        self.id_to_label = {value: key for key, value in self.label_ids.items()}
        self.prompt_cache: dict[str, tuple[Any, Any]] = {}
        self.model_calls = 0
        self.input_tokens = 0
        self.reference_model: Policy | None = None
        self.label_token_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

    def encode(self, messages: list[dict[str, str]]):
        import torch
        key = prompt_hash(messages)
        cached = self.prompt_cache.get(key)
        if cached is not None:
            return cached
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        if input_ids.shape[1] > MAX_CONTEXT_TOKENS:
            raise RuntimeError(f"prompt exceeds {MAX_CONTEXT_TOKENS} tokens: {input_ids.shape[1]}")
        result = (input_ids, attention)
        if len(self.prompt_cache) < 20000:
            self.prompt_cache[key] = result
        return result

    def _logits(self, ids, attention, grad: bool = False, reference: bool = False):
        import torch
        if reference and hasattr(self.model, "disable_adapter"):
            context = self.model.disable_adapter()
        else:
            context = None
        manager = torch.enable_grad() if grad else torch.no_grad()
        with manager:
            if context is not None:
                with context:
                    output = self.model(input_ids=ids, attention_mask=attention, use_cache=False)
            else:
                output = self.model(input_ids=ids, attention_mask=attention, use_cache=False)
        logits = output.logits[:, -1, :]
        self.model_calls += 1
        self.input_tokens += int(attention.sum().item())
        return logits

    def restricted(self, logits, allowed: list[str]):
        import torch
        ids = torch.tensor([self.label_ids[label] for label in allowed], dtype=torch.long,
                           device=self.device)
        return torch.log_softmax(logits[:, ids].float(), dim=-1), ids

    def sample(self, messages: list[dict[str, str]], allowed: list[str],
               sample: bool = True, with_reference: bool = True) -> dict[str, Any]:
        import torch
        ids, attention = self.encode(messages)
        logits = self._logits(ids, attention, grad=False)
        logprobs, label_ids = self.restricted(logits, allowed)
        probs = logprobs.exp()
        if sample:
            choice = int(torch.multinomial(probs, num_samples=1).item())
        else:
            choice = int(torch.argmax(probs, dim=-1).item())
        label = allowed[choice]
        ref_logprobs = self.reference_logprobs(messages, allowed) if with_reference else None
        return {
            "label": label,
            "old_logprob": float(logprobs[0, choice].item()),
            "allowed": list(allowed),
            "ref_logprobs": ref_logprobs.detach().cpu() if ref_logprobs is not None else None,
            "input_ids": ids.detach().cpu(),
            "attention_mask": attention.detach().cpu(),
            "prompt_tokens": int(attention.sum().item()),
        }

    def reference_logprobs(self, messages: list[dict[str, str]], allowed: list[str]):
        if self.reference_model is not None:
            ids, attention = self.reference_model.encode(messages)
            logits = self.reference_model._logits(ids, attention, grad=False)
            logprobs, _ = self.reference_model.restricted(logits, allowed)
            return logprobs[0].detach()
        ids, attention = self.encode(messages)
        logits = self._logits(ids, attention, grad=False, reference=True)
        logprobs, _ = self.restricted(logits, allowed)
        return logprobs[0].detach()

    def grad_logprobs(self, input_ids, attention_mask, allowed: list[str]):
        import torch
        ids = input_ids.to(self.device)
        attention = attention_mask.to(self.device)
        logits = self._logits(ids, attention, grad=True)
        logprobs, label_ids = self.restricted(logits, allowed)
        return logprobs[0]

    def save_adapter(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory), safe_serialization=True)
        files = [path for path in directory.rglob("*") if path.is_file()]
        sizes = {path.name: path.stat().st_size for path in files}
        total = sum(sizes.values())
        if total > 128 * 1024 * 1024:
            raise RuntimeError(f"adapter exceeds 128 MiB: {total}")
        return {"path": str(directory), "files": sizes, "bytes": total,
                "within_128MiB": total <= 128 * 1024 * 1024}


def terminal_metrics(worker, case: dict[str, Any], world: dict[str, Any], label: str,
                     cache: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    key = (case["family_id"], world["world_key"], label)
    if key in cache:
        return cache[key]
    final = case_final_prefix(case)
    source = None
    for candidate in final["candidates"]:
        if candidate["label"] == label:
            source = candidate
            break
    if label == UNKNOWN_LABEL:
        mode = "query"
    else:
        if source is None:
            result = {"ok": False, "error_type": "InvalidLabel", "error": "label not in final candidates",
                      "queries": 0, "success": False, "wrong_side_effect": False,
                      "tool_error": True, "abort": False, "utility": 0.0}
            cache[key] = result
            return result
        mode = "use"
    spec = {
        "workflow": case["workflow"], "target_id": case["target_id"],
        "snapshot": world["snapshot"], "mode": mode,
    }
    if mode == "use":
        spec["value"] = source["source_value"]
    response = worker.call({"op": "execute", "spec": spec})
    queries = 1 if mode == "query" else 0
    success = wrong = tool_error = abort = False
    rows = response.get("result") if response.get("ok") else None
    expected = world["final_value"]
    if not response.get("ok"):
        if response.get("error_type") == "NoTargetObservation":
            abort = True
        else:
            tool_error = True
    else:
        rows = rows or []
        if not rows:
            abort = True
        elif case["workflow"] == "contact":
            row = rows[0]
            success = (row.get("recipient_person_id") == case["target_id"]
                       and row.get("recipient_phone_number") == expected
                       and row.get("content") == "G7 local simulation: status update.")
            wrong = not success
        else:
            row = rows[0]
            timestamp = row.get("reminder_timestamp")
            success = (row.get("reminder_id") == case["target_id"]
                       and isinstance(timestamp, (int, float))
                       and abs(float(timestamp) - (float(expected) + 1800.0)) <= 1e-6)
            wrong = not success
    if success:
        utility = 10.0 - queries
    elif wrong:
        utility = -30.0 - queries
    elif abort or tool_error:
        utility = -queries
    else:
        utility = -queries
    result = {
        "ok": bool(response.get("ok")), "mode": mode, "queries": queries,
        "success": success, "wrong_side_effect": wrong, "tool_error": tool_error,
        "abort": abort, "utility": utility,
        "error_type": response.get("error_type"),
    }
    cache[key] = result
    return result


def run_episode(worker, case: dict[str, Any], representation: str, world: dict[str, Any],
                choose: Callable[[list[dict[str, str]], list[str], int], dict[str, Any]],
                terminal_cache: dict[tuple[str, str, str], dict[str, Any]],
                execute_terminal: bool = True) -> dict[str, Any]:
    previous: list[str] = []
    steps: list[dict[str, Any]] = []
    for prefix in case["prefixes"]:
        messages = build_action_prompt(case, prefix, representation, previous)
        allowed = allowed_labels(prefix)
        decision = choose(messages, allowed, prefix["step"])
        label = decision["label"]
        correct = label in prefix["correct_labels"]
        steps.append({
            "step": prefix["step"], "label": label, "correct": bool(correct),
            "allowed": allowed, "final_event": prefix["step"] == len(case["prefixes"]) - 1,
            "event_type": case["events"][prefix["step"]]["type"],
            "public_state_known": prefix["public_state_known"],
        })
        previous.append(label)
    if execute_terminal:
        terminal = terminal_metrics(worker, case, world, steps[-1]["label"], terminal_cache)
    else:
        terminal = {"queries": 0, "success": False, "wrong_side_effect": False,
                    "tool_error": False, "abort": False, "utility": 0.0}
    process_scores = [1.0 if step["correct"] else -1.0 for step in steps]
    process_mean = sum(process_scores) / len(process_scores) if process_scores else 0.0
    support = sum(1 for step in steps if step["correct"]) / len(steps) if steps else 0.0
    utility = float(terminal["utility"])
    record = {
        "family_id": case["family_id"], "split": case["split"], "subset": case["subset"],
        "workflow": case["workflow"], "initial_state": case["initial_state"],
        "length": len(case["events"]), "representation": representation,
        "world_key": world["world_key"], "world_weight": world["weight"],
        "steps": steps, "process_scores": process_scores, "process_support": support,
        "queries": terminal["queries"], "success": terminal["success"],
        "wrong_side_effect": terminal["wrong_side_effect"], "tool_error": terminal["tool_error"],
        "abort": terminal["abort"], "utility": utility,
        "outcome_reward": utility / 10.0,
        "process_reward": utility / 10.0 + 0.5 * process_mean,
        "final_label": steps[-1]["label"],
        "final_correct": steps[-1]["correct"],
        "terminal": terminal,
    }
    return record


# ---------------------------------------------------------------------------
# Evaluation and family-level aggregation.
# ---------------------------------------------------------------------------

def select_dev_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dev = sorted(cases_by_split(cases, "dev"), key=lambda case: case["family_id"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in dev:
        groups[(case["subset"], case["workflow"])].append(case)
    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][:8])
    return sorted(selected, key=lambda case: case["family_id"])


def record_family_values(records: list[dict[str, Any]], field: str,
                         representation: str = "N", subset: str | None = None,
                         workflow: str | None = None) -> dict[str, float]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        if record["representation"] != representation:
            continue
        if subset is not None and record["subset"] != subset:
            continue
        if workflow is not None and record["workflow"] != workflow:
            continue
        value = record.get(field)
        if value is None:
            continue
        grouped[record["family_id"]].append((float(value), float(record["world_weight"])))
    result = {}
    for family_id, pairs in grouped.items():
        weight = sum(weight for _, weight in pairs)
        if weight:
            result[family_id] = sum(value * weight for value, weight in pairs) / weight
    return result


def mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "n": 0, "ci95": None}
    mean = sum(values) / len(values)
    return {"mean": mean, "n": len(values), "ci95": None}


def bootstrap_mean(values: list[float], seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    if not values:
        return {"mean": None, "n": 0, "ci95": None}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low = means[int(0.025 * (reps - 1))]
    high = means[int(0.975 * (reps - 1))]
    return {"mean": sum(values) / n, "n": n, "ci95": [low, high]}


def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    deltas = [a[key] - b[key] for key in keys]
    if not keys:
        return {"delta": None, "n": 0, "ci95": None, "n_pos": 0, "n_neg": 0}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(reps):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "delta": sum(deltas) / n, "n": n,
        "ci95": [means[int(0.025 * (reps - 1))], means[int(0.975 * (reps - 1))]],
        "n_pos": sum(1 for value in deltas if value > 0),
        "n_neg": sum(1 for value in deltas if value < 0),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["utility", "success", "wrong_side_effect", "tool_error", "abort",
              "queries", "process_support", "final_correct", "process_reward", "outcome_reward"]
    result: dict[str, Any] = {"n_records": len(records),
                              "n_families": len({record["family_id"] for record in records})}
    for rep in ("N", "J"):
        rep_result: dict[str, Any] = {}
        for field in fields:
            values = list(record_family_values(records, field, rep).values())
            rep_result[field] = bootstrap_mean(values)
        for label, predicate in (
            ("all", lambda record: True),
            ("iid", lambda record: record["subset"] == "iid"),
            ("composition", lambda record: record["subset"] == "composition"),
            ("contact", lambda record: record["workflow"] == "contact"),
            ("reminder", lambda record: record["workflow"] == "reminder"),
            ("length2", lambda record: record["length"] == 2),
            ("length4", lambda record: record["length"] == 4),
            ("length6", lambda record: record["length"] == 6),
            ("initial_known", lambda record: record["initial_state"] == "known"),
            ("initial_unknown", lambda record: record["initial_state"] == "unknown"),
        ):
            subset = [record for record in records if predicate(record)]
            if subset:
                values = list(record_family_values(subset, "utility", rep).values())
                rep_result.setdefault("utility_by_cell", {})[label] = bootstrap_mean(values)
        result[rep] = rep_result
    step_correct: dict[str, list[float]] = defaultdict(list)
    step_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in records:
        if record["representation"] != "N":
            continue
        for step in record["steps"]:
            key = step.get("event_type", "?")
            step_counts[key]["total"] += 1
            if step["correct"]:
                step_counts[key]["correct"] += 1
    result["step_diagnostics_N"] = {key: {**counts,
                                          "rate": counts["correct"] / counts["total"] if counts["total"] else None}
                                    for key, counts in sorted(step_counts.items())}
    final_by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in records:
        if record["representation"] != "N":
            continue
        final_event = record["steps"][-1].get("event_type", "?")
        final_by_type[final_event]["total"] += 1
        if record["steps"][-1]["correct"]:
            final_by_type[final_event]["correct"] += 1
    result["final_event_diagnostics_N"] = {key: {**counts,
                                                 "rate": counts["correct"] / counts["total"] if counts["total"] else None}
                                           for key, counts in sorted(final_by_type.items())}
    return result


def evaluate_model(policy: Policy, cases: list[dict[str, Any]], worker, phase: str,
                   arm: str, seed: int | None, terminal_cache: dict[str, Any],
                   representations: tuple[str, ...] = ("J", "N")) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case in cases:
        for representation in representations:
            for world in case["worlds"]:
                def choose(messages, allowed, step, representation=representation, case=case, world=world):
                    return policy.sample(messages, allowed, sample=False, with_reference=False)
                try:
                    record = run_episode(worker, case, representation, world, choose,
                                         terminal_cache, execute_terminal=True)
                    record.update({"phase": phase, "arm": arm, "seed": seed,
                                   "run_id": None})
                    records.append(record)
                except Exception as error:
                    errors.append({"phase": phase, "arm": arm, "seed": seed,
                                   "family_id": case["family_id"], "representation": representation,
                                   "world_key": world["world_key"], "error_type": type(error).__name__,
                                   "error": str(error)})
                    if len(errors) >= 12:
                        return records, errors
    return records, errors


def evaluate_rule(cases: list[dict[str, Any]], worker, strategy: str, phase: str,
                  terminal_cache: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case in cases:
        for representation in ("J", "N"):
            for world in case["worlds"]:
                previous: list[str] = []
                def choose(messages, allowed, step, case=case, representation=representation):
                    if strategy == "QUERY_ONCE":
                        return {"label": UNKNOWN_LABEL}
                    if strategy == "DEPENDENCY_RULE":
                        prefix = case["prefixes"][step]
                        if not prefix["public_state_known"]:
                            return {"label": UNKNOWN_LABEL}
                        return {"label": prefix["correct_labels"][0]}
                    raise ValueError(strategy)
                try:
                    record = run_episode(worker, case, representation, world, choose,
                                         terminal_cache, execute_terminal=True)
                    record.update({"phase": phase, "arm": strategy, "seed": None, "run_id": None})
                    records.append(record)
                except Exception as error:
                    errors.append({"phase": phase, "arm": strategy, "family_id": case["family_id"],
                                   "representation": representation, "world_key": world["world_key"],
                                   "error_type": type(error).__name__, "error": str(error)})
                    if len(errors) >= 12:
                        return records, errors
    return records, errors


def metrics_from_records(records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summarize_records(records)
    if extra:
        summary.update(extra)
    return summary


# ---------------------------------------------------------------------------
# Training schedules and SFT.
# ---------------------------------------------------------------------------

def training_schedule(train_cases: list[dict[str, Any]], seed: int) -> list[list[dict[str, Any]]]:
    families = sorted(train_cases, key=lambda case: case["family_id"])
    if len(families) != 512:
        raise ValueError(f"expected 512 train families, found {len(families)}")
    rng = random.Random(seed)
    order = list(families)
    rng.shuffle(order)
    schedule = []
    for update in range(TRAIN_UPDATES):
        group = order[update * FAMILIES_PER_UPDATE:(update + 1) * FAMILIES_PER_UPDATE]
        entries = []
        for case in group:
            pick = random.Random(f"{seed}:{case['family_id']}:pick")
            representation = pick.choice(["J", "N"])
            world = pick.choice(case["worlds"])
            entries.append({"case": case, "representation": representation, "world": world})
        schedule.append(entries)
    return schedule


def make_sft_examples(train_cases: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for case in sorted(train_cases, key=lambda item: item["family_id"]):
        for representation in ("J", "N"):
            previous: list[str] = []
            for prefix in case["prefixes"]:
                label = prefix["correct_labels"][0]
                messages = build_action_prompt(case, prefix, representation, previous)
                examples.append({
                    "messages": messages, "allowed": allowed_labels(prefix),
                    "label": label, "family_id": case["family_id"],
                    "representation": representation, "step": prefix["step"],
                })
                previous.append(label)
    random.Random(seed).shuffle(examples)
    needed = TRAIN_UPDATES * SFT_EXAMPLES_PER_UPDATE
    if len(examples) < needed:
        raise ValueError(f"not enough SFT examples: {len(examples)} < {needed}")
    return examples[:needed]


def sft_forward_update(policy: Policy, optimizer, examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    policy.model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for start in range(0, len(examples), SFT_MICRO_BATCH):
        micro = examples[start:start + SFT_MICRO_BATCH]
        encoded = [policy.encode(example["messages"]) for example in micro]
        max_len = max(int(ids.shape[1]) for ids, _ in encoded)
        pad_id = policy.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = policy.tokenizer.eos_token_id
        input_ids = torch.full((len(micro), max_len), int(pad_id), dtype=torch.long,
                               device=policy.device)
        attention = torch.zeros((len(micro), max_len), dtype=torch.long, device=policy.device)
        for index, (ids, mask) in enumerate(encoded):
            length = int(ids.shape[1])
            input_ids[index, :length] = ids[0]
            attention[index, :length] = mask[0]
        logits = policy.model(input_ids=input_ids, attention_mask=attention,
                              use_cache=False).logits
        policy.model_calls += 1
        policy.input_tokens += int(attention.sum().item())
        total_tokens += int(attention.sum().item())
        last = attention.sum(dim=1) - 1
        batch_logits = logits[torch.arange(len(micro), device=policy.device), last]
        losses = []
        for index, example in enumerate(micro):
            allowed_ids = torch.tensor([policy.label_ids[label] for label in example["allowed"]],
                                       dtype=torch.long, device=policy.device)
            allowed_logits = batch_logits[index:index + 1, allowed_ids]
            target_index = example["allowed"].index(example["label"])
            target = torch.tensor([target_index], dtype=torch.long, device=policy.device)
            losses.append(F.cross_entropy(allowed_logits, target))
            with torch.no_grad():
                correct += int(torch.argmax(allowed_logits, dim=-1).item() == target_index)
        micro_loss = torch.stack(losses).mean() * (len(micro) / len(examples))
        micro_loss.backward()
        total_loss += float(micro_loss.detach().cpu())
    torch.nn.utils.clip_grad_norm_([parameter for parameter in policy.model.parameters()
                                    if parameter.requires_grad], 1.0)
    optimizer.step()
    return {"loss": total_loss, "samples": len(examples), "tokens": total_tokens,
            "action_accuracy": correct / len(examples) if examples else None}


def sft_learning_rate(optimizer, update: int) -> float:
    # Short SFT: constant LR; no hidden schedule.
    return SFT_LR


# ---------------------------------------------------------------------------
# RL rollout and group-relative update.
# ---------------------------------------------------------------------------

def rollout_for_training(policy: Policy, worker, case: dict[str, Any], representation: str,
                         world: dict[str, Any], terminal_cache: dict[tuple[str, str, str], Any]
                         ) -> dict[str, Any]:
    previous: list[str] = []
    actions: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for prefix in case["prefixes"]:
        messages = build_action_prompt(case, prefix, representation, previous)
        allowed = allowed_labels(prefix)
        decision = policy.sample(messages, allowed, sample=True, with_reference=True)
        label = decision["label"]
        correct = label in prefix["correct_labels"]
        actions.append({
            "input_ids": decision["input_ids"], "attention_mask": decision["attention_mask"],
            "allowed": list(allowed), "label": label, "old_logprob": decision["old_logprob"],
            "ref_logprobs": decision["ref_logprobs"], "step": prefix["step"],
        })
        steps.append({
            "step": prefix["step"], "label": label, "correct": bool(correct),
            "allowed": allowed, "final_event": prefix["step"] == len(case["prefixes"]) - 1,
            "event_type": case["events"][prefix["step"]]["type"],
            "public_state_known": prefix["public_state_known"],
        })
        previous.append(label)
    terminal = terminal_metrics(worker, case, world, steps[-1]["label"], terminal_cache)
    process_scores = [1.0 if step["correct"] else -1.0 for step in steps]
    process_mean = sum(process_scores) / len(process_scores) if process_scores else 0.0
    support = sum(1 for step in steps if step["correct"]) / len(steps) if steps else 0.0
    utility = float(terminal["utility"])
    return {
        "family_id": case["family_id"], "split": case["split"], "subset": case["subset"],
        "workflow": case["workflow"], "initial_state": case["initial_state"],
        "length": len(case["events"]), "representation": representation,
        "world_key": world["world_key"], "world_weight": world["weight"],
        "steps": steps, "actions": actions, "process_scores": process_scores,
        "process_support": support, "queries": terminal["queries"],
        "success": terminal["success"], "wrong_side_effect": terminal["wrong_side_effect"],
        "tool_error": terminal["tool_error"], "abort": terminal["abort"],
        "utility": utility, "outcome_reward": utility / 10.0,
        "process_reward": utility / 10.0 + 0.5 * process_mean,
        "final_label": steps[-1]["label"], "final_correct": steps[-1]["correct"],
        "terminal": terminal,
    }


def rl_update(policy: Policy, optimizer, rollouts: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    import torch
    reward_key = "outcome_reward" if arm == "OUTCOME" else "process_reward"
    rewards = [float(rollout[reward_key]) for rollout in rollouts]
    tensor_rewards = torch.tensor(rewards, dtype=torch.float32)
    mean_reward = float(tensor_rewards.mean().item())
    std_reward = float(tensor_rewards.std(unbiased=False).item())
    if std_reward < 1e-8:
        advantages = [0.0 for _ in rewards]
    else:
        advantages = [float((tensor_rewards[index] - mean_reward).item() / (std_reward + 1e-8))
                      for index in range(len(rewards))]
    policy.model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    action_count = 0
    kl_sum = 0.0
    entropy_sum = 0.0
    ratio_sum = 0.0
    for rollout, advantage in zip(rollouts, advantages):
        if abs(advantage) < 1e-12:
            continue
        for action in rollout["actions"]:
            allowed = action["allowed"]
            logprobs = policy.grad_logprobs(action["input_ids"], action["attention_mask"], allowed)
            index = allowed.index(action["label"])
            selected_logprob = logprobs[index]
            old = float(action["old_logprob"])
            ratio = torch.exp(selected_logprob - old)
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
            surrogate = -torch.min(ratio * advantage, clipped * advantage)
            ref = action["ref_logprobs"]
            if ref is None:
                raise RuntimeError("RL action missing reference logprobs")
            ref = ref.to(policy.device)
            probabilities = torch.exp(logprobs)
            kl = torch.sum(probabilities * (logprobs - ref))
            entropy = -torch.sum(probabilities * logprobs)
            loss = surrogate + KL_COEF * kl
            loss.backward()
            total_loss += float(loss.detach().cpu())
            action_count += 1
            kl_sum += float(kl.detach().cpu())
            entropy_sum += float(entropy.detach().cpu())
            ratio_sum += float(ratio.detach().cpu())
    if action_count:
        for parameter in policy.model.parameters():
            if parameter.grad is not None:
                parameter.grad /= action_count
        torch.nn.utils.clip_grad_norm_([parameter for parameter in policy.model.parameters()
                                        if parameter.requires_grad], 1.0)
        optimizer.step()
    return {
        "reward_mean": mean_reward, "reward_std": std_reward,
        "utility_mean": sum(float(r["utility"]) for r in rollouts) / len(rollouts),
        "success_rate": sum(1 for r in rollouts if r["success"]) / len(rollouts),
        "query_rate": sum(float(r["queries"]) for r in rollouts) / len(rollouts),
        "support_mean": sum(float(r["process_support"]) for r in rollouts) / len(rollouts),
        "zero_advantage": all(abs(value) < 1e-12 for value in advantages),
        "advantages": advantages, "loss": total_loss / action_count if action_count else 0.0,
        "kl": kl_sum / action_count if action_count else 0.0,
        "entropy": entropy_sum / action_count if action_count else 0.0,
        "ratio_mean": ratio_sum / action_count if action_count else 1.0,
        "actions": action_count,
        "rewards": rewards,
        "samples": len(rollouts),
    }


def rl_update_groups(policy: Policy, optimizer, groups: list[list[dict[str, Any]]],
                     arm: str) -> dict[str, Any]:
    """One optimizer step for four family groups; advantages stay within-family."""
    import torch
    reward_key = "outcome_reward" if arm == "OUTCOME" else "process_reward"
    policy.model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    action_count = 0
    kl_sum = entropy_sum = ratio_sum = 0.0
    all_rewards: list[float] = []
    all_utility: list[float] = []
    all_success: list[float] = []
    all_queries: list[float] = []
    all_support: list[float] = []
    zero_groups = 0
    group_means: list[float] = []
    for rollouts in groups:
        rewards = [float(rollout[reward_key]) for rollout in rollouts]
        all_rewards.extend(rewards)
        all_utility.extend(float(rollout["utility"]) for rollout in rollouts)
        all_success.extend(1.0 if rollout["success"] else 0.0 for rollout in rollouts)
        all_queries.extend(float(rollout["queries"]) for rollout in rollouts)
        all_support.extend(float(rollout["process_support"]) for rollout in rollouts)
        tensor_rewards = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = float(tensor_rewards.mean().item())
        std_reward = float(tensor_rewards.std(unbiased=False).item())
        group_means.append(mean_reward)
        if std_reward < 1e-8:
            advantages = [0.0 for _ in rewards]
            zero_groups += 1
        else:
            advantages = [float((tensor_rewards[index] - mean_reward).item() / (std_reward + 1e-8))
                          for index in range(len(rewards))]
        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-12:
                continue
            for action in rollout["actions"]:
                allowed = action["allowed"]
                logprobs = policy.grad_logprobs(action["input_ids"], action["attention_mask"], allowed)
                index = allowed.index(action["label"])
                selected_logprob = logprobs[index]
                ratio = torch.exp(selected_logprob - float(action["old_logprob"]))
                clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                surrogate = -torch.min(ratio * advantage, clipped * advantage)
                ref = action["ref_logprobs"]
                if ref is None:
                    raise RuntimeError("RL action missing reference logprobs")
                ref = ref.to(policy.device)
                probabilities = torch.exp(logprobs)
                kl = torch.sum(probabilities * (logprobs - ref))
                entropy = -torch.sum(probabilities * logprobs)
                loss = surrogate + KL_COEF * kl
                loss.backward()
                total_loss += float(loss.detach().cpu())
                action_count += 1
                kl_sum += float(kl.detach().cpu())
                entropy_sum += float(entropy.detach().cpu())
                ratio_sum += float(ratio.detach().cpu())
    if action_count:
        for parameter in policy.model.parameters():
            if parameter.grad is not None:
                parameter.grad /= action_count
        torch.nn.utils.clip_grad_norm_([parameter for parameter in policy.model.parameters()
                                        if parameter.requires_grad], 1.0)
        optimizer.step()
    return {
        "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else None,
        "group_reward_means": group_means,
        "utility_mean": sum(all_utility) / len(all_utility) if all_utility else None,
        "success_rate": sum(all_success) / len(all_success) if all_success else None,
        "query_rate": sum(all_queries) / len(all_queries) if all_queries else None,
        "support_mean": sum(all_support) / len(all_support) if all_support else None,
        "zero_advantage_groups": zero_groups,
        "loss": total_loss / action_count if action_count else 0.0,
        "kl": kl_sum / action_count if action_count else 0.0,
        "entropy": entropy_sum / action_count if action_count else 0.0,
        "ratio_mean": ratio_sum / action_count if action_count else 1.0,
        "actions": action_count,
        "samples": len(all_rewards),
    }


# ---------------------------------------------------------------------------
# Run lifecycle and artifacts.
# ---------------------------------------------------------------------------

def gpu_inventory() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], text=True)
    except Exception:
        output = ""
    return {"nvidia_smi": output.strip().splitlines()}


def run_id_for(arm: str, seed: int | None) -> str:
    if seed is None:
        return f"G7-{arm}-a01"
    return f"G7-{arm}-s{seed}-a01"


def run_directory(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_error(path: Path, error: dict[str, Any], existing: int) -> int:
    if existing >= 12:
        return existing
    append_jsonl(path, error)
    return existing + 1


def test_evaluation(policy: Policy, test_cases: list[dict[str, Any]], worker,
                    arm: str, seed: int | None, run_id: str, run_dir: Path,
                    terminal_cache: dict[tuple[str, str, str], Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.time()
    records, errors = evaluate_model(policy, test_cases, worker, "test", arm, seed, terminal_cache)
    for record in records:
        record["run_id"] = run_id
    predictions = run_dir / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    for error in errors:
        add_error(run_dir / "errors.jsonl", error, 0)
    metrics = summarize_records(records)
    metrics["phase_seconds"] = time.time() - started
    metrics["n_predictions"] = len(records)
    return metrics, errors


def dev_evaluation(policy: Policy, dev_cases: list[dict[str, Any]], worker,
                   arm: str, seed: int | None, update: int,
                   terminal_cache: dict[tuple[str, str, str], Any]) -> dict[str, Any]:
    old_mode = policy.model.training
    policy.model.eval()
    records, errors = evaluate_model(policy, dev_cases, worker, "dev", arm, seed, terminal_cache)
    policy.model.train(old_mode)
    metrics = summarize_records(records)
    metrics["update"] = update
    metrics["n_errors"] = len(errors)
    return metrics


def base_run(arm: str, seed: int | None, device: str, run_id: str) -> dict[str, Any]:
    return {
        "attempt_id": run_id, "protocol_id": PROTOCOL_ID, "task_prompt_version": TASK_PROMPT_VERSION,
        "arm": arm, "seed": seed, "status": "running", "started_at": now_cst(),
        "pid": os.getpid(), "device": device, "gpu_inventory_start": gpu_inventory(),
        "model": {"snapshot": str(QWEN_SNAPSHOT), "dtype": "bfloat16",
                  "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                           "target_modules": ["q_proj", "v_proj"],
                           "init": "seeded gaussian lora_B", "init_std": INIT_STD,
                           "init_seed": seed},
                  "kl_reference": ("frozen initial adapter policy for RL"
                                   if arm in {"OUTCOME", "PROCESS"}
                                   else "not applicable (SFT)" if arm == "PROCESS_SFT"
                                   else "base model")},
        "data": {"cases": str(DATA_DIR / "cases.jsonl"),
                 "manifest": str(DATA_DIR / "manifest.json")},
        "training": {"updates": TRAIN_UPDATES, "families_per_update": FAMILIES_PER_UPDATE,
                     "group_size": GROUP_SIZE,
                     "sft_lr": SFT_LR, "rl_lr": RL_LR, "kl_coef": KL_COEF, "clip_eps": CLIP_EPS},
        "dev_curve": [], "test": None, "adapter": None, "cost": {"started_unix": time.time()},
        "errors": [], "notes": {
            "main_comparison": "composition test native utility PROCESS - OUTCOME",
            "test_used_only_final": True,
            "checkpoint_rule": "no optimizer/RNG/tokenizer/checkpoint copies; only final adapter",
        },
    }


def finish_run(run: dict[str, Any], run_dir: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    run["status"] = status
    run["finished_at"] = now_cst()
    run["cost"]["wall_seconds"] = time.time() - run["cost"].get("started_unix", time.time())
    run["cost"]["gpu_hours"] = run["cost"]["wall_seconds"] / 3600.0
    run["gpu_inventory_end"] = gpu_inventory()
    if extra:
        run.update(extra)
    write_json(run_dir / "run.json", run)


def load_policy(device: str, adapter: str | Path | None = None, trainable: bool = False,
                seed: int | None = None, init_std: float = INIT_STD) -> Policy:
    return Policy(device=device, adapter=adapter, trainable=trainable, init_seed=seed,
                  init_std=init_std)


def build_reference_policy(policy: Policy, device: str, temp_dir: Path) -> Policy:
    """Freeze the seeded initial adapter as the KL reference policy.

    Zero LoRA-B initialization gives a near-deterministic base and zero-variance
    four-rollout groups.  The seeded nonzero init is part of the initial policy;
    regularizing to the *base* would erase that exploration.  The reference here
    is therefore the frozen initial adapter, which is the standard initial-policy
    reference.  The temporary adapter is deleted after loading; only the final
    policy adapter is retained.
    """
    policy.save_adapter(temp_dir)
    reference = Policy(device=device, adapter=temp_dir, trainable=False)
    reference.model.eval()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return reference


def run_sft(run_id: str, seed: int, device: str, max_seconds: float = 5400.0) -> None:
    from g7_data import G7Worker
    cases = load_cases()
    train_cases = cases_by_split(cases, "train")
    dev_cases = select_dev_cases(cases)
    test_cases = cases_by_split(cases, "test")
    run_dir = run_directory(run_id)
    log_path = run_dir / "run.log"
    run = base_run("PROCESS_SFT", seed, device, run_id)
    run["data"]["manifest_sha256"] = sha256_file(DATA_DIR / "manifest.json")
    run["data"]["cases_sha256"] = sha256_file(DATA_DIR / "cases.jsonl")
    write_json(run_dir / "run.json", run)
    worker = G7Worker()
    started = time.time()
    try:
        policy = load_policy(device, trainable=True, seed=seed, init_std=0.0)
        run["model"]["lora"]["init"] = "zero lora_B (standard SFT)"
        run["model"]["lora"]["init_std"] = 0.0
        optimizer = policy.torch.optim.AdamW(
            [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
            lr=SFT_LR,
        )
        examples = make_sft_examples(train_cases, seed)
        run["training"]["kl_coef"] = None
        run["training"]["kl_reference"] = "not applicable (SFT)"
        run["training"]["sft_examples_total"] = len(examples)
        run["training"]["sft_examples_per_update"] = SFT_EXAMPLES_PER_UPDATE
        run["training"]["sft_micro_batch"] = SFT_MICRO_BATCH
        write_json(run_dir / "run.json", run)
        terminal_cache: dict[tuple[str, str, str], Any] = {}
        dev_records_all: list[dict[str, Any]] = []
        metrics_path = run_dir / "metrics.jsonl"
        errors_path = run_dir / "errors.jsonl"
        errors = 0
        update = 0
        sft_train_tokens = 0
        sft_train_samples = 0
        for update in range(1, TRAIN_UPDATES + 1):
            batch = examples[(update - 1) * SFT_EXAMPLES_PER_UPDATE:
                             update * SFT_EXAMPLES_PER_UPDATE]
            calls_before = policy.model_calls
            tokens_before = policy.input_tokens
            update_metrics = sft_forward_update(policy, optimizer, batch)
            update_metrics.update({
                "update": update, "elapsed": time.time() - started,
                "model_calls": policy.model_calls - calls_before,
                "input_tokens": policy.input_tokens - tokens_before,
                "output_tokens": 0, "lr": SFT_LR, "arm": "PROCESS_SFT", "seed": seed,
            })
            sft_train_tokens += int(update_metrics["input_tokens"])
            sft_train_samples += int(update_metrics["samples"])
            append_jsonl(metrics_path, update_metrics)
            if update in {32, 64, 128}:
                dev_metrics = dev_evaluation(policy, dev_cases, worker, "PROCESS_SFT", seed,
                                             update, terminal_cache)
                run["dev_curve"].append(dev_metrics)
                run["training"]["updates_completed"] = update
                run["cost"]["worker_calls"] = worker.calls
                run["cost"]["model_calls"] = policy.model_calls
                run["cost"]["input_tokens"] = policy.input_tokens
                write_json(run_dir / "run.json", run)
            if time.time() - started > max_seconds:
                run["training"]["stopped_by_time_cap"] = True
                break
        run["training"]["updates_completed"] = update
        run["training"]["sft_samples_seen"] = sft_train_samples
        run["training"]["sft_tokens_seen"] = sft_train_tokens
        run["training"]["sft_action_tokens"] = sft_train_samples
        run["training"]["sft_action_token_only"] = True
        policy.model.eval()
        run["adapter"] = policy.save_adapter(run_dir / "adapter")
        write_json(run_dir / "run.json", run)
        run["cost"]["eval_started_unix"] = time.time()
        test_metrics, test_errors = test_evaluation(policy, test_cases, worker, "PROCESS_SFT",
                                                    seed, run_id, run_dir, terminal_cache)
        run["test"] = test_metrics
        for error in test_errors[:12]:
            errors = add_error(errors_path, error, errors)
        run["errors"] = test_errors
        run["cost"]["worker_calls"] = worker.calls
        run["cost"]["model_calls"] = policy.model_calls
        run["cost"]["input_tokens"] = policy.input_tokens
        run["cost"]["eval_seconds"] = time.time() - run["cost"]["eval_started_unix"]
        finish_run(run, run_dir, "completed")
    except Exception as error:
        run["errors"] = [{"error_type": type(error).__name__, "error": str(error)}]
        finish_run(run, run_dir, "failed")
        raise
    finally:
        worker.close()


def run_rl(run_id: str, arm: str, seed: int, device: str, max_seconds: float = 5400.0) -> None:
    from g7_data import G7Worker
    if arm not in {"OUTCOME", "PROCESS"}:
        raise ValueError(arm)
    cases = load_cases()
    train_cases = cases_by_split(cases, "train")
    dev_cases = select_dev_cases(cases)
    test_cases = cases_by_split(cases, "test")
    run_dir = run_directory(run_id)
    run = base_run(arm, seed, device, run_id)
    run["data"]["manifest_sha256"] = sha256_file(DATA_DIR / "manifest.json")
    run["data"]["cases_sha256"] = sha256_file(DATA_DIR / "cases.jsonl")
    write_json(run_dir / "run.json", run)
    worker = G7Worker()
    started = time.time()
    try:
        policy = load_policy(device, trainable=True, seed=seed)
        reference_policy = build_reference_policy(
            policy, device, run_dir / ".reference_initial_adapter")
        policy.reference_model = reference_policy
        run["model"]["kl_reference"] = "frozen initial adapter policy (seed-matched)"
        optimizer = policy.torch.optim.AdamW(
            [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
            lr=RL_LR,
        )
        schedule = training_schedule(train_cases, seed)
        terminal_cache: dict[tuple[str, str, str], Any] = {}
        metrics_path = run_dir / "metrics.jsonl"
        errors_path = run_dir / "errors.jsonl"
        errors = 0
        update = 0
        action_tokens_seen = 0
        rollouts_seen = 0
        for update in range(1, TRAIN_UPDATES + 1):
            groups = []
            calls_before = policy.model_calls
            tokens_before = policy.input_tokens
            for entry in schedule[update - 1]:
                rollouts = [
                    rollout_for_training(policy, worker, entry["case"],
                                         entry["representation"], entry["world"], terminal_cache)
                    for _ in range(GROUP_SIZE)
                ]
                groups.append(rollouts)
            update_metrics = rl_update_groups(policy, optimizer, groups, arm)
            update_metrics.update({
                "update": update, "elapsed": time.time() - started,
                "model_calls": policy.model_calls - calls_before,
                "input_tokens": policy.input_tokens - tokens_before,
                "output_tokens": sum(len(group[0]["actions"]) * len(group) for group in groups),
                "lr": RL_LR, "arm": arm, "seed": seed,
                "families": [entry["case"]["family_id"] for entry in schedule[update - 1]],
                "family_rewards": [[float(rollout["utility"]) for rollout in group] for group in groups],
                "family_process_support": [[float(rollout["process_support"]) for rollout in group]
                                           for group in groups],
            })
            action_tokens_seen += int(update_metrics["actions"])
            rollouts_seen += int(update_metrics["samples"])
            append_jsonl(metrics_path, update_metrics)
            if update in {32, 64, 128}:
                dev_metrics = dev_evaluation(policy, dev_cases, worker, arm, seed, update,
                                             terminal_cache)
                run["dev_curve"].append(dev_metrics)
                run["training"]["updates_completed"] = update
                run["cost"]["worker_calls"] = worker.calls
                run["cost"]["model_calls"] = policy.model_calls
                run["cost"]["input_tokens"] = policy.input_tokens
                write_json(run_dir / "run.json", run)
            if time.time() - started > max_seconds:
                run["training"]["stopped_by_time_cap"] = True
                break
        run["training"]["updates_completed"] = update
        run["training"]["rollouts_seen"] = rollouts_seen
        run["training"]["action_tokens_seen"] = action_tokens_seen
        policy.reference_model = None
        del reference_policy
        policy.torch.cuda.empty_cache()
        policy.model.eval()
        run["adapter"] = policy.save_adapter(run_dir / "adapter")
        write_json(run_dir / "run.json", run)
        run["cost"]["eval_started_unix"] = time.time()
        test_metrics, test_errors = test_evaluation(policy, test_cases, worker, arm, seed,
                                                    run_id, run_dir, terminal_cache)
        run["test"] = test_metrics
        for error in test_errors[:12]:
            errors = add_error(errors_path, error, errors)
        run["errors"] = test_errors
        run["cost"]["worker_calls"] = worker.calls
        run["cost"]["model_calls"] = policy.model_calls
        run["cost"]["input_tokens"] = policy.input_tokens
        run["cost"]["eval_seconds"] = time.time() - run["cost"]["eval_started_unix"]
        finish_run(run, run_dir, "completed")
    except Exception as error:
        run["errors"] = [{"error_type": type(error).__name__, "error": str(error)}]
        finish_run(run, run_dir, "failed")
        raise
    finally:
        worker.close()


def run_baselines(device: str, max_seconds: float = 7200.0) -> None:
    from g7_data import G7Worker
    cases = load_cases()
    test_cases = cases_by_split(cases, "test")
    run_id = "G7-BASELINES-a01"
    run_dir = run_directory(run_id)
    run = {
        "attempt_id": run_id, "protocol_id": PROTOCOL_ID, "status": "running",
        "started_at": now_cst(), "pid": os.getpid(), "device": device,
        "gpu_inventory_start": gpu_inventory(),
        "data": {"cases": str(DATA_DIR / "cases.jsonl"),
                 "manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
                 "cases_sha256": sha256_file(DATA_DIR / "cases.jsonl")},
        "baselines": {}, "cost": {"started_unix": time.time()}, "errors": [],
    }
    write_json(run_dir / "run.json", run)
    worker = G7Worker()
    started = time.time()
    all_records: list[dict[str, Any]] = []
    try:
        terminal_cache: dict[tuple[str, str, str], Any] = {}
        base_policy = Policy(device=device, adapter=None, trainable=False)
        base_records, base_errors = evaluate_model(base_policy, test_cases, worker, "test",
                                                   "BASE_POINTER", None, terminal_cache)
        for record in base_records:
            record["run_id"] = run_id
            record["arm"] = "BASE_POINTER"
        run["baselines"]["BASE_POINTER"] = {
            "metrics": summarize_records(base_records),
            "model_calls": base_policy.model_calls,
            "input_tokens": base_policy.input_tokens,
            "seconds": time.time() - started,
        }
        all_records.extend(base_records)
        for strategy in ("QUERY_ONCE", "DEPENDENCY_RULE"):
            strategy_started = time.time()
            records, errors = evaluate_rule(test_cases, worker, strategy, "test", terminal_cache)
            for record in records:
                record["run_id"] = run_id
            run["baselines"][strategy] = {
                "metrics": summarize_records(records),
                "seconds": time.time() - strategy_started,
                "model_calls": 0, "input_tokens": 0,
            }
            all_records.extend(records)
            for error in errors:
                run["errors"].append(error)
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in all_records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        if base_errors:
            for error in base_errors[:12]:
                append_jsonl(run_dir / "errors.jsonl", error)
        run["status"] = "completed"
        run["finished_at"] = now_cst()
        run["cost"]["wall_seconds"] = time.time() - started
        run["cost"]["gpu_hours"] = run["cost"]["wall_seconds"] / 3600.0
        run["cost"]["worker_calls"] = worker.calls
        run["gpu_inventory_end"] = gpu_inventory()
        write_json(run_dir / "run.json", run)
    except Exception as error:
        run["status"] = "failed"
        run["errors"] = [{"error_type": type(error).__name__, "error": str(error)}]
        run["finished_at"] = now_cst()
        write_json(run_dir / "run.json", run)
        raise
    finally:
        worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="G7 pointer-action policy training/evaluation")
    parser.add_argument("--mode", choices=["train", "baselines", "summary", "prepare", "data-check"],
                        required=True)
    parser.add_argument("--arm", choices=["OUTCOME", "PROCESS", "PROCESS_SFT"])
    parser.add_argument("--seed", type=int, choices=[11, 23])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=5400.0)
    args = parser.parse_args()
    if args.mode == "train":
        if args.arm is None or args.seed is None:
            parser.error("--arm and --seed are required for train")
        if args.arm == "PROCESS_SFT":
            run_sft(run_id_for(args.arm, args.seed), args.seed, args.device,
                    max_seconds=args.max_seconds)
        else:
            run_rl(run_id_for(args.arm, args.seed), args.arm, args.seed, args.device,
                   max_seconds=args.max_seconds)
    elif args.mode == "baselines":
        run_baselines(args.device, max_seconds=args.max_seconds)
    elif args.mode == "data-check":
        import g7_data
        manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    elif args.mode == "prepare":
        prepare_coordinator()
    elif args.mode == "summary":
        summarize_all()
    else:
        parser.error("unsupported mode")




# ---------------------------------------------------------------------------
# G6 correction, data checks, coordinator.
# ---------------------------------------------------------------------------

def g6_correction() -> dict[str, Any]:
    import validity_experiment as V
    g6_run = json.loads((ROOT / "runs" / "G6-VALIDITY-a01" / "run.json").read_text())
    cases, nodes, preds, trajectories = V._node_rows()
    audit = [node for node in nodes if node["split"] == "audit"]
    analysis = V.analyze_nodes(audit, preds)
    strict = {}
    for method, records in (("R", preds["review_generic"]), ("V", preds["review_validity"])):
        for representation in ("J", "N"):
            joint = 0
            for family_id in sorted({node["family_id"] for node in audit}):
                workflow = next(case["workflow"] for case in cases if case["family_id"] == family_id)
                good = True
                for condition in V.CONDITIONS:
                    node = next(node for node in audit if node["family_id"] == family_id
                                and node["representation"] == representation
                                and node["stage"] == "root" and node["condition"] == condition)
                    good = good and V._state_correct(workflow, records[node["node_id"]]["parsed"], node)
                joint += int(good)
            strict[f"{method}_{representation}_joint"] = joint
        for representation in ("J", "N"):
            for workflow in ("contact", "reminder"):
                selected = [node for node in audit if node["workflow"] == workflow
                            and node["representation"] == representation
                            and node["stage"] == "root" and node["condition"] == "U"]
                strict[f"{method}_{representation}_{workflow}_U"] = sum(
                    V._state_correct(workflow, records[node["node_id"]]["parsed"], node)
                    for node in selected)
                strict[f"{method}_{representation}_{workflow}_U_total"] = len(selected)
    raw = {}
    invalid_counts: dict[str, Any] = {}
    all_rows = read_jsonl(ROOT / "runs" / "G6-VALIDITY-a01" / "predictions.jsonl")
    for method, record_type in (("R", "review_generic"), ("V", "review_validity")):
        relations = Counter()
        parser_relations = Counter()
        invalid = 0
        for row in all_rows:
            if row.get("record_type") != record_type:
                continue
            parsed = row.get("parsed") or {}
            if not parsed.get("valid", False):
                invalid += 1
                try:
                    raw_json = json.loads(row.get("raw_output", "{}"))
                except Exception:
                    raw_json = {}
                relations[raw_json.get("relation")] += 1
                parser_relations[parsed.get("relation")] += 1
        invalid_counts[method] = {
            "invalid_total": invalid,
            "invalid_raw_relation_counts": dict(relations),
            "invalid_parser_relation_counts": dict(parser_relations),
        }
    comparisons = g6_run["metrics"]["g6"]["strategies"]["comparisons"]
    v_minus_r = comparisons.get("VALIDITY_REVIEW__minus__GENERIC_REVIEW", {})
    return {
        "source": "recomputed on frozen G6 cases/predictions; no G6 GPU rerun",
        "strict_validity_fix": {"rule": "parsed.valid must be true before knowledge/value counts",
                                "joint": strict},
        "raw_relation_fix": {"rule": "diagnostics read the original raw JSON relation, not parser placeholder",
                             "invalid_counts": invalid_counts},
        "main_N_joint_V_minus_R_corrected": analysis.get("main_N_joint_V_minus_R"),
        "V_minus_R_utility_and_success_unchanged": {
            "utility": v_minus_r.get("utility"),
            "success": v_minus_r.get("success"),
            "note": "deployment fields are unchanged; strict joint state metric is the corrected one",
        },
        "U_strict": {key: value for key, value in strict.items() if key.endswith("_U") or key.endswith("_U_total")},
        "support_source_cache_post_hoc": {
            "N": {"utility": 9.59375, "success": 1.0, "wrong_side_effect": 0.0, "queries": 0.40625},
            "J": {"utility": 4.947916666666667, "success": 0.875,
                  "wrong_side_effect": 0.109375, "queries": 0.40625},
            "status": "post-hoc diagnostic supplied by the G7 protocol; not used in the G7 main comparison",
        },
        "g6_original_run_untouched": True,
        "g6_predictions_sha256": sha256_file(ROOT / "runs" / "G6-VALIDITY-a01" / "predictions.jsonl"),
        "g6_run_sha256": sha256_file(ROOT / "runs" / "G6-VALIDITY-a01" / "run.json"),
    }


def data_checks() -> dict[str, Any]:
    cases = load_cases()
    errors: list[str] = []
    candidates_equal = 0
    hidden_leaks = 0
    correct_subset = 0
    unknown_rules = 0
    world_weight_ok = 0
    source_visible = 0
    total_prefixes = 0
    initial_known = initial_unknown = 0
    for case in cases:
        if case["initial_state"] == "known":
            initial_known += 1
        else:
            initial_unknown += 1
        for prefix in case["prefixes"]:
            total_prefixes += 1
            labels = [candidate["label"] for candidate in prefix["candidates"]]
            if len(labels) == len(set(labels)):
                candidates_equal += 1
            else:
                errors.append(f"duplicate candidate label {case['family_id']} {prefix['step']}")
            if set(prefix["correct_labels"]).issubset(set(labels) | {UNKNOWN_LABEL}):
                correct_subset += 1
            else:
                errors.append(f"correct label not offered {case['family_id']} {prefix['step']}")
            if (not prefix["public_state_known"] and prefix["correct_labels"] == [UNKNOWN_LABEL]):
                unknown_rules += 1
            if prefix["public_state_known"]:
                source_visible += 1
            for event in case["events"]:
                if event["type"] == "U":
                    for value in event.get("values", []):
                        if str(value) in prefix["history_j"] or str(value) in prefix["history_n"]:
                            hidden_leaks += 1
        weight = sum(float(world["weight"]) for world in case["worlds"])
        if abs(weight - 1.0) <= 1e-9:
            world_weight_ok += 1
        else:
            errors.append(f"world weights {case['family_id']} {weight}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_SNAPSHOT), local_files_only=True)
    label_tokens = {}
    for label in LABELS + UNKNOWN_LABEL:
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_tokens[label] = ids
        if len(ids) != 1:
            errors.append(f"label token not single: {label} {ids}")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    return {
        "ok": not errors,
        "errors": errors[:20],
        "n_families": len(cases),
        "n_prefixes": total_prefixes,
        "n_initial_known": initial_known,
        "n_initial_unknown": initial_unknown,
        "n_prefix_known": source_visible,
        "n_prefix_unknown": total_prefixes - source_visible,
        "unknown_rule_prefixes": unknown_rules,
        "hidden_u_value_leaks_into_public_history": hidden_leaks,
        "candidate_labels_unique": candidates_equal == total_prefixes,
        "correct_labels_always_offered": correct_subset == total_prefixes,
        "world_weights_sum_one": world_weight_ok == len(cases),
        "label_token_ids": label_tokens,
        "manifest": {key: manifest.get(key) for key in (
            "dataset_version", "n_families", "n_prefixes", "n_worlds", "event_type_counts",
            "initial_state_counts", "final_known_counts", "candidate_count_histogram",
            "world_count_histogram", "length_counts", "cases_sha256", "cases_bytes",
            "generation_worker_calls", "test_surface_isolated")},
        "candidate_order": "one fixed family order; same J/N order; labels A..J; Z=UNKNOWN",
    }


def prepare_coordinator() -> None:
    coordinator_dir = RUNS_DIR / "G7-COORD-a01"
    coordinator_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "attempt_id": "G7-COORD-a01", "protocol_id": PROTOCOL_ID,
        "task_prompt_version": TASK_PROMPT_VERSION,
        "status": "prepared", "started_at": now_cst(),
        "new_source_files": ["g7_data.py", "g7_process.py"],
        "data": data_checks(),
        "g6_correction": g6_correction(),
        "planned": {
            "runs": [f"{arm}-s{seed}" for arm in ("OUTCOME", "PROCESS", "PROCESS_SFT")
                     for seed in TRAIN_SEEDS],
            "baselines": ["BASE_POINTER", "QUERY_ONCE", "DEPENDENCY_RULE"],
            "main_comparison": "composition test, N representation, native utility PROCESS - OUTCOME",
            "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                          "unit": "family", "per_seed": True},
        },
        "resource_limits": {"max_gpus": 2, "max_gpu_hours": 12,
                            "max_gpu_hours_per_run": 1.5, "max_eval_gpu_hours": 2,
                            "max_adapter_bytes_each": 128 * 1024 * 1024},
        "baselines": None, "runs": {}, "comparisons": None,
        "finished_at": None,
    }
    write_json(coordinator_dir / "run.json", run)
    print(json.dumps({"coordinator": str(coordinator_dir / "run.json"),
                      "data_ok": run["data"]["ok"],
                      "data_errors": run["data"]["errors"],
                      "g6_joint": run["g6_correction"]["strict_validity_fix"]["joint"]},
                     indent=2, ensure_ascii=False))


def _load_predictions(run_id: str) -> list[dict[str, Any]]:
    path = RUNS_DIR / run_id / "predictions.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def _test_family_metrics(records: list[dict[str, Any]], field: str, representation: str = "N",
                         subset: str | None = "composition") -> dict[str, float]:
    return record_family_values(records, field, representation, subset)


def _run_summary(run_id: str) -> dict[str, Any]:
    path = RUNS_DIR / run_id / "run.json"
    if not path.exists():
        return {"run_id": run_id, "status": "missing"}
    run = json.loads(path.read_text())
    return {
        "run_id": run_id, "arm": run.get("arm"), "seed": run.get("seed"),
        "status": run.get("status"), "training": run.get("training"),
        "adapter": run.get("adapter"), "cost": run.get("cost"),
        "dev_curve": run.get("dev_curve"),
        "test": run.get("test"),
        "errors": run.get("errors", [])[:5],
    }


def summarize_all() -> None:
    coordinator_path = RUNS_DIR / "G7-COORD-a01" / "run.json"
    coordinator = json.loads(coordinator_path.read_text()) if coordinator_path.exists() else {}
    coordinator.setdefault("attempt_id", "G7-COORD-a01")
    coordinator.setdefault("protocol_id", PROTOCOL_ID)
    runs: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for arm in ("OUTCOME", "PROCESS", "PROCESS_SFT"):
        for seed in TRAIN_SEEDS:
            run_id = run_id_for(arm, seed)
            runs[run_id] = _run_summary(run_id)
            predictions[run_id] = _load_predictions(run_id)
    baseline_path = RUNS_DIR / "G7-BASELINES-a01" / "run.json"
    baseline_run = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    baseline_predictions = _load_predictions("G7-BASELINES-a01")
    baseline_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in baseline_predictions:
        baseline_records[record.get("arm")].append(record)
    comparisons: dict[str, Any] = {}
    for seed in TRAIN_SEEDS:
        process_id = run_id_for("PROCESS", seed)
        outcome_id = run_id_for("OUTCOME", seed)
        process_records = predictions.get(process_id, [])
        outcome_records = predictions.get(outcome_id, [])
        for field in ("utility", "success", "wrong_side_effect", "queries",
                      "process_support", "final_correct"):
            comparisons.setdefault(f"seed{seed}_composition_N", {})[field] = paired_bootstrap(
                _test_family_metrics(process_records, field),
                _test_family_metrics(outcome_records, field),
            )
        for subset in ("iid", "composition"):
            comparisons.setdefault(f"seed{seed}_{subset}_N_process", {})["utility"] = bootstrap_mean(
                list(_test_family_metrics(process_records, "utility", subset=subset).values()))
            comparisons.setdefault(f"seed{seed}_{subset}_N_outcome", {})["utility"] = bootstrap_mean(
                list(_test_family_metrics(outcome_records, "utility", subset=subset).values()))
    # Seed means use the two independent paired point estimates.
    seed_means = {}
    for field in ("utility", "success", "wrong_side_effect", "queries", "process_support", "final_correct"):
        deltas = [comparisons[f"seed{seed}_composition_N"][field]["delta"] for seed in TRAIN_SEEDS]
        seed_means[field] = {"mean": sum(deltas) / len(deltas), "deltas": deltas}
    comparisons["seed_mean_composition_N"] = seed_means
    # Baselines and trained test table.
    table: dict[str, Any] = {}
    for run_id, records in predictions.items():
        if records:
            table[run_id] = summarize_records(records)
    for arm, records in baseline_records.items():
        if records:
            table[f"G7-BASELINES-a01/{arm}"] = summarize_records(records)
    if baseline_run:
        for arm, summary in baseline_run.get("baselines", {}).items():
            table.setdefault(f"G7-BASELINES-a01/{arm}", summary.get("metrics"))
    dynamics = {}
    for run_id, run_summary in runs.items():
        path = RUNS_DIR / run_id / "metrics.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        if str(run_summary.get("arm", "")).startswith("PROCESS_SFT"):
            accuracies = [row.get("action_accuracy") for row in rows
                          if row.get("action_accuracy") is not None]
            dynamics[run_id] = {
                "kind": "SFT", "updates": len(rows), "last_update": rows[-1],
                "sft_loss_mean": sum(row.get("loss", 0.0) for row in rows) / len(rows),
                "sft_loss_first": rows[0].get("loss"),
                "sft_loss_last": rows[-1].get("loss"),
                "action_accuracy_mean": (sum(accuracies) / len(accuracies)) if accuracies else None,
                "action_accuracy_last": rows[-1].get("action_accuracy"),
                "samples": sum(int(row.get("samples", 0)) for row in rows),
                "tokens": sum(int(row.get("tokens", 0)) for row in rows),
                "notes": "SFT rows do not have group/RL reward fields; zero/utility fields are not applicable",
            }
        else:
            actions = sum(int(row.get("actions", 0)) for row in rows)
            weighted_entropy = (sum(float(row.get("entropy", 0.0)) * int(row.get("actions", 0))
                                    for row in rows) / actions) if actions else None
            weighted_kl = (sum(float(row.get("kl", 0.0)) * int(row.get("actions", 0))
                               for row in rows) / actions) if actions else None
            dynamics[run_id] = {
                "kind": "RL", "updates": len(rows), "last_update": rows[-1],
                "zero_advantage_groups_mean": sum(row.get("zero_advantage_groups", 0) for row in rows) / len(rows),
                "zero_advantage_updates": sum(row.get("zero_advantage_groups", 0) >= FAMILIES_PER_UPDATE for row in rows),
                "loss_mean": sum(row.get("loss", 0.0) for row in rows) / len(rows),
                "kl_mean": sum(row.get("kl", 0.0) for row in rows) / len(rows),
                "kl_action_weighted": weighted_kl,
                "entropy_mean_unweighted": sum(row.get("entropy", 0.0) for row in rows) / len(rows),
                "entropy_action_weighted": weighted_entropy,
                "support_mean": sum(row.get("support_mean", 0.0) for row in rows) / len(rows),
                "utility_mean": sum(row.get("utility_mean", 0.0) for row in rows) / len(rows),
                "actions": actions,
                "notes": "zero-group updates contribute entropy=0 in per-row mean; use action-weighted entropy",
            }
    all_runs_complete = all(run.get("status") == "completed" for run in runs.values())
    baseline_complete = bool(baseline_run and baseline_run.get("status") == "completed")
    total_gpu_hours = sum(float((run.get("cost") or {}).get("gpu_hours") or 0) for run in runs.values())
    if baseline_run:
        total_gpu_hours += float((baseline_run.get("cost") or {}).get("gpu_hours") or 0)
    coordinator.update({
        "updated_at": now_cst(),
        "status": "completed" if all_runs_complete and baseline_complete else "partial",
        "runs": runs,
        "baselines": {
            "status": baseline_run.get("status") if baseline_run else "missing",
            "metrics": {arm: summary.get("metrics") for arm, summary in
                        (baseline_run or {}).get("baselines", {}).items()},
            "cost": (baseline_run or {}).get("cost"),
        },
        "test_table": table,
        "comparisons": comparisons,
        "training_dynamics": dynamics,
        "cost": {
            "known_gpu_hours": total_gpu_hours,
            "within_12_gpu_hours": total_gpu_hours <= 12.0,
            "max_gpu_hours_per_run": max(
                [float((run.get("cost") or {}).get("gpu_hours") or 0) for run in runs.values()] or [0.0]),
        },
        "finished_at": now_cst(),
    })
    if all_runs_complete and baseline_complete:
        coordinator["status"] = "completed"
    coordinator["notes"] = {
        "main_comparison": "composition test N representation, PROCESS - OUTCOME native utility",
        "primary_inference": "per-seed family-paired bootstrap (2000, seed23); seed mean is descriptive",
        "test_predictions_final_only": True,
        "g6_no_gpu_rerun": True,
        "support_source_cache_post_hoc": True,
    }
    write_json(coordinator_path, coordinator)
    print(json.dumps({
        "status": coordinator["status"],
        "comparisons": comparisons.get("seed_mean_composition_N"),
        "seed11": comparisons.get("seed11_composition_N"),
        "seed23": comparisons.get("seed23_composition_N"),
        "gpu_hours": total_gpu_hours,
    }, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
