"""Generate matched conversational DPO pairs from a finite CPU oracle.

Both label modes receive identical prompts and unordered candidate pairs.
If either teacher is indifferent, that pair is excluded from BOTH DPO sets
and preserved in the audit file. All variants/pairs of a scenario share a
split. Labels/teacher scores are never placed in the model prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from core import Inspection, Scenario, first_action_values, first_actions, pair_preference

TEMPLATES = (
    ("pump", "You are deciding how to service a pump with one of two mutually exclusive faults."),
    ("cooling unit", "Plan a safe, cost-aware service action for a cooling unit whose fault type is uncertain."),
    ("inspection instrument", "Choose the first operation in a two-stage maintenance task for an inspection instrument."),
)


def scenarios() -> list[Scenario]:
    return [
        Scenario(p, delta, (Inspection("quick", q_quick, c_quick), Inspection("precise", q_precise, c_precise)))
        for p in (0.1, 0.25, 0.5, 0.75, 0.9)
        for delta in (0.0, 0.05, 0.1, 0.2, 0.35)
        for q_quick in (0.6, 0.75, 0.9)
        for q_precise in (0.85, 0.95, 1.0)
        if q_precise >= q_quick
        for c_quick in (0.25, 1.0)
        for c_precise in (1.5, 3.0)
    ]


def split_scenario(scenario: Scenario, seed: int = 17) -> tuple[str, str]:
    """OOD drift and reliability combinations are absent from training/validation."""
    if scenario.flip_probability >= 0.2:
        return "test", "ood_drift"
    reliability = tuple(test.accuracy for test in scenario.inspections)
    if reliability == (0.75, 0.95):
        return "test", "ood_reliability_combination"
    digest = hashlib.sha256(f"{seed}:{scenario.scenario_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < 0.75:
        return "train", "in_distribution"
    if fraction < 0.875:
        return "validation", "in_distribution"
    return "test", "in_distribution"


def action_text(action: str) -> str:
    call = {"tool": "skip_inspection", "arguments": {}} if action == "skip" else {"tool": "inspect", "arguments": {"test": action}}
    return json.dumps(call, separators=(",", ":"))


def format_prompt(scenario: Scenario, variant: int = 0) -> list[dict[str, str]]:
    if scenario.terminal_visible:
        raise ValueError("Pair-generation prompts target partial observability, not the full-observation control")
    object_name, opening = TEMPLATES[variant % len(TEMPLATES)]
    test_descriptions = "\n".join(
        f"- {test.name}: cost {test.cost:g}; symmetric accuracy {test.accuracy:.2f}. "
        f"P(report A | current fault A) = P(report B | current fault B) = {test.accuracy:.2f}."
        for test in scenario.inspections
    )
    available = "\n".join(action_text(action) for action in first_actions(scenario))
    user = (
        f"{opening}\n"
        f"The {object_name} initially has fault A with probability {scenario.p_a:.2f} and otherwise fault B. "
        "Its actual fault is not directly visible.\n"
        "Stage 1: either skip inspection at zero cost, or buy exactly one inspection below. "
        "An inspection reports A or B about the initial fault; it does not repair it.\n"
        f"{test_descriptions}\n"
        f"After stage 1, the fault flips A<->B with probability {scenario.flip_probability:.2f}, "
        "independently of the report and your action. This transition is not observed.\n"
        "Stage 2: choose repair_A, repair_B, or defer using only the available report and history. "
        "There will be no additional fault revelation. A repair matching the stage-2 fault pays +10; "
        "a mismatching repair pays -30; defer pays 0. Subtract any inspection cost from this payoff.\n"
        "Maximize expected total payoff. Choose only the next stage-1 tool call now. "
        "You will choose the terminal action later, after receiving any inspection report.\n"
        f"Available next calls:\n{available}"
    )
    return [
        {"role": "system", "content": "Select a tool action using only the supplied evidence and public task rules. Return exactly one JSON tool call with no extra text."},
        {"role": "user", "content": user},
    ]


def generate(output_dir: Path, variants: int = 2, seed: int = 17, summary_path: Path | None = None) -> dict:
    if not 1 <= variants <= len(TEMPLATES):
        raise ValueError(f"variants must be between 1 and {len(TEMPLATES)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = {}
    for mode in ("information", "clairvoyant"):
        directory = output_dir / mode
        directory.mkdir(exist_ok=True)
        for split in ("train", "validation", "test"):
            streams[mode, split] = (directory / f"{split}.jsonl").open("w", encoding="utf-8")
    audit_stream = (output_dir / "pairs_audit.jsonl").open("w", encoding="utf-8")
    scenario_stream = (output_dir / "scenario_manifest.jsonl").open("w", encoding="utf-8")
    counts = Counter()
    scenario_counts = Counter()
    try:
        for scenario in scenarios():
            split, regime = split_scenario(scenario, seed)
            scenario_counts[split] += 1
            scenario_counts[f"regime:{regime}"] += 1
            scenario_stream.write(json.dumps({"scenario_id": scenario.scenario_id, "split": split, "regime": regime, "config": scenario.to_dict()}, sort_keys=True) + "\n")
            scores = {mode: first_action_values(scenario, mode) for mode in ("information", "clairvoyant")}
            for variant in range(variants):
                prompt = format_prompt(scenario, variant)
                for left, right in itertools.combinations(first_actions(scenario), 2):
                    pair_id = f"{scenario.scenario_id}:v{variant}:{left}:{right}"
                    preferences = {mode: pair_preference(scores[mode][left], scores[mode][right]) for mode in scores}
                    matched = all(preferences.values())
                    audit = {
                        "pair_id": pair_id, "scenario_id": scenario.scenario_id, "split": split, "regime": regime,
                        "variant": variant, "left": left, "right": right, "kept_in_both_modes": bool(matched),
                        "preferences": preferences,
                        "scores": {mode: {left: scores[mode][left], right: scores[mode][right]} for mode in scores},
                    }
                    audit_stream.write(json.dumps(audit, sort_keys=True) + "\n")
                    counts["candidate_pairs"] += 1
                    if not matched:
                        counts["dropped_from_both_due_to_tie"] += 1
                        continue
                    counts["matched_pairs"] += 1
                    counts[f"split:{split}"] += 1
                    counts[f"regime:{regime}"] += 1
                    if preferences["information"] != preferences["clairvoyant"]:
                        counts["opposite_labels"] += 1
                    for mode in scores:
                        chosen, rejected = (left, right) if preferences[mode] == 1 else (right, left)
                        record = {
                            "prompt": prompt,
                            "chosen": [{"role": "assistant", "content": action_text(chosen)}],
                            "rejected": [{"role": "assistant", "content": action_text(rejected)}],
                            "pair_id": pair_id,
                            "scenario_id": scenario.scenario_id,
                            "regime": regime,
                        }
                        streams[mode, split].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for stream in streams.values():
            stream.close()
        audit_stream.close()
        scenario_stream.close()
    summary = {
        "dataset_type": "Synthetic exact-expectation toy labels; no LLM-generated labels or GPU training",
        "seed": seed,
        "prompt_variants_per_scenario": variants,
        "num_scenarios": sum(scenario_counts[s] for s in ("train", "validation", "test")),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "pair_counts": dict(sorted(counts.items())),
        "matching": "For every retained pair, both teachers see the identical prompt and the same unordered actions. A tie under either teacher drops the pair from both datasets.",
        "splits": "All pairs and prompt variants of a complete scenario configuration share one split. Drift >=0.2 and the (0.75,0.95) reliability combination are held out from training and validation.",
        "format": "Conversational DPO: prompt/chosen/rejected. pair_id/scenario_id/regime are audit metadata, not model input.",
        "scope": "These are first-decision labels only. A learned model must later be tested in a closed-loop environment; preference accuracy alone is insufficient.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--summary-path", type=Path, default=Path(__file__).parent / "results" / "dataset_summary.json")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.variants, args.seed, args.summary_path), indent=2))


if __name__ == "__main__":
    main()
