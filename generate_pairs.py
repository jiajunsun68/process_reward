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
import random

from core import Inspection, Scenario, TIE_TOLERANCE, first_action_values, first_actions, pair_preference

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


CONDITIONAL_CONTROLLER_CLAUSE = (
    "After your stage-1 choice, a FIXED, history-only-observing Bayes-optimal "
    "controller plays stage 2 for you: it sees only the public task rules and any "
    "inspection report you receive, never the hidden fault or any other hidden "
    "information, and it always selects the terminal action maximizing expected "
    "payoff given exactly that evidence. You are not asked to simulate stage 2."
)


def format_conditional_prompt(scenario: Scenario, variant: int = 0) -> list[dict[str, str]]:
    """Like format_prompt, but states explicitly that stage 2 is a fixed Bayes-optimal
    history-only controller, not something the model must itself reason through."""
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
        f"{CONDITIONAL_CONTROLLER_CLAUSE}\n"
        "A repair matching the stage-2 fault pays +10; a mismatching repair pays -30; defer pays 0. "
        "Subtract any inspection cost from this payoff.\n"
        "Maximize expected total payoff. Choose only the next stage-1 tool call now.\n"
        f"Available next calls:\n{available}"
    )
    return [
        {"role": "system", "content": "Select a tool action using only the supplied evidence and public task rules. Return exactly one JSON tool call with no extra text."},
        {"role": "user", "content": user},
    ]


def build_conditional_family(p_a: float, flip: float, q_quick: float, q_precise: float) -> dict | None:
    """One family fixes prior/drift/accuracies and realizes all 6 strict action
    orderings exactly once by varying only the two inspection costs."""
    zero_cost = Scenario(p_a, flip, (Inspection("quick", q_quick, 0.0), Inspection("precise", q_precise, 0.0)))
    zero_values = first_action_values(zero_cost, "information")
    v0 = zero_values["skip"]
    g_quick = zero_values["quick"] - v0
    g_precise = zero_values["precise"] - v0
    if g_quick < 1.0 or g_precise < 1.0:
        return None
    m = min(0.5, min(g_quick, g_precise) / 4.0)
    scenarios_by_ranking: dict[tuple[str, ...], Scenario] = {}
    for ordering in itertools.permutations(("skip", "quick", "precise")):
        rank = {action: position for position, action in enumerate(ordering)}
        cost_quick = g_quick - (rank["skip"] - rank["quick"]) * m
        cost_precise = g_precise - (rank["skip"] - rank["precise"]) * m
        if cost_quick <= 0.0 or cost_precise <= 0.0:
            return None
        scenario = Scenario(p_a, flip, (Inspection("quick", q_quick, cost_quick), Inspection("precise", q_precise, cost_precise)))
        values = first_action_values(scenario, "information")
        ranked_actions = tuple(sorted(values, key=lambda action: -values[action]))
        if ranked_actions != ordering:
            return None
        sorted_values = sorted(values.values(), reverse=True)
        if min(sorted_values[0] - sorted_values[1], sorted_values[1] - sorted_values[2]) < 0.25:
            return None
        scenarios_by_ranking[ordering] = scenario
    return {
        "p_a": p_a, "flip_probability": flip, "q_quick": q_quick, "q_precise": q_precise,
        "v0": v0, "g_quick": g_quick, "g_precise": g_precise, "m": m,
        "scenarios_by_ranking": scenarios_by_ranking,
    }


def search_conditional_families(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    found: list[dict] = []
    seen: set[tuple[float, float, float, float]] = set()
    attempts = 0
    max_attempts = 500_000
    while len(found) < count and attempts < max_attempts:
        attempts += 1
        p_a = round(rng.uniform(0.15, 0.85), 3)
        flip = round(rng.uniform(0.0, 0.18), 3)
        q_quick = round(rng.uniform(0.55, 0.85), 3)
        q_precise = round(rng.uniform(q_quick + 0.03, 1.0), 3)
        key = (p_a, flip, q_quick, q_precise)
        if key in seen:
            continue
        seen.add(key)
        family = build_conditional_family(p_a, flip, q_quick, q_precise)
        if family is not None:
            found.append(family)
    if len(found) < count:
        raise RuntimeError(f"only found {len(found)}/{count} valid conditional families after {attempts} attempts")
    return found


def ranking_preference(ranking: tuple[str, ...], left: str, right: str) -> int:
    return 1 if ranking.index(left) < ranking.index(right) else -1


def constant_action_oracle_regret(scenarios_list: list[Scenario], action: str) -> float:
    regrets = []
    for scenario in scenarios_list:
        values = first_action_values(scenario, "information")
        regrets.append(max(0.0, max(values.values()) - values[action]))
    return sum(regrets) / len(regrets) if regrets else 0.0


def knn_cost_baseline(train_rows: list[dict], query_rows: list[dict], k: int = 31) -> list[str]:
    """Fit on train only; raw Euclidean distance over (cost_quick, cost_precise);
    tie-break on distance by frozen training-row order; tie-break on majority vote
    by skip > quick > precise priority."""
    priority = {"skip": 0, "quick": 1, "precise": 2}
    predictions = []
    for query in query_rows:
        qc = (query["cost_quick"], query["cost_precise"])
        distances = [
            ((row["cost_quick"] - qc[0]) ** 2 + (row["cost_precise"] - qc[1]) ** 2, index, row["label"])
            for index, row in enumerate(train_rows)
        ]
        distances.sort(key=lambda item: (item[0], item[1]))
        neighbors = distances[:k]
        votes = Counter(label for _, _, label in neighbors)
        best_count = max(votes.values())
        tied = [action for action, count in votes.items() if count == best_count]
        tied.sort(key=lambda action: priority[action])
        predictions.append(tied[0])
    return predictions


def evaluate_cpu_baselines(families: list[dict], split_name: str) -> dict:
    scenarios_list = [scenario for family in families for scenario in family["scenarios_by_ranking"].values()]
    rows = [
        {"cost_quick": family["scenarios_by_ranking"][ordering].inspections[0].cost,
         "cost_precise": family["scenarios_by_ranking"][ordering].inspections[1].cost,
         "label": ordering[0]}
        for family in families for ordering in family["scenarios_by_ranking"]
    ]
    result = {
        "n_scenarios": len(scenarios_list),
        "constant_skip_regret": constant_action_oracle_regret(scenarios_list, "skip"),
        "constant_quick_regret": constant_action_oracle_regret(scenarios_list, "quick"),
        "constant_precise_regret": constant_action_oracle_regret(scenarios_list, "precise"),
        "exact_oracle_regret": 0.0,
    }
    return result, rows


def evaluate_knn_baseline(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    predictions = knn_cost_baseline(train_rows, eval_rows, k=31)
    correct = sum(1 for prediction, row in zip(predictions, eval_rows) if prediction == row["label"])
    return {"accuracy": correct / len(eval_rows) if eval_rows else None, "n": len(eval_rows)}


def build_fixed_cost_probe(seed: int = 1701) -> list[dict]:
    """48 pairs (96 scenarios) from the legacy grid, held out of all training.
    One Random(seed) processes drift/prior/reliability sequentially, greedily
    selecting 16 not-yet-used-scenario pairs per group."""
    pool = [scenario for scenario in scenarios() if scenario.flip_probability <= 0.10]

    def margin_optimal(scenario: Scenario) -> tuple[str, float]:
        values = first_action_values(scenario, "information")
        best = max(values, key=values.get)
        sorted_values = sorted(values.values(), reverse=True)
        return best, sorted_values[0] - sorted_values[1]

    def field_key(scenario: Scenario, exclude: str) -> tuple:
        reliability = tuple(test.accuracy for test in scenario.inspections)
        costs = tuple(test.cost for test in scenario.inspections)
        fields = {"p_a": scenario.p_a, "flip": scenario.flip_probability, "reliability": reliability, "costs": costs}
        del fields[exclude]
        return tuple(sorted(fields.items()))

    rng = random.Random(seed)
    used: set[str] = set()
    probe: list[dict] = []
    for group_name, varied_field in (("drift", "flip"), ("prior", "p_a"), ("reliability", "reliability")):
        buckets: dict[tuple, list[Scenario]] = {}
        for scenario in pool:
            if scenario.scenario_id in used:
                continue
            _, margin = margin_optimal(scenario)
            if margin < 0.25:
                continue
            buckets.setdefault(field_key(scenario, varied_field), []).append(scenario)
        candidates = []
        for members in buckets.values():
            for left, right in itertools.combinations(members, 2):
                left_action, _ = margin_optimal(left)
                right_action, _ = margin_optimal(right)
                if left_action != right_action:
                    candidates.append((left, right))
        rng.shuffle(candidates)
        selected = []
        for left, right in candidates:
            if left.scenario_id in used or right.scenario_id in used:
                continue
            selected.append((left, right))
            used.add(left.scenario_id)
            used.add(right.scenario_id)
            if len(selected) == 16:
                break
        if len(selected) < 16:
            raise RuntimeError(f"fixed-cost probe group {group_name} only found {len(selected)}/16 pairs")
        for left, right in selected:
            left_action, left_margin = margin_optimal(left)
            right_action, right_margin = margin_optimal(right)
            probe.append({
                "pair_id": f"probe:{group_name}:{left.scenario_id}:{right.scenario_id}",
                "group": group_name, "varied_field": varied_field,
                "left": {"scenario_id": left.scenario_id, "config": left.to_dict(), "optimal_action": left_action, "margin": left_margin},
                "right": {"scenario_id": right.scenario_id, "config": right.to_dict(), "optimal_action": right_action, "margin": right_margin},
            })
    return probe


def self_test_conditional_construction() -> None:
    """Required CPU self-test: split/margin correctness, oracle-and-tie-scoring
    correctness, probe control-variable correctness. Run before any GPU work."""
    families = search_conditional_families(8, seed=17)
    assert len(families) == 8
    for family in families:
        assert len(family["scenarios_by_ranking"]) == 6
        for ordering, scenario in family["scenarios_by_ranking"].items():
            values = first_action_values(scenario, "information")
            ranked = tuple(sorted(values, key=lambda action: -values[action]))
            assert ranked == ordering
            sorted_values = sorted(values.values(), reverse=True)
            assert sorted_values[0] - sorted_values[1] >= 0.25 - 1e-9
            assert sorted_values[1] - sorted_values[2] >= 0.25 - 1e-9
            for inspection in scenario.inspections:
                assert inspection.cost > 0.0
    probe = build_fixed_cost_probe(seed=1701)
    assert len(probe) == 48
    seen_ids: set[str] = set()
    group_counts = Counter(pair["group"] for pair in probe)
    assert group_counts == {"drift": 16, "prior": 16, "reliability": 16}
    for pair in probe:
        left, right = pair["left"], pair["right"]
        assert left["scenario_id"] not in seen_ids and right["scenario_id"] not in seen_ids
        seen_ids.add(left["scenario_id"]); seen_ids.add(right["scenario_id"])
        assert left["optimal_action"] != right["optimal_action"]
        assert left["margin"] >= 0.25 - 1e-9 and right["margin"] >= 0.25 - 1e-9
        assert left["config"]["flip_probability"] <= 0.10 + 1e-9
        assert right["config"]["flip_probability"] <= 0.10 + 1e-9
        left_costs = tuple(i["cost"] for i in left["config"]["inspections"])
        right_costs = tuple(i["cost"] for i in right["config"]["inspections"])
        assert left_costs == right_costs
        varied = pair["varied_field"]
        if varied == "flip":
            assert left["config"]["p_a"] == right["config"]["p_a"]
            assert left["config"]["flip_probability"] != right["config"]["flip_probability"]
        elif varied == "p_a":
            assert left["config"]["flip_probability"] == right["config"]["flip_probability"]
            assert left["config"]["p_a"] != right["config"]["p_a"]
        else:
            left_rel = tuple(i["accuracy"] for i in left["config"]["inspections"])
            right_rel = tuple(i["accuracy"] for i in right["config"]["inspections"])
            assert left["config"]["p_a"] == right["config"]["p_a"]
            assert left["config"]["flip_probability"] == right["config"]["flip_probability"]
            assert left_rel != right_rel
    print(json.dumps({"self_test": "pass", "families_checked": len(families), "probe_pairs_checked": len(probe)}))


def generate_conditional(
    output_dir: Path,
    family_seed: int = 17,
    shuffle_seed: int = 1_000_020,
    probe_seed: int = 1701,
    train_families: int = 120,
    validation_families: int = 24,
    test_families: int = 48,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_families = train_families + validation_families + test_families
    families = search_conditional_families(total_families, seed=family_seed)
    split_assignment = (
        [("train", family) for family in families[:train_families]]
        + [("validation", family) for family in families[train_families : train_families + validation_families]]
        + [("test", family) for family in families[train_families + validation_families :]]
    )

    scenario_rows: list[dict] = []
    scenario_true_ranking: dict[str, tuple[str, ...]] = {}
    scenario_split: dict[str, str] = {}
    scenario_family: dict[str, str] = {}
    for family_index, (split, family) in enumerate(split_assignment):
        family_id = f"F{family_index:04d}"
        for ordering, scenario in family["scenarios_by_ranking"].items():
            scenario_true_ranking[scenario.scenario_id] = ordering
            scenario_split[scenario.scenario_id] = split
            scenario_family[scenario.scenario_id] = family_id
            scenario_rows.append({
                "scenario_id": scenario.scenario_id, "family_id": family_id, "split": split,
                "true_ranking": list(ordering), "config": scenario.to_dict(),
                "v0": family["v0"], "g_quick": family["g_quick"], "g_precise": family["g_precise"], "m": family["m"],
            })

    train_ids_sorted = sorted(sid for sid, split in scenario_split.items() if split == "train")
    true_orderings_in_order = [scenario_true_ranking[sid] for sid in train_ids_sorted]
    shuffled_orderings = list(true_orderings_in_order)
    random.Random(shuffle_seed).shuffle(shuffled_orderings)
    scenario_shuffle_ranking: dict[str, tuple[str, ...]] = dict(scenario_true_ranking)
    for sid, shuffled in zip(train_ids_sorted, shuffled_orderings):
        scenario_shuffle_ranking[sid] = shuffled

    scenario_by_id = {
        scenario.scenario_id: scenario
        for _, family in split_assignment
        for scenario in family["scenarios_by_ranking"].values()
    }

    streams = {}
    for mode in ("true", "shuffle"):
        directory = output_dir / mode
        directory.mkdir(exist_ok=True)
        for split in ("train", "validation", "test"):
            streams[mode, split] = (directory / f"{split}.jsonl").open("w", encoding="utf-8")
    audit_stream = (output_dir / "pairs_audit.jsonl").open("w", encoding="utf-8")
    scenario_stream = (output_dir / "scenario_manifest.jsonl").open("w", encoding="utf-8")
    counts = Counter()
    rankings_for_mode = {"true": scenario_true_ranking, "shuffle": scenario_shuffle_ranking}
    try:
        for row in scenario_rows:
            scenario_stream.write(json.dumps(row, sort_keys=True) + "\n")
        for scenario_id, scenario in scenario_by_id.items():
            split = scenario_split[scenario_id]
            family_id = scenario_family[scenario_id]
            prompt = format_conditional_prompt(scenario, variant=0)
            for left, right in itertools.combinations(first_actions(scenario), 2):
                pair_id = f"{scenario_id}:v0:{left}:{right}"
                true_pref = ranking_preference(scenario_true_ranking[scenario_id], left, right)
                shuffle_pref = ranking_preference(scenario_shuffle_ranking[scenario_id], left, right)
                audit_stream.write(json.dumps({
                    "pair_id": pair_id, "scenario_id": scenario_id, "family_id": family_id, "split": split,
                    "left": left, "right": right, "true_preference": true_pref, "shuffle_preference": shuffle_pref,
                    "true_ranking": list(scenario_true_ranking[scenario_id]), "shuffle_ranking": list(scenario_shuffle_ranking[scenario_id]),
                }, sort_keys=True) + "\n")
                counts["pairs_total"] += 1
                counts[f"split:{split}"] += 1
                for mode in ("true", "shuffle"):
                    preference = rankings_for_mode[mode]
                    pref = ranking_preference(preference[scenario_id], left, right)
                    chosen, rejected = (left, right) if pref == 1 else (right, left)
                    record = {
                        "prompt": prompt,
                        "chosen": [{"role": "assistant", "content": action_text(chosen)}],
                        "rejected": [{"role": "assistant", "content": action_text(rejected)}],
                        "pair_id": pair_id, "scenario_id": scenario_id, "family_id": family_id, "split": split,
                    }
                    streams[mode, split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[f"{mode}:{split}:pairs"] += 1
    finally:
        for stream in streams.values():
            stream.close()
        audit_stream.close()
        scenario_stream.close()

    probe = build_fixed_cost_probe(seed=probe_seed)
    probe_path = output_dir / "fixed_cost_probes.jsonl"
    with probe_path.open("w", encoding="utf-8") as stream:
        for pair in probe:
            stream.write(json.dumps(pair, sort_keys=True) + "\n")

    train_families_list = [family for split, family in split_assignment if split == "train"]
    validation_families_list = [family for split, family in split_assignment if split == "validation"]
    test_families_list = [family for split, family in split_assignment if split == "test"]
    baseline_train, train_rows = evaluate_cpu_baselines(train_families_list, "train")
    baseline_validation, validation_rows = evaluate_cpu_baselines(validation_families_list, "validation")
    baseline_test, test_rows = evaluate_cpu_baselines(test_families_list, "test")
    baseline_validation["knn_k31"] = evaluate_knn_baseline(train_rows, validation_rows)
    baseline_test["knn_k31"] = evaluate_knn_baseline(train_rows, test_rows)

    summary = {
        "dataset_type": "Synthetic exact-expectation conditional-cost toy labels; no LLM-generated labels or GPU training",
        "family_seed": family_seed, "shuffle_seed": shuffle_seed, "probe_seed": probe_seed,
        "prompt_variants_per_scenario": 1,
        "family_counts": {"train": train_families, "validation": validation_families, "test": test_families},
        "scenario_counts": {"train": train_families * 6, "validation": validation_families * 6, "test": test_families * 6},
        "pair_counts": dict(sorted(counts.items())),
        "shuffle_control": "TRUE labels are each scenario's own constructed ranking. SHUFFLE permutes the complete ranking only among the 720 TRAIN scenarios (seed shown above); validation/test always use the correct ranking in both modes.",
        "cpu_baselines": {"train": baseline_train, "validation": baseline_validation, "test": baseline_test},
        "fixed_cost_probe": {"n_pairs": len(probe), "n_scenarios": 2 * len(probe), "groups": dict(Counter(pair["group"] for pair in probe))},
        "format": "Conversational DPO: prompt/chosen/rejected. pair_id/scenario_id/family_id/split are audit metadata, not model input.",
        "scope": "First-decision labels only, same scope limitation as the legacy information/clairvoyant datasets.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional", action="store_true", help="Generate the G3 conditional-cost dataset into --output-dir instead of the legacy information/clairvoyant pairs")
    parser.add_argument("--self-test", action="store_true", help="Run the required CPU self-test for the conditional dataset and exit")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--summary-path", type=Path, default=Path(__file__).parent / "results" / "dataset_summary.json")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.self_test:
        self_test_conditional_construction()
        return
    if args.conditional:
        print(json.dumps(generate_conditional(args.output_dir), indent=2))
        return
    print(json.dumps(generate(args.output_dir, args.variants, args.seed, args.summary_path), indent=2))


if __name__ == "__main__":
    main()
