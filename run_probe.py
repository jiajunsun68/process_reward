"""Run a CPU-only analytic toy diagnostic. This is NOT an LLM experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from core import Inspection, Scenario, canonical_counterexample, evaluate_scenario, pair_preference


def make_grid() -> list[Scenario]:
    return [
        Scenario(p, delta, (Inspection("inspect", accuracy, cost),))
        for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        for cost in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        for accuracy in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        for delta in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
    ]


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "num_scenarios": count,
        "selected_action_disagreement_count": sum(row["selected_actions_disagree"] for row in rows),
        "selected_action_disagreement_rate": mean(row["selected_actions_disagree"] for row in rows),
        "strict_preference_reversal_count": sum(row["strict_preference_reversal"] for row in rows),
        "strict_preference_reversal_rate": mean(row["strict_preference_reversal"] for row in rows),
        "positive_deployment_regret_count": sum(row["clairvoyant_decision_regret"] > 1e-10 for row in rows),
        "mean_deployment_regret": mean(row["clairvoyant_decision_regret"] for row in rows),
        "max_deployment_regret": max(row["clairvoyant_decision_regret"] for row in rows),
        "mean_information_return": mean(row["information_deployed_return"] for row in rows),
        "mean_clairvoyant_deployed_return": mean(row["clairvoyant_deployed_return"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scenarios = make_grid()
    for scenario in scenarios:
        result = evaluate_scenario(scenario)
        info = result["information_values"]
        clair = result["clairvoyant_values"]
        test = scenario.inspections[0]
        rows.append({
            "scenario_id": scenario.scenario_id,
            "p_a": scenario.p_a,
            "cost": test.cost,
            "accuracy": test.accuracy,
            "flip_probability": scenario.flip_probability,
            "information_skip": info["skip"],
            "information_inspect": info["inspect"],
            "clairvoyant_skip": clair["skip"],
            "clairvoyant_inspect": clair["inspect"],
            "information_action": result["information_action"],
            "clairvoyant_action": result["clairvoyant_action"],
            "information_deployed_return": result["information_deployed_return"],
            "clairvoyant_deployed_return": result["clairvoyant_deployed_return"],
            "clairvoyant_decision_regret": result["clairvoyant_decision_regret"],
            "selected_actions_disagree": result["selected_actions_disagree"],
            "strict_preference_reversal": pair_preference(info["inspect"], info["skip"])
            * pair_preference(clair["inspect"], clair["skip"]) == -1,
        })
    with (args.output_dir / "grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_drift = []
    for delta in sorted({row["flip_probability"] for row in rows}):
        by_drift.append({"flip_probability": delta, **summarize([row for row in rows if row["flip_probability"] == delta])})
    with (args.output_dir / "by_drift.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_drift[0]))
        writer.writeheader()
        writer.writerows(by_drift)
    visible_results = [evaluate_scenario(Scenario(s.p_a, s.flip_probability, s.inspections, terminal_visible=True)) for s in scenarios]
    summary = {
        "experiment_type": "CPU analytic finite-state toy enumeration; no language model was trained or evaluated",
        "arithmetic": "Enumerated expectations evaluated with Python floating point; no Monte Carlo sampling",
        "canonical_counterexample": evaluate_scenario(canonical_counterexample()),
        "grid": summarize(rows),
        "grid_axes": {
            "p_a": sorted({s.p_a for s in scenarios}),
            "cost": sorted({s.inspections[0].cost for s in scenarios}),
            "accuracy": sorted({s.inspections[0].accuracy for s in scenarios}),
            "flip_probability": sorted({s.flip_probability for s in scenarios}),
        },
        "by_drift": by_drift,
        "full_observability_control": {
            "num_scenarios": len(visible_results),
            "max_value_gap": max(abs(v["information_values"][a] - v["clairvoyant_values"][a]) for v in visible_results for a in v["information_values"]),
            "max_deployment_regret": max(v["clairvoyant_decision_regret"] for v in visible_results),
        },
        "interpretation": [
            "Grid averages describe this enumerated design, not a population or an empirical LLM failure rate.",
            "Selected-action disagreement uses the documented skip-first tie break. Strict preference reversals exclude ties.",
            "The clairvoyant teacher predicts future full observability. Its deployed terminal controller receives no extra information.",
            "The toy exposes information-infeasible continuation labels; it does not prove that every privileged critic or process reward is biased.",
        ],
    }
    (args.output_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counterexample": summary["canonical_counterexample"], "grid": summary["grid"], "full_observability_control": summary["full_observability_control"]}, indent=2))


if __name__ == "__main__":
    main()
