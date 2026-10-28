"""Meaningful oracle, information-boundary, and paired-dataset checks."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from core import (
    Inspection, Scenario, best_terminal_action, canonical_counterexample,
    evaluate_scenario, first_action_values, information_value,
    observation_branches, transitioned_belief,
)
from generate_pairs import format_prompt, generate, scenarios, split_scenario
from audit_cases import MECHANISMS, STRATA, generate_cases, validate_cases
from audit_teacher import build_messages, call_plan, parse_scores


class DecisionProblemTests(unittest.TestCase):
    def test_analytic_counterexample(self) -> None:
        result = evaluate_scenario(canonical_counterexample())
        self.assertEqual(result["information_values"], {"skip": 0.0, "precise": 9.0})
        self.assertEqual(result["clairvoyant_values"], {"skip": 10.0, "precise": 9.0})
        self.assertEqual(result["information_action"], "precise")
        self.assertEqual(result["clairvoyant_action"], "skip")
        self.assertEqual(result["clairvoyant_deployed_return"], 0.0)
        self.assertEqual(result["clairvoyant_decision_regret"], 9.0)
        self.assertEqual(best_terminal_action(0.5), ("defer", 0.0))

    def test_full_observability_removes_gap(self) -> None:
        for p in (0.0, 0.1, 0.5, 0.9, 1.0):
            for delta in (0.0, 0.2, 0.5):
                scenario = Scenario(p, delta, (Inspection("test", 0.8, 1.0),), terminal_visible=True)
                self.assertEqual(first_action_values(scenario, "information"), first_action_values(scenario, "clairvoyant"))
                self.assertEqual(evaluate_scenario(scenario)["clairvoyant_decision_regret"], 0.0)

    def test_random_bayes_probabilities_and_total_expectation(self) -> None:
        rng = random.Random(42)
        for _ in range(1000):
            p, q, delta = rng.random(), rng.random(), rng.random()
            branches = observation_branches(p, q)
            self.assertAlmostEqual(sum(prob for _, prob, _ in branches), 1.0)
            self.assertAlmostEqual(sum(prob * posterior for _, prob, posterior in branches), p)
            self.assertAlmostEqual(sum(prob * transitioned_belief(posterior, delta) for _, prob, posterior in branches), transitioned_belief(p, delta))
            for _, probability, posterior in branches:
                self.assertGreater(probability, 0.0)
                self.assertTrue(0.0 <= posterior <= 1.0)

    def test_impossible_observations_are_not_normalized(self) -> None:
        self.assertEqual(observation_branches(1.0, 1.0), (("A", 1.0, 1.0),))
        self.assertEqual(observation_branches(0.0, 1.0), (("B", 1.0, 0.0),))

    def test_uninformative_inspection_has_no_benefit(self) -> None:
        for p in (0.0, 0.2, 0.5, 0.8, 1.0):
            for delta in (0.0, 0.1, 0.5):
                scenario = Scenario(p, delta, (Inspection("test", 0.5, 1.0),))
                self.assertAlmostEqual(information_value(scenario, "test"), information_value(scenario, "skip") - 1.0)

    def test_drift_can_destroy_information_value(self) -> None:
        scenario = Scenario(0.5, 0.5, (Inspection("test", 1.0, 1.0),))
        self.assertEqual(information_value(scenario, "skip"), 0.0)
        self.assertEqual(information_value(scenario, "test"), -1.0)

    def test_costless_observation_never_worse(self) -> None:
        rng = random.Random(11)
        for _ in range(500):
            scenario = Scenario(rng.random(), rng.random(), (Inspection("test", rng.random(), 0.0),))
            self.assertGreaterEqual(information_value(scenario, "test") + 1e-10, information_value(scenario, "skip"))

    def test_invalid_config_is_rejected(self) -> None:
        for bad in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                Scenario(p_a=bad)
        with self.assertRaises(ValueError):
            Inspection("test", 0.9, -1.0)


class DatasetTests(unittest.TestCase):
    def test_prompt_uses_only_public_config(self) -> None:
        scenario = canonical_counterexample()
        prompt = json.dumps(format_prompt(scenario), sort_keys=True)
        for forbidden in ("hidden_state", "future_state", "oracle_value", "information_values", "clairvoyant_values", "decision_regret"):
            self.assertNotIn(forbidden, prompt)
        self.assertNotIn("fault is A.", prompt)
        self.assertNotIn("fault is B.", prompt)
        self.assertEqual(format_prompt(scenario), format_prompt(scenario))

    def test_ood_configs_never_enter_training(self) -> None:
        for scenario in scenarios():
            split, _ = split_scenario(scenario)
            if scenario.flip_probability >= 0.2 or tuple(t.accuracy for t in scenario.inspections) == (0.75, 0.95):
                self.assertEqual(split, "test")

    def test_matched_pairs_and_no_scenario_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = generate(root, variants=1)
            ids_per_split = {}
            retained = 0
            for split in ("train", "validation", "test"):
                modes = {}
                for mode in ("information", "clairvoyant"):
                    modes[mode] = [json.loads(line) for line in (root / mode / f"{split}.jsonl").read_text().splitlines()]
                left, right = modes["information"], modes["clairvoyant"]
                self.assertEqual([item["pair_id"] for item in left], [item["pair_id"] for item in right])
                for a, b in zip(left, right):
                    self.assertEqual(a["prompt"], b["prompt"])
                    self.assertEqual({a["chosen"][0]["content"], a["rejected"][0]["content"]}, {b["chosen"][0]["content"], b["rejected"][0]["content"]})
                    self.assertNotEqual(a["chosen"], a["rejected"])
                retained += len(left)
                ids_per_split[split] = {item["scenario_id"] for item in left}
            self.assertEqual(retained, summary["pair_counts"]["matched_pairs"])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse(ids_per_split[a] & ids_per_split[b])
            audits = [json.loads(line) for line in (root / "pairs_audit.jsonl").read_text().splitlines()]
            for item in audits:
                self.assertEqual(item["kept_in_both_modes"], all(item["preferences"].values()))


class AuditProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = generate_cases()

    def test_exact_root_case_cells_and_true_posteriors(self) -> None:
        summary = validate_cases(self.cases)
        self.assertEqual(summary["independent_root_cases"], 360)
        self.assertEqual(summary["counts"]["split:dev"], 72)
        self.assertEqual(summary["counts"]["split:audit"], 288)
        for mechanism in MECHANISMS:
            for stratum in STRATA:
                self.assertEqual(summary["counts"][f"cell:{mechanism}:{stratum}:dev"], 8)
                self.assertEqual(summary["counts"][f"cell:{mechanism}:{stratum}:audit"], 32)
        for case in self.cases:
            worlds = case["private_context"]["worlds"]
            self.assertAlmostEqual(sum(world["posterior"] for world in worlds), 1.0)
            for action, exact in case["gold"]["q_pi"].items():
                self.assertAlmostEqual(exact, sum(world["posterior"] * world["q_pi"][action] for world in worlds))
                outcome = case["gold"]["candidate_outcomes_under_pi"][action]
                self.assertTrue(0.0 <= outcome["harm_probability"] <= 1.0)
                self.assertTrue(0.0 <= outcome["pause_probability"] <= 1.0)

    def test_call_budgets_and_H_prompt_boundary(self) -> None:
        commit = "0" * 40
        self.assertEqual(len(call_plan(self.cases, "dev24", ("H", "P", "PR", "PR-IC"), commit, 42)), 216)
        self.assertEqual(len(call_plan(self.cases, "dev72", ("H", "P", "PR", "PR-IC"), commit, 42)), 648)
        self.assertEqual(len(call_plan(self.cases, "audit288", ("H", "P", "PR", "PR-IC", "PR-full", "PR-full-IC"), commit, 42)), 2976)
        messages = json.dumps(build_messages(self.cases[0], "H", None), ensure_ascii=False)
        self.assertNotIn("Privileged current hidden state", messages)
        self.assertNotIn("Reference action IDs", messages)

    def test_teacher_parser_is_strict_and_keeps_invalid_outputs_invalid(self) -> None:
        order = ["a", "b", "c"]
        parsed, error = parse_scores('{"scores":{"a":1,"b":0,"c":-1}}', order)
        self.assertEqual(parsed, {"a": 1, "b": 0, "c": -1})
        self.assertIsNone(error)
        self.assertIsNone(parse_scores('{"scores":{"a":1}}', order)[0])
        self.assertIsNone(parse_scores('{"scores":{"a":1.0,"b":0,"c":-1}}', order)[0])


if __name__ == "__main__":
    unittest.main()
