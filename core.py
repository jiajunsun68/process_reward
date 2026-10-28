"""A finite, two-stage partial-observation decision problem.

This module uses only the Python standard library.  It computes expectations
by enumerating the two possible inspection results, not by sampling outcomes.
The ``clairvoyant`` teacher is deliberately information-infeasible: its
continuation may condition on the otherwise unobserved terminal fault.  It is
a diagnostic baseline, not a deployable policy or a faithful implementation
of every algorithm called QMDP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

Teacher = Literal["information", "clairvoyant"]
CORRECT_REWARD = 10.0
WRONG_REWARD = -30.0
DEFER_REWARD = 0.0
TIE_TOLERANCE = 1e-10


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")


@dataclass(frozen=True)
class Inspection:
    name: str
    accuracy: float
    cost: float

    def __post_init__(self) -> None:
        if not self.name or self.name == "skip":
            raise ValueError("inspection name must be nonempty and not 'skip'")
        _probability(self.accuracy, "accuracy")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("inspection cost must be finite and nonnegative")


@dataclass(frozen=True)
class Scenario:
    p_a: float = 0.5
    flip_probability: float = 0.0
    inspections: tuple[Inspection, ...] = (Inspection("precise", 1.0, 1.0),)
    terminal_visible: bool = False

    def __post_init__(self) -> None:
        _probability(self.p_a, "p_a")
        _probability(self.flip_probability, "flip_probability")
        names = [test.name for test in self.inspections]
        if len(names) != len(set(names)):
            raise ValueError("inspection names must be unique")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def scenario_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def transitioned_belief(p_a: float, flip_probability: float) -> float:
    """P(terminal fault=A) after an unobserved symmetric fault flip."""
    _probability(p_a, "p_a")
    _probability(flip_probability, "flip_probability")
    return (1.0 - flip_probability) * p_a + flip_probability * (1.0 - p_a)


def observation_branches(p_a: float, accuracy: float) -> tuple[tuple[str, float, float], ...]:
    """Return (report, report probability, posterior before drift).

    Accuracy is symmetric: P(report=A|fault=A)=P(report=B|fault=B).
    Zero-probability reports are excluded; their posterior is undefined.
    """
    _probability(p_a, "p_a")
    _probability(accuracy, "accuracy")
    branches = []
    for report, likelihood_a, likelihood_b in (
        ("A", accuracy, 1.0 - accuracy),
        ("B", 1.0 - accuracy, accuracy),
    ):
        probability = p_a * likelihood_a + (1.0 - p_a) * likelihood_b
        if probability > 0.0:
            posterior = p_a * likelihood_a / probability
            branches.append((report, probability, posterior))
    return tuple(branches)


def terminal_values(p_a: float) -> dict[str, float]:
    """Expected utility of a terminal action under a visible-history belief."""
    _probability(p_a, "p_a")
    # Defer first gives a deterministic, conservative tie break.
    return {
        "defer": DEFER_REWARD,
        "repair_A": p_a * CORRECT_REWARD + (1.0 - p_a) * WRONG_REWARD,
        "repair_B": (1.0 - p_a) * CORRECT_REWARD + p_a * WRONG_REWARD,
    }


def choose_best(values: dict[str, float]) -> str:
    """Return the first maximizer in insertion order, with a numerical tolerance."""
    maximum = max(values.values())
    return next(action for action, value in values.items() if maximum - value <= TIE_TOLERANCE)


def best_terminal_action(p_a: float) -> tuple[str, float]:
    values = terminal_values(p_a)
    action = choose_best(values)
    return action, values[action]


def first_actions(scenario: Scenario) -> tuple[str, ...]:
    return ("skip", *(test.name for test in scenario.inspections))


def action_cost(scenario: Scenario, action: str) -> float:
    if action == "skip":
        return 0.0
    for test in scenario.inspections:
        if test.name == action:
            return test.cost
    raise ValueError(f"unknown first action: {action}")


def information_value(scenario: Scenario, action: str) -> float:
    """Best executable continuation using only the available observations.

    A continuation can depend on an inspection report. Without a report it
    must use one shared terminal action across both possible fault states.
    Terminal visibility is an explicit full-information control condition.
    """
    cost = action_cost(scenario, action)
    if scenario.terminal_visible:
        return CORRECT_REWARD - cost
    if action == "skip":
        belief = transitioned_belief(scenario.p_a, scenario.flip_probability)
        return best_terminal_action(belief)[1]
    test = next(test for test in scenario.inspections if test.name == action)
    continuation = sum(
        probability
        * best_terminal_action(transitioned_belief(posterior, scenario.flip_probability))[1]
        for _, probability, posterior in observation_branches(scenario.p_a, test.accuracy)
    )
    return continuation - cost


def clairvoyant_value(scenario: Scenario, action: str) -> float:
    """Optimistic continuation that is given the unobserved terminal state.

    Even an uninspected fault is magically revealed before the terminal
    action, so a correct repair is always selected. Inspection only costs
    money. This moves max inside the latent-state expectation and destroys
    the decision value of sensing.
    """
    return CORRECT_REWARD - action_cost(scenario, action)


def first_action_values(scenario: Scenario, teacher: Teacher = "information") -> dict[str, float]:
    if teacher not in ("information", "clairvoyant"):
        raise ValueError(f"unknown teacher: {teacher}")
    scorer = information_value if teacher == "information" else clairvoyant_value
    return {action: scorer(scenario, action) for action in first_actions(scenario)}


def pair_preference(left_value: float, right_value: float) -> int:
    """1 prefers left, -1 prefers right, 0 is a genuine/numerical tie."""
    difference = left_value - right_value
    if abs(difference) <= TIE_TOLERANCE:
        return 0
    return 1 if difference > 0 else -1


def evaluate_scenario(scenario: Scenario) -> dict:
    """Evaluate both teachers' first actions under the SAME deployable information.

    The clairvoyant teacher's prediction is never used as its deployed return.
    At deployment the terminal controller always uses the available belief.
    """
    feasible = first_action_values(scenario, "information")
    optimistic = first_action_values(scenario, "clairvoyant")
    feasible_action = choose_best(feasible)
    optimistic_action = choose_best(optimistic)
    best_value = feasible[feasible_action]
    deployed_optimistic = feasible[optimistic_action]
    return {
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.to_dict(),
        "information_values": feasible,
        "clairvoyant_values": optimistic,
        "information_action": feasible_action,
        "clairvoyant_action": optimistic_action,
        "information_deployed_return": best_value,
        "clairvoyant_deployed_return": deployed_optimistic,
        "clairvoyant_decision_regret": max(0.0, best_value - deployed_optimistic),
        "selected_actions_disagree": feasible_action != optimistic_action,
        "deployment_information": "Both terminal controllers use only the inspection report/history; no latent fault is revealed unless terminal_visible=True.",
    }


def canonical_counterexample() -> Scenario:
    return Scenario(p_a=0.5, flip_probability=0.0, inspections=(Inspection("precise", 1.0, 1.0),))
