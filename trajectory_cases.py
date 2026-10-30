"""G4 evidence-sensitivity cases: local tool simulator, paired histories, exact gold.

Three mechanisms (version-bound reports, scoped/revocable authorization, timeout
and exactly-once execution) over a finite public-history/information-constrained
decision process.  The fixed continuation pi0 uses only public information; the
exact Q^{pi0} for each candidate is computed by enumerating the at most two hidden
worlds consistent with the public history.  Only public facts enter model inputs;
gold, posterior, world bits and the analytic state table stay in memory/private.

Data are generated exactly once and frozen before any model call.  This file is
CPU-only and imports nothing from the GPU stack.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

DATA_SEED = 42
SPLIT_SEED = 43
N_FAMILIES_PER_CELL = 30
N_DEV_PER_CELL = 6
N_AUDIT_PER_CELL = 24
MECHANISMS = ("versioned_report", "scoped_authorization", "exactly_once")
PAIR_TYPES = ("reversal", "invariant")

# Canonical hidden states and the six analytic rows required by the protocol.
# key: (mechanism, state name)
ANALYTIC_TABLE = {
    ("versioned_report", "good"): {"query": 9.0, "direct": 10.0, "stop": 0.0},
    ("versioned_report", "bad"): {"query": -1.0, "direct": -30.0, "stop": 0.0},
    ("versioned_report", "unknown"): {"query": 4.0, "direct": -10.0, "stop": 0.0},
    ("scoped_authorization", "allow"): {"query": 9.0, "direct": 10.0, "stop": 0.0},
    ("scoped_authorization", "deny"): {"query": -1.0, "direct": -30.0, "stop": 0.0},
    ("scoped_authorization", "unknown"): {"query": 4.0, "direct": -10.0, "stop": 0.0},
    ("exactly_once", "not_committed"): {"query": 9.0, "direct": 10.0, "stop": 0.0},
    ("exactly_once", "committed"): {"query": 9.0, "direct": -30.0, "stop": 10.0},
    ("exactly_once", "unknown"): {"query": 9.0, "direct": -10.0, "stop": 5.0},
}

RESOURCES = [
    "billing-account", "storage-bucket", "analytics-dataset", "api-gateway",
    "report-worker", "cache-cluster", "billing-ledger", "search-index",
    "document-store", "notification-queue", "image-registry", "audit-log",
]
REQUESTS = [
    "refund-2041", "migration-7312", "cleanup-9188", "reindex-4402",
    "rotation-5521", "backfill-6610", "sync-7731", "archive-8842",
    "purge-3305", "replicate-2256", "restore-6184", "rebuild-9077",
]
OPERATIONS = [
    "rollback-3310", "deploy-8821", "payment-7742", "sync-5510",
    "migration-9921", "rotation-6630", "reindex-7712", "cleanup-4480",
    "transfer-2054", "checkpoint-3376", "publish-5581", "compact-9120",
]
VERSION_STYLES = ["r{n}", "v{n}", "rev-{n}", "2026.{n:02d}", "build-{n}"]


def _version(style: str, n: int) -> str:
    return style.format(n=n)


def _times(count: int, start_minute: int) -> list[str]:
    return [f"{9 + (start_minute + 3 * i) // 60:02d}:{(start_minute + 3 * i) % 60:02d}" for i in range(count)]


def _action(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


def _step(tool: str, observation: dict[str, Any], **arguments: Any) -> dict[str, Any]:
    return {"action": _action(tool, **arguments), "observation": observation}


# ---------------------------------------------------------------------------
# Public task text (task / tools / rules).  Semantics are identical within a
# mechanism; two wording variants add natural-language diversity across cells.
# ---------------------------------------------------------------------------

def mechanism_text(mechanism: str, variant: int) -> tuple[str, str, str]:
    """Return (tool_spec, public_rules, generic task suffix) for one mechanism."""
    if mechanism == "versioned_report":
        tool_spec = (
            "inspect_resource(resource_id): reads the exact current condition of one resource "
            "revision. It returns the resource, the revision observed, an evidence id, and the "
            "condition (good or bad). It costs 1 utility.\n"
            "execute(resource_id, request_id): runs the requested operation against the resource. "
            "It pays +10 when the current condition is good and -30 when it is bad.\n"
            "abort(request_id): stops the workflow without executing. It pays 0."
            if variant == 0 else
            "inspect_resource(resource_id) -> {resource_id, revision, evidence_id, condition}. "
            "Reads the exact condition of the revision it observes; cost 1.\n"
            "execute(resource_id, request_id) -> terminal. Pays +10 on a good current condition, "
            "-30 on a bad current condition.\n"
            "abort(request_id) -> terminal. Pays 0."
        )
        rules = (
            "1. An inspection report is bound to one resource and one revision, and it is accurate "
            "for exactly that revision.\n"
            "2. When the target resource moves to a new visible revision, an older report no longer "
            "describes the current condition; the new condition is drawn from the public prior 1/2.\n"
            "3. Version changes of unrelated resources and ordinary passage of time do not change "
            "the target condition and do not invalidate a report that matches the current revision.\n"
            "4. No further external version changes occur after this decision point.\n"
            "5. Unknown binary conditions have public prior 1/2."
            if variant == 0 else
            "Reports are revision-bound: a report applies only to the resource and revision it names. "
            "A visible revision change of the target resource makes prior reports stale and the new "
            "condition is redrawn from the 1/2 public prior. Unrelated resources and elapsed time "
            "neither change the target nor invalidate matching reports. No external version changes "
            "occur after this point. Unknown binary states use prior 1/2."
        )
        return tool_spec, rules, "Complete request {request} against resource {resource}."
    if mechanism == "scoped_authorization":
        tool_spec = (
            "query_authorization(resource_id, request_id): reads the exact current authorization for "
            "that resource and request pair. It returns allow or deny with an evidence id. It costs 1 utility.\n"
            "execute(resource_id, request_id): runs the requested operation. It pays +10 when the "
            "current authorization is allow and -30 when it is deny.\n"
            "abort(request_id): stops the workflow without executing. It pays 0."
            if variant == 0 else
            "query_authorization(resource_id, request_id) -> {authorization, evidence_id}. Reads the "
            "exact current authorization for that exact pair; cost 1.\n"
            "execute(resource_id, request_id) -> terminal. +10 if the current authorization is allow, "
            "-30 if deny.\n"
            "abort(request_id) -> terminal. Pays 0."
        )
        rules = (
            "1. Grants and denials are bound to one resource and one request pair.\n"
            "2. A valid grant scoped to the target pair is evidence that the current authorization is "
            "allow until it is explicitly revoked. An explicit revocation of the target grant means "
            "deny; an accurate query returning deny also establishes deny.\n"
            "3. A grant for a different resource or request provides no information about the target "
            "authorization. If no applicable target evidence exists, the target state is unknown and "
            "the public prior 1/2 applies; no separate certificate or possession check is required.\n"
            "4. Revoking a different grant does not affect the target authorization.\n"
            "5. Querying reads the current state accurately and cannot restore a revoked grant. No "
            "external authorization changes occur after this decision point."
            if variant == 0 else
            "Evidence is scoped to one (resource, request) pair. A valid grant to the target pair is "
            "evidence of allow until explicitly revoked; explicit revocation or an accurate query "
            "returning deny establishes deny. Grants and revocations for other pairs do not inform "
            "the target; with no applicable target evidence the state is unknown under prior 1/2, not "
            "automatically negative and not conditional on possessing a certificate. Queries only "
            "read the state and cannot re-grant. No external authorization changes occur after this point."
        )
        return tool_spec, rules, "Complete request {request} for resource {resource} under the authorization rules."
    if mechanism == "exactly_once":
        tool_spec = (
            "query_status(operation_id): reads the authoritative committed / not_committed state of "
            "that exact operation. It returns an evidence id. It costs 1 utility.\n"
            "retry(operation_id): executes the operation once. It pays +10 when the operation has not "
            "committed and -30 when it has already committed (duplicate side effect).\n"
            "close(operation_id): ends the workflow. It pays +10 if the operation has committed and no "
            "duplicate was caused; otherwise it pays 0."
            if variant == 0 else
            "query_status(operation_id) -> {status, evidence_id}. Authoritative only for the exact "
            "operation named; cost 1.\n"
            "retry(operation_id) -> terminal. +10 if not committed; -30 if already committed.\n"
            "close(operation_id) -> terminal. +10 if committed with no duplicate; otherwise 0."
        )
        rules = (
            "1. The target operation is non-idempotent and its submission returned a timeout. It may "
            "or may not have committed, with public probability 1/2 each; a timeout is not proof of failure.\n"
            "2. An authoritative status reading is accurate for the exact operation_id it names.\n"
            "3. Status readings for other operations provide no information about the target.\n"
            "4. No background completion occurs after this decision point; the committed state can "
            "only change through a retry.\n"
            "5. A retry executes once and succeeds when the operation has not committed; retrying an "
            "already committed operation creates a duplicate side effect."
            if variant == 0 else
            "The target submission timed out with public commit probability 1/2, so timeout is not "
            "failure. Status readings are authoritative only for the operation they name; other "
            "operations' readings say nothing about the target. No background completion occurs after "
            "this point; only a retry changes committed state. Retrying a not-committed operation "
            "succeeds; retrying a committed one duplicates the side effect."
        )
        return tool_spec, rules, "Ensure operation {operation} is completed exactly once."
    raise ValueError(mechanism)


# ---------------------------------------------------------------------------
# History construction.  Each variant is a public history plus a target id.
# ---------------------------------------------------------------------------

class FamilyBuilder:
    def __init__(self, index: int, mechanism: str, pair_type: str, rng: random.Random) -> None:
        self.index = index
        self.mechanism = mechanism
        self.pair_type = pair_type
        self.rng = rng
        self.resource = rng.choice(RESOURCES)
        self.other_resource = rng.choice([r for r in RESOURCES if r != self.resource])
        self.request = rng.choice(REQUESTS)
        self.other_request = rng.choice([q for q in REQUESTS if q != self.request])
        self.other_request_2 = rng.choice(
            [q for q in REQUESTS if q not in {self.request, self.other_request}]
        )
        self.operation = rng.choice(OPERATIONS)
        self.other_operation = rng.choice([o for o in OPERATIONS if o != self.operation])
        self.third_operation = rng.choice(
            [o for o in OPERATIONS if o not in {self.operation, self.other_operation}]
        )
        self.style = rng.choice(VERSION_STYLES)
        self.start = rng.randrange(0, 40)
        self.text_variant = index % 2
        self.rev1 = _version(self.style, rng.randrange(3, 20))
        self.rev2 = _version(self.style, rng.randrange(21, 40))
        self.ev1 = f"ev-{self.index:03d}-a"
        self.ev2 = f"ev-{self.index:03d}-b"
        self.ev3 = f"ev-{self.index:03d}-c"

    def task_text(self) -> str:
        tool_spec, rules, suffix = mechanism_text(self.mechanism, self.text_variant)
        if self.mechanism == "exactly_once":
            task = suffix.format(operation=self.operation)
        else:
            task = suffix.format(request=self.request, resource=self.resource)
        return task, tool_spec, rules

    # -- mechanism A -------------------------------------------------------
    def history_a(self, condition: str, stale: bool, unrelated_update: bool) -> list[dict[str, Any]]:
        times = _times(4, self.start)
        r, r2 = self.resource, self.other_resource
        steps = [
            _step("read_request", {"time": times[0], "request_id": self.request,
                                   "resource_id": r, "state": "open"},
                  request_id=self.request),
            _step("read_resource", {"time": times[1], "resource_id": r, "revision": self.rev1},
                  resource_id=r),
            _step("inspect_resource", {"time": times[2], "resource_id": r, "revision": self.rev1,
                                       "evidence_id": self.ev1, "condition": condition},
                  resource_id=r),
        ]
        if stale:
            steps.append(_step("update_resource",
                               {"time": times[3], "resource_id": r,
                                "previous_revision": self.rev1, "revision": self.rev2},
                               resource_id=r))
        elif unrelated_update:
            steps.append(_step("update_resource",
                               {"time": times[3], "resource_id": r2,
                                "previous_revision": _version(self.style, 41), "revision": _version(self.style, 42)},
                               resource_id=r2))
        else:
            steps.append(_step("read_request", {"time": times[3], "request_id": self.other_request,
                                                "resource_id": r2, "state": "open"},
                               request_id=self.other_request))
        return steps

    def history_a_unknown_invariant(self) -> list[dict[str, Any]]:
        times = _times(4, self.start)
        r, r2 = self.resource, self.other_resource
        return [
            _step("inspect_resource", {"time": times[0], "resource_id": r, "revision": self.rev1,
                                       "evidence_id": self.ev1, "condition": "good"},
                  resource_id=r),
            _step("update_resource", {"time": times[1], "resource_id": r,
                                      "previous_revision": self.rev1, "revision": self.rev2},
                  resource_id=r),
            _step("read_resource", {"time": times[2], "resource_id": r, "revision": self.rev2}, resource_id=r),
            _step("read_request", {"time": times[3], "request_id": self.other_request,
                                   "resource_id": r2, "state": "open"}, request_id=self.other_request),
        ]

    # -- mechanism B -------------------------------------------------------
    def history_b(self, mode: str) -> list[dict[str, Any]]:
        """Four-round history.  The last round is a variant-specific unrelated event
        supplied by build_family for invariant families; reversal modes fix it here."""
        times = _times(4, self.start)
        r, r2 = self.resource, self.other_resource
        q, q2 = self.request, self.other_request
        steps = [
            _step("read_request", {"time": times[0], "request_id": q, "resource_id": r, "state": "open"},
                  request_id=q),
        ]
        if mode in {"valid", "revoked", "allow_invariant", "deny_invariant"}:
            steps.append(_step("grant_authorization",
                               {"time": times[1], "resource_id": r, "request_id": q,
                                "evidence_id": self.ev1, "authorization": "allow"},
                               resource_id=r, request_id=q))
        elif mode in {"other_scope", "unknown_invariant"}:
            # A grant with the same resource but a different request says nothing about the target pair.
            steps.append(_step("grant_authorization",
                               {"time": times[1], "resource_id": r, "request_id": q2,
                                "evidence_id": self.ev1, "authorization": "allow"},
                               resource_id=r, request_id=q2))
        else:
            raise ValueError(mode)
        if mode in {"revoked", "deny_invariant"}:
            steps.append(_step("revoke_authorization",
                               {"time": times[2], "resource_id": r, "request_id": q,
                                "evidence_id": self.ev2, "revoked": True,
                                "explicit_reference": self.ev1},
                               resource_id=r, request_id=q))
        elif mode in {"other_scope", "unknown_invariant"}:
            steps.append(_step("grant_authorization",
                               {"time": times[2], "resource_id": r2, "request_id": q2,
                                "evidence_id": self.ev3, "authorization": "allow"},
                               resource_id=r2, request_id=q2))
        else:  # valid / allow_invariant: an unrelated request read
            steps.append(_step("read_request",
                               {"time": times[2], "request_id": q2, "resource_id": r2, "state": "open"},
                               request_id=q2))
        # Fourth round: an unrelated event only.  Invariant variants replace it later.
        if mode in {"unknown_invariant"}:
            steps.append(_step("read_request",
                               {"time": times[3], "request_id": q2, "resource_id": r2, "state": "open"},
                               request_id=q2))
        else:
            steps.append(_step("read_resource", {"time": times[3], "resource_id": r2,
                                                "revision": _version(self.style, 43)}, resource_id=r2))
        return steps

    # -- mechanism C -------------------------------------------------------
    def history_c(self, mode: str, committed: bool = False) -> list[dict[str, Any]]:
        """mode: known_not | unknown_not | known_committed | unknown_committed | invariant_known | invariant_unknown."""
        times = _times(4, self.start)
        op, op2 = self.operation, self.other_operation
        status = "committed" if committed else "not_committed"
        steps = [
            _step("submit_operation", {"time": times[0], "operation_id": op, "status": "timeout"},
                  operation_id=op),
            _step("submit_operation", {"time": times[1], "operation_id": op2, "status": "timeout"},
                  operation_id=op2),
        ]
        if mode in {"known_not", "known_committed", "invariant_known"}:
            steps.append(_step("query_status",
                               {"time": times[2], "operation_id": op, "evidence_id": self.ev1,
                                "status": status}, operation_id=op))
        elif mode in {"unknown_not", "unknown_committed", "invariant_unknown"}:
            # The authoritative reading belongs to another operation and says nothing about the target.
            steps.append(_step("query_status",
                               {"time": times[2], "operation_id": op2, "evidence_id": self.ev1,
                                "status": status}, operation_id=op2))
        else:
            raise ValueError(mode)
        # Fourth round: never contradicts or erases an authoritative target reading.
        if mode == "known_committed" or (mode == "invariant_known" and committed):
            steps.append(_step("query_status",
                               {"time": times[3], "operation_id": op2, "evidence_id": self.ev3,
                                "status": "not_committed"}, operation_id=op2))
        else:
            steps.append(_step("read_operation", {"time": times[3], "operation_id": op2,
                                                  "state": "submitted"}, operation_id=op2))
        return steps


# ---------------------------------------------------------------------------
# Exact posterior and Q^{pi0}.
# ---------------------------------------------------------------------------

def target_of(mechanism: str, builder: FamilyBuilder) -> dict[str, str]:
    if mechanism == "versioned_report":
        return {"resource_id": builder.resource, "request_id": builder.request}
    if mechanism == "scoped_authorization":
        return {"resource_id": builder.resource, "request_id": builder.request}
    return {"operation_id": builder.operation}


ALLOWED_STATES = {
    "versioned_report": ("good", "bad", "unknown"),
    "scoped_authorization": ("allow", "deny", "unknown"),
    "exactly_once": ("not_committed", "committed", "unknown"),
}


def allowed_states(mechanism: str) -> tuple[str, ...]:
    return ALLOWED_STATES[mechanism]


def target_from_public(mechanism: str, public: dict[str, Any]) -> dict[str, str]:
    """Recover public target identifiers from candidate arguments only."""
    candidates = public["candidates"]
    if mechanism == "exactly_once":
        for candidate in candidates:
            operation = candidate.get("arguments", {}).get("operation_id")
            if operation:
                return {"operation_id": operation}
    else:
        for candidate in candidates:
            arguments = candidate.get("arguments", {})
            if arguments.get("resource_id") and arguments.get("request_id"):
                return {"resource_id": arguments["resource_id"],
                        "request_id": arguments["request_id"]}
    raise ValueError(f"cannot recover target identifiers for {mechanism}")


def public_target_text(mechanism: str, target: dict[str, str]) -> str:
    if mechanism == "exactly_once":
        return f"operation_id={target['operation_id']}"
    return f"resource_id={target['resource_id']}, request_id={target['request_id']}"


def state_from_worlds(mechanism: str, worlds: list[dict[str, Any]]) -> str:
    if len(worlds) != 1:
        return "unknown"
    world = worlds[0]
    if mechanism == "versioned_report":
        return world["condition"]
    if mechanism == "scoped_authorization":
        return world["authorization"]
    return "committed" if world["committed"] else "not_committed"


def information_state(mechanism: str, history: list[dict[str, Any]],
                      target: dict[str, str]) -> str:
    """O_STATE: the information state implied by public history, not a hidden bit."""
    worlds, _ = posterior(mechanism, history, target)
    return state_from_worlds(mechanism, worlds)


def state_q_table(mechanism: str, state: str, remaining: int) -> dict[str, float]:
    """The fixed CPU Q table in the G5 protocol (sections 7 and 5)."""
    if mechanism == "versioned_report":
        rows = {
            "good": {"query": 9.0, "direct": 10.0, "stop": 0.0},
            "bad": {"query": -1.0, "direct": -30.0, "stop": 0.0},
            "unknown": {"query": 4.0, "direct": -10.0, "stop": 0.0},
        }
    elif mechanism == "scoped_authorization":
        rows = {
            "allow": {"query": 9.0, "direct": 10.0, "stop": 0.0},
            "deny": {"query": -1.0, "direct": -30.0, "stop": 0.0},
            "unknown": {"query": 4.0, "direct": -10.0, "stop": 0.0},
        }
    else:
        rows = {
            "not_committed": {"query": 9.0, "direct": 10.0, "stop": 0.0},
            "committed": {"query": 9.0, "direct": -30.0, "stop": 10.0},
            "unknown": {"query": 9.0, "direct": -10.0, "stop": 5.0},
        }
    if state == "unknown" and mechanism == "exactly_once":
        rows["unknown"] = {"query": 9.0, "direct": -10.0, "stop": 5.0}
    values = dict(rows[state])
    if remaining <= 1:
        if mechanism == "exactly_once":
            close_value = {"not_committed": 0.0, "committed": 10.0, "unknown": 5.0}[state]
            values["query"] = -1.0 + close_value
        else:
            values["query"] = -1.0
    return values


def posterior(mechanism: str, history: list[dict[str, Any]], target: dict[str, str]
              ) -> tuple[list[dict[str, Any]], list[float]]:
    """Exact posterior over hidden worlds from public history only."""
    if mechanism == "versioned_report":
        current_rev = None
        report_value = None
        for step in history:
            obs = step["observation"]
            if obs.get("resource_id") != target["resource_id"]:
                continue
            if "previous_revision" in obs:  # visible revision change
                current_rev = obs["revision"]
                report_value = None  # an older report cannot describe the new revision
            elif "revision" in obs:
                current_rev = obs["revision"]
                if "condition" in obs:
                    report_value = obs["condition"]
        if report_value is not None:
            return [{"condition": report_value}], [1.0]
        return [{"condition": "good"}, {"condition": "bad"}], [0.5, 0.5]
    if mechanism == "scoped_authorization":
        state = None
        for step in history:
            obs = step["observation"]
            if obs.get("resource_id") != target["resource_id"] or obs.get("request_id") != target["request_id"]:
                continue
            if obs.get("revoked") is True:
                state = "deny"
            elif obs.get("authorization") in {"allow", "deny"}:
                # An accurate query returning deny is evidence, just like a
                # grant/revocation. G4 omitted this branch.
                state = obs["authorization"]
        if state is not None:
            return [{"authorization": state}], [1.0]
        return [{"authorization": "allow"}, {"authorization": "deny"}], [0.5, 0.5]
    if mechanism == "exactly_once":
        state = None
        for step in history:
            obs = step["observation"]
            if obs.get("operation_id") != target["operation_id"]:
                continue
            if obs.get("status") in {"committed", "not_committed"}:
                state = obs["status"]
        if state is not None:
            return [{"committed": state == "committed"}], [1.0]
        return [{"committed": False}, {"committed": True}], [0.5, 0.5]
    raise ValueError(mechanism)


def _positive(mechanism: str, world: dict[str, Any]) -> bool:
    if mechanism == "versioned_report":
        return world["condition"] == "good"
    if mechanism == "scoped_authorization":
        return world["authorization"] == "allow"
    return world["committed"] is False  # retry is safe only when not committed


def _terminal_utility(mechanism: str, kind: str, world: dict[str, Any]) -> float:
    """kind: query | direct | stop.  Query is handled separately (cost + continuation)."""
    if kind == "stop":
        if mechanism == "exactly_once":
            return 10.0 if world["committed"] else 0.0
        return 0.0
    if kind == "direct":
        return 10.0 if _positive(mechanism, world) else -30.0
    raise ValueError(kind)


def _pi0_action(mechanism: str, worlds: list[dict[str, Any]], remaining: int) -> str:
    """Fixed continuation: returns 'direct' or 'stop' (never called at unknown with time left)."""
    positive = _positive(mechanism, worlds[0])
    return "direct" if positive else "stop"


def q_value(mechanism: str, history: list[dict[str, Any]], target: dict[str, str],
            kind: str, remaining: int) -> float:
    """Q^{pi0}(h,a) for one candidate category (query/direct/stop)."""
    worlds, weights = posterior(mechanism, history, target)
    if kind == "query":
        if remaining <= 1:
            # No action follows the query inside the declared horizon. For
            # execution status the horizon closes the operation implicitly.
            if mechanism == "exactly_once":
                close = sum(weight * _terminal_utility(mechanism, "stop", world)
                            for world, weight in zip(worlds, weights))
                return -1.0 + close
            return -1.0
        total = 0.0
        for world, weight in zip(worlds, weights):
            # The query reveals the truth to pi0; it then acts without further search.
            continuation = _terminal_utility(mechanism, _pi0_action(mechanism, [world], remaining - 1), world)
            total += weight * (-1.0 + continuation)
        return total
    return sum(weight * _terminal_utility(mechanism, kind, world) for world, weight in zip(worlds, weights))


def candidate_kinds(mechanism: str) -> dict[str, str]:
    if mechanism == "versioned_report":
        return {"query": "inspect_resource", "direct": "execute", "stop": "abort"}
    if mechanism == "scoped_authorization":
        return {"query": "query_authorization", "direct": "execute", "stop": "abort"}
    return {"query": "query_status", "direct": "retry", "stop": "close"}


def candidates_for(mechanism: str, builder: FamilyBuilder) -> list[dict[str, Any]]:
    if mechanism == "versioned_report":
        return [
            _action("inspect_resource", resource_id=builder.resource),
            _action("execute", resource_id=builder.resource, request_id=builder.request),
            _action("abort", request_id=builder.request),
        ]
    if mechanism == "scoped_authorization":
        return [
            _action("query_authorization", resource_id=builder.resource, request_id=builder.request),
            _action("execute", resource_id=builder.resource, request_id=builder.request),
            _action("abort", request_id=builder.request),
        ]
    return [
        _action("query_status", operation_id=builder.operation),
        _action("retry", operation_id=builder.operation),
        _action("close", operation_id=builder.operation),
    ]


def kind_of_action(mechanism: str, action: dict[str, Any]) -> str:
    kinds = candidate_kinds(mechanism)
    for kind, tool in kinds.items():
        if action["tool"] == tool:
            return kind
    raise ValueError(action)


def query_observations(mechanism: str, builder: FamilyBuilder, world: dict[str, Any],
                       history: list[dict[str, Any]], time: str) -> dict[str, Any]:
    if mechanism == "versioned_report":
        rev = None
        for step in history:
            obs = step["observation"]
            if obs.get("resource_id") == builder.resource and "revision" in obs:
                rev = obs["revision"]
        return {"time": time, "resource_id": builder.resource, "revision": rev,
                "evidence_id": builder.ev2, "condition": world["condition"]}
    if mechanism == "scoped_authorization":
        return {"time": time, "resource_id": builder.resource, "request_id": builder.request,
                "evidence_id": builder.ev2, "authorization": world["authorization"]}
    return {"time": time, "operation_id": builder.operation, "evidence_id": builder.ev2,
            "status": "committed" if world["committed"] else "not_committed"}


def build_prefix(mechanism: str, builder: FamilyBuilder, history: list[dict[str, Any]],
                 remaining: int, candidate_map: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    task, tool_spec, rules = builder.task_text()
    target = target_of(mechanism, builder)
    worlds, weights = posterior(mechanism, history, target)
    q = {}
    kinds = candidate_kinds(mechanism)
    for candidate in candidate_map:
        kind = kind_of_action(mechanism, candidate)
        q[candidate["candidate_id"]] = q_value(mechanism, history, target, kind, remaining)
    best = max(q.values())
    gold = sorted(candidate["candidate_id"] for candidate in candidate_map if best - q[candidate["candidate_id"]] <= 1e-9)
    public = {
        "task": task, "tool_spec": tool_spec, "public_rules": rules,
        "history": history, "candidates": candidate_map, "remaining_steps": remaining,
    }
    private = {
        "worlds": worlds, "posterior": weights, "q_pi0": q, "gold_actions": gold,
        "target": target,
    }
    return public, private


def _family_skeleton(index: int, mechanism: str, pair_type: str, subtype: str,
                     split: str, rng: random.Random) -> dict[str, Any]:
    return {
        "family_id": f"F{index:04d}",
        "split": split,
        "mechanism": mechanism,
        "pair_type": pair_type,
        "subtype": subtype,
        "cell": f"{mechanism}:{pair_type}",
    }


def _candidate_map(mechanism: str, builder: FamilyBuilder) -> list[dict[str, Any]]:
    candidates = candidates_for(mechanism, builder)
    order = list(range(3))
    builder.rng.shuffle(order)
    return [{"candidate_id": f"c{position}", **candidates[source]} for position, source in enumerate(order)]


def build_family(index: int, mechanism: str, pair_type: str, subtype: str, split: str,
                 rng: random.Random) -> dict[str, Any]:
    builder = FamilyBuilder(index, mechanism, pair_type, rng)
    family = _family_skeleton(index, mechanism, pair_type, subtype, split, rng)
    candidate_map = _candidate_map(mechanism, builder)

    histories: list[tuple[str, list[dict[str, Any]]]]
    if mechanism == "versioned_report":
        if pair_type == "reversal":
            condition = "good" if subtype == "reversal_good" else "bad"
            histories = [
                ("current", builder.history_a(condition, stale=False, unrelated_update=False)),
                ("stale", builder.history_a(condition, stale=True, unrelated_update=False)),
            ]
        else:
            if subtype == "invariant_good":
                histories = [
                    ("v0", builder.history_a("good", stale=False, unrelated_update=False)),
                    ("v1", builder.history_a("good", stale=False, unrelated_update=True)),
                ]
            elif subtype == "invariant_bad":
                histories = [
                    ("v0", builder.history_a("bad", stale=False, unrelated_update=False)),
                    ("v1", builder.history_a("bad", stale=False, unrelated_update=True)),
                ]
            else:
                base = builder.history_a_unknown_invariant()
                alt = list(base)
                alt[-1] = _step("update_resource",
                                {"time": base[-1]["observation"]["time"], "resource_id": builder.other_resource,
                                 "previous_revision": _version(builder.style, 41),
                                 "revision": _version(builder.style, 42)},
                                resource_id=builder.other_resource)
                histories = [("v0", base), ("v1", alt)]
    elif mechanism == "scoped_authorization":
        if pair_type == "reversal":
            if subtype == "reversal_revoked":
                histories = [("valid", builder.history_b("valid")), ("revoked", builder.history_b("revoked"))]
            else:
                histories = [("valid", builder.history_b("valid")), ("other_scope", builder.history_b("other_scope"))]
        else:
            if subtype in {"invariant_allow", "invariant_deny"}:
                base = builder.history_b("allow_invariant" if subtype == "invariant_allow" else "deny_invariant")
                alt = list(base)
                alt[-1] = _step("grant_authorization",
                                {"time": base[-1]["observation"]["time"], "resource_id": builder.other_resource,
                                 "request_id": builder.other_request, "evidence_id": builder.ev3,
                                 "authorization": "allow"},
                                resource_id=builder.other_resource, request_id=builder.other_request)
                histories = [("v0", base), ("v1", alt)]
            else:
                base = builder.history_b("unknown_invariant")
                alt = list(base)
                alt[-1] = _step("read_resource",
                                {"time": base[-1]["observation"]["time"], "resource_id": builder.other_resource,
                                 "revision": _version(builder.style, 43)},
                                resource_id=builder.other_resource)
                histories = [("v0", base), ("v1", alt)]
    else:  # exactly_once
        if pair_type == "reversal":
            if subtype == "reversal_not_committed":
                histories = [("known_not", builder.history_c("known_not")),
                             ("unknown_not", builder.history_c("unknown_not"))]
            else:
                histories = [("known_committed", builder.history_c("known_committed", committed=True)),
                             ("unknown_committed", builder.history_c("unknown_committed", committed=True))]
        else:
            if subtype == "invariant_not_committed":
                base = builder.history_c("invariant_known")
                alt = list(base)
                alt[-1] = _step("query_status",
                                {"time": base[-1]["observation"]["time"],
                                 "operation_id": builder.other_operation, "evidence_id": builder.ev3,
                                 "status": "committed"}, operation_id=builder.other_operation)
                histories = [("v0", base), ("v1", alt)]
            elif subtype == "invariant_committed":
                base = builder.history_c("invariant_known", committed=True)
                alt = list(base)
                alt[-1] = _step("read_operation",
                                {"time": base[-1]["observation"]["time"],
                                 "operation_id": builder.other_operation, "state": "submitted"},
                                operation_id=builder.other_operation)
                histories = [("v0", base), ("v1", alt)]
            else:
                base = builder.history_c("invariant_unknown")
                alt = list(base)
                alt[-1] = _step("submit_operation",
                                {"time": base[-1]["observation"]["time"],
                                 "operation_id": builder.third_operation, "status": "timeout"},
                                operation_id=builder.third_operation)
                histories = [("v0", base), ("v1", alt)]

    variants = []
    for variant_index, (label, history) in enumerate(histories):
        public, private = build_prefix(mechanism, builder, history, 4, candidate_map)
        query_candidate = next(c for c in candidate_map if kind_of_action(mechanism, c) == "query")
        secondary = []
        worlds, weights = private["worlds"], private["posterior"]
        for world, weight in zip(worlds, weights):
            time = f"10:{variant_index * 10 + 5:02d}"
            observation = query_observations(mechanism, builder, world, history, time)
            new_history = history + [{"action": dict(query_candidate), "observation": observation}]
            sec_public, sec_private = build_prefix(mechanism, builder, new_history, 3, candidate_map)
            sec_private["branch_posterior"] = weight
            sec_public["query_world_index"] = worlds.index(world)
            secondary.append({"prefix_id": f"{family['family_id']}:{label}:sec{witness(history, world, mechanism, query_candidate)}",
                              "public": sec_public, "private": sec_private})
        variants.append({
            "variant_id": f"{family['family_id']}:{label}",
            "label": label,
            "public": public,
            "private": private,
            "prefixes": secondary,
        })
    family["variants"] = variants
    return family


def witness(history: list[dict[str, Any]], world: dict[str, Any], mechanism: str,
            query_candidate: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps([history, world, mechanism], sort_keys=True).encode()).hexdigest()
    return digest[:6]


# ---------------------------------------------------------------------------
# Dataset construction, balancing and freezing.
# ---------------------------------------------------------------------------

def _subtype_sequence(mechanism: str, pair_type: str, rng: random.Random) -> list[str]:
    if pair_type == "reversal":
        if mechanism == "versioned_report":
            subtypes = ["reversal_good", "reversal_bad"]
        elif mechanism == "scoped_authorization":
            subtypes = ["reversal_revoked", "reversal_other_scope"]
        else:
            subtypes = ["reversal_not_committed", "reversal_committed"]
        # exactly half in each split: dev 3/3, audit 12/12
        dev = [subtypes[0]] * 3 + [subtypes[1]] * 3
        audit = [subtypes[0]] * 12 + [subtypes[1]] * 12
        rng.shuffle(dev)
        rng.shuffle(audit)
        return dev + audit
    if mechanism == "versioned_report":
        subtypes = ["invariant_good", "invariant_bad", "invariant_unknown"]
    elif mechanism == "scoped_authorization":
        subtypes = ["invariant_allow", "invariant_deny", "invariant_unknown"]
    else:
        subtypes = ["invariant_not_committed", "invariant_committed", "invariant_unknown"]
    dev = [s for s in subtypes for _ in range(2)]
    audit = [s for s in subtypes for _ in range(8)]
    rng.shuffle(dev)
    rng.shuffle(audit)
    return dev + audit


def _validate_family(family: dict[str, Any]) -> None:
    mechanism = family["mechanism"]
    for variant in family["variants"]:
        pub, priv = variant["public"], variant["private"]
        q = priv["q_pi0"]
        values = sorted(q.values(), reverse=True)
        # Unique optimum and a top-2 gap of at least 0.5 at every primary node.
        if len(values) < 3 or values[0] - values[1] < 0.5 - 1e-9:
            raise ValueError(f"{family['family_id']}: primary top-2 gap too small: {q}")
        if len(priv["gold_actions"]) != 1:
            raise ValueError(f"{family['family_id']}: gold not unique: {priv['gold_actions']}")
        for prefix in variant["prefixes"]:
            sec_values = sorted(prefix["private"]["q_pi0"].values(), reverse=True)
            if len(sec_values) < 3 or sec_values[0] - sec_values[1] < 0.5 - 1e-9:
                raise ValueError(f"{family['family_id']}: secondary gap too small")
            if len(prefix["private"]["gold_actions"]) != 1:
                raise ValueError(f"{family['family_id']}: secondary gold not unique")
    if family["pair_type"] == "reversal":
        golds = [variant["private"]["gold_actions"][0] for variant in family["variants"]]
        if golds[0] == golds[1]:
            raise ValueError(f"{family['family_id']}: reversal endpoints share the same gold action")
    else:
        golds = [variant["private"]["gold_actions"][0] for variant in family["variants"]]
        if golds[0] != golds[1]:
            raise ValueError(f"{family['family_id']}: invariant endpoints differ in gold action")


def verify_analytic_table() -> None:
    """Build minimal one-state histories and check the protocol's six-state table."""
    rng = random.Random(101)
    for mechanism in MECHANISMS:
        builder = FamilyBuilder(999, mechanism, "invariant", rng)
        kinds = candidate_kinds(mechanism)
        states = {
            "versioned_report": ["good", "bad", "unknown"],
            "scoped_authorization": ["allow", "deny", "unknown"],
            "exactly_once": ["not_committed", "committed", "unknown"],
        }[mechanism]
        for state in states:
            if mechanism == "versioned_report":
                if state in {"good", "bad"}:
                    history = builder.history_a(state, stale=False, unrelated_update=False)
                else:
                    history = builder.history_a_unknown_invariant()
            elif mechanism == "scoped_authorization":
                mode = {"allow": "allow_invariant", "deny": "deny_invariant", "unknown": "unknown_invariant"}[state]
                history = builder.history_b(mode)
            else:
                mode = {"not_committed": "known_not", "committed": "known_committed",
                        "unknown": "unknown_not"}[state]
                history = builder.history_c(mode, committed=(state == "committed"))
            target = target_of(mechanism, builder)
            got = {kind: q_value(mechanism, history, target, kind, 4) for kind in kinds}
            expected = ANALYTIC_TABLE[(mechanism, state)]
            for kind in kinds:
                if abs(got[kind] - expected[kind]) > 1e-9:
                    raise ValueError(f"analytic state mismatch {mechanism}/{state}/{kind}: {got[kind]} vs {expected[kind]}")


def construct_families(data_seed: int = DATA_SEED, split_seed: int = SPLIT_SEED
                       ) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, int]]:
    rng = random.Random(data_seed)
    split_rng = random.Random(split_seed)
    families: list[dict[str, Any]] = []
    cell_counts: dict[str, dict[str, int]] = {}
    rejections: Counter[str] = Counter()
    index = 0
    for mechanism in MECHANISMS:
        for pair_type in PAIR_TYPES:
            subtypes = _subtype_sequence(mechanism, pair_type, split_rng)
            cell_counts[f"{mechanism}:{pair_type}"] = {"attempted": 0, "accepted": 0}
            for position, subtype in enumerate(subtypes):
                split = "dev" if position < N_DEV_PER_CELL else "audit"
                accepted = None
                for attempt in range(50):
                    cell_counts[f"{mechanism}:{pair_type}"]["attempted"] += 1
                    family = build_family(index, mechanism, pair_type, subtype, split, rng)
                    try:
                        _validate_family(family)
                    except ValueError:
                        rejections[f"{mechanism}:{pair_type}"] += 1
                        continue
                    accepted = family
                    break
                if accepted is None:
                    raise RuntimeError(f"could not build a valid {mechanism}/{pair_type}/{subtype} family")
                cell_counts[f"{mechanism}:{pair_type}"]["accepted"] += 1
                families.append(accepted)
                index += 1
    return families, cell_counts, dict(rejections)


def build_dataset(output_dir: Path, data_seed: int = DATA_SEED, split_seed: int = SPLIT_SEED,
                  dataset_version: str = "g4-evidence-v1") -> dict[str, Any]:
    verify_analytic_table()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    manifest_path = output_dir / "manifest.json"
    if cases_path.exists() or manifest_path.exists():
        raise FileExistsError("dataset already exists; data are generated exactly once")

    families, cell_counts, rejections = construct_families(data_seed, split_seed)

    # Deterministic closed-loop subset: first 6 audit families per cell, 6 reversal + 6 invariant per mechanism.
    closed_loop_ids = []
    for mechanism in MECHANISMS:
        for pair_type in PAIR_TYPES:
            audit_ids = sorted(f["family_id"] for f in families
                               if f["mechanism"] == mechanism and f["pair_type"] == pair_type and f["split"] == "audit")
            closed_loop_ids.extend(audit_ids[:6])
    closed_loop_set = set(closed_loop_ids)
    for family in families:
        family["closed_loop"] = family["family_id"] in closed_loop_set

    with cases_path.open("x", encoding="utf-8") as stream:
        for family in families:
            stream.write(json.dumps(family, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    split_counts = Counter(family["split"] for family in families)
    prefix_counts = {
        split: sum(1 + len(v["prefixes"]) for f in families if f["split"] == split for v in f["variants"])
        for split in ("dev", "audit")
    }
    manifest = {
        "dataset": "Evidence-sensitivity trajectories; exact-expectation hidden-world labels",
        "dataset_version": dataset_version,
        "call_key_namespace": f"{dataset_version}:{data_seed}:{split_seed}",
        "data_seed": data_seed, "split_seed": split_seed, "bootstrap_seed": 23,
        "family_counts": dict(split_counts),
        "cell_counts": cell_counts,
        "pair_type_counts": dict(Counter(family["pair_type"] for family in families)),
        "subtype_counts": dict(Counter(family["subtype"] for family in families)),
        "mechanism_counts": dict(Counter(family["mechanism"] for family in families)),
        "prefix_counts": prefix_counts,
        "rejections": rejections,
        "analytic_table_verified": True,
        "closed_loop_family_ids": closed_loop_ids,
        "closed_loop_counts": dict(Counter(
            f["mechanism"] + ":" + f["pair_type"] for f in families if f["closed_loop"]
        )),
        "candidate_permutation": "per-family seeded shuffle; both variants and all prefixes share the mapping; c0 is not always gold",
        "split_unit": "whole family (both variants, all prefixes and views)",
        "prompt_visibility": "public block only; private worlds/posterior/gold never enter model inputs",
        "focus": "finite local tool simulator; 3-5 historical rounds; max 4 actions from the primary decision",
        "counterfactual_rule": "same family fixes task, tools, candidates and continuation; only relevant evidence or one unrelated event differs",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "frozen": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# ---------------------------------------------------------------------------
# CPU baselines and symbolic planner.
# ---------------------------------------------------------------------------

def policy_action(mechanism: str, candidate_map: list[dict[str, Any]], history: list[dict[str, Any]],
                  target: dict[str, str], name: str) -> dict[str, Any]:
    """CPU baselines: pi0, always_query, always_direct, always_stop, exact_planner."""
    kinds = candidate_kinds(mechanism)
    by_kind = {kind_of_action(mechanism, c): c for c in candidate_map}
    if name == "always_query":
        return by_kind["query"]
    if name == "always_direct":
        return by_kind["direct"]
    if name == "always_stop":
        return by_kind["stop"]
    worlds, _ = posterior(mechanism, history, target)
    known = len(worlds) == 1
    if name in {"pi0", "exact_planner"}:
        if known:
            return by_kind["direct"] if _positive(mechanism, worlds[0]) else by_kind["stop"]
        return by_kind["query"]
    raise ValueError(name)


def regret_of_action(private: dict[str, Any], action_id: str) -> float:
    q = private["q_pi0"]
    return max(q.values()) - q[action_id]


def evaluate_cpu_baselines(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Primary-prefix baseline metrics on audit split (family mean over variants)."""
    out: dict[str, Any] = {}
    for name in ("pi0", "exact_planner", "always_query", "always_direct", "always_stop"):
        by_cell: dict[str, list[float]] = {}
        optimal: dict[str, list[float]] = {}
        for family in cases:
            if family["split"] != "audit":
                continue
            regrets = []
            opt = []
            for variant in family["variants"]:
                pub, priv = variant["public"], variant["private"]
                action = policy_action(family["mechanism"], pub["candidates"], pub["history"],
                                       priv["target"], name)
                q = priv["q_pi0"]
                regrets.append(regret_of_action(priv, action["candidate_id"]))
                opt.append(1.0 if abs(q[action["candidate_id"]] - max(q.values())) <= 1e-9 else 0.0)
            by_cell.setdefault(family["cell"], []).append(sum(regrets) / len(regrets))
            optimal.setdefault(family["cell"], []).append(sum(opt) / len(opt))
        cells = {cell: {"mean_regret": sum(v) / len(v), "optimal_action_rate": sum(optimal[cell]) / len(optimal[cell]),
                        "n_families": len(v)} for cell, v in by_cell.items()}
        out[name] = {
            "macro_regret": sum(c["mean_regret"] for c in cells.values()) / len(cells),
            "macro_optimal_action_rate": sum(c["optimal_action_rate"] for c in cells.values()) / len(cells),
            "cells": cells,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data" / "g4_evidence")
    parser.add_argument("--self-test", action="store_true", help="Run the CPU verification only and exit")
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--dataset-version", default="g4-evidence-v1")
    args = parser.parse_args()
    if args.self_test:
        verify_analytic_table()
        families, cell_counts, rejections = construct_families(args.data_seed, args.split_seed)
        assert len(families) == 180, len(families)
        assert Counter(f["split"] for f in families) == {"dev": 36, "audit": 144}
        assert len({f["family_id"] for f in families}) == 180
        cell_split = Counter((f["cell"], f["split"]) for f in families)
        assert all(cell_split[(cell, split)] == (6 if split == "dev" else 24)
                   for cell in cell_counts for split in ("dev", "audit"))
        # invariant gold-action balance: dev 2 each, audit 8 each
        gold_counter = Counter()
        for family in families:
            if family["pair_type"] != "invariant":
                continue
            gold_kind = {
                "versioned_report": {"execute": "direct", "abort": "stop", "inspect_resource": "query"},
                "scoped_authorization": {"execute": "direct", "abort": "stop", "query_authorization": "query"},
                "exactly_once": {"retry": "direct", "close": "stop", "query_status": "query"},
            }[family["mechanism"]]
            action = family["variants"][0]["public"]["candidates"]
            gold_id = family["variants"][0]["private"]["gold_actions"][0]
            tool = next(c["tool"] for c in action if c["candidate_id"] == gold_id)
            gold_counter[(family["mechanism"], family["split"], gold_kind[tool])] += 1
        for mechanism in MECHANISMS:
            for split, expected in (("dev", 2), ("audit", 8)):
                for kind in ("direct", "stop", "query"):
                    assert gold_counter[(mechanism, split, kind)] == expected, (
                        mechanism, split, kind, gold_counter)
        # c0 is not always gold; both variants share one candidate mapping
        c0_gold = sum(1 for f in families
                      if all(v["private"]["gold_actions"] == ["c0"] for v in f["variants"]))
        prefix_counts = Counter(len(v["prefixes"]) for f in families for v in f["variants"])
        assert set(prefix_counts) <= {1, 2}, prefix_counts
        print(json.dumps({
            "self_test": "pass", "analytic_rows": len(ANALYTIC_TABLE),
            "families": len(families), "cells": cell_counts, "rejections": rejections,
            "invariant_gold": {f"{k[0]}:{k[1]}": v for k, v in sorted(gold_counter.items())},
            "families_with_c0_gold_at_both_ends": c0_gold,
            "secondary_branches_per_variant": dict(prefix_counts),
        }, indent=2))
        return
    manifest = build_dataset(args.output_dir, args.data_seed, args.split_seed, args.dataset_version)
    cases = load_cases(args.output_dir / "cases.jsonl")
    baselines = evaluate_cpu_baselines(cases)
    manifest["cpu_baselines"] = baselines
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "baselines": baselines}, indent=2))


if __name__ == "__main__":
    main()
