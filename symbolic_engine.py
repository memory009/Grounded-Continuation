#!/usr/bin/env python3
"""
Symbolic Epistemic Engine — Reference Implementation

This module implements the core symbolic engine described in the paper:
- Hypothesis lifecycle tracking (active → weakened → abandoned → resolved)
- Observation registry with explanatory links
- Dependency tracking: Dep(hypothesis) → {assumptions}
- Counterfactual reasoning: Affected(assumption) (dependency soundness)
- Awareness set management
- Consistency checking

The engine processes epistemic operations produced by the LLM Interpreter
and maintains the formal model. It demonstrates the hybrid architecture's
symbolic component on the Phase 2 debugging scenario (13 turns).

Usage:
    python symbolic_engine.py              # Run Phase 2 demo
    python symbolic_engine.py --counterfactual  # Run with counterfactual analysis
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ============================================================
# Core Data Structures
# ============================================================

class HypothesisStatus(Enum):
    ACTIVE = "active"
    WEAKENED = "weakened"
    ABANDONED = "abandoned"
    RESOLVED = "resolved"


@dataclass
class Observation:
    id: str
    content: str
    turn: str
    speaker: str
    explained_by: Optional[str] = None


@dataclass
class Hypothesis:
    id: str
    content: str
    turn: str
    speaker: str
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    plausibility: str = "medium"  # low, medium, high, certain, none
    explains: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    undermines: list = field(default_factory=list)
    abandoned_reason: Optional[str] = None
    abandoned_turn: Optional[str] = None


@dataclass
class EngineEvent:
    operation: str
    turn: str
    details: dict = field(default_factory=dict)


# ============================================================
# Symbolic Epistemic Engine
# ============================================================

class EpistemicEngine:
    """
    The symbolic engine that maintains the epistemic model.
    Implements the formal framework of the paper.
    """

    def __init__(self, ablate_del_awareness: bool = False):
        self.observations: dict[str, Observation] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.awareness: set[str] = set()
        self.dependencies: dict[str, set[str]] = {}  # Dep: hyp_id → {assumption_ids}
        self._dependents: dict[str, set[str]] = {}   # reverse index, maintained by set_dependencies
        self.open_questions: list[dict] = []
        self.history: list[EngineEvent] = []
        self.current_turn: int = 0
        # Ablation: when True, the DEL plausibility layer
        # (soft upgrades of the `plausibility` field) and the awareness layer
        # (awareness-set membership, Expand-Awareness) are disabled. What
        # remains is the argument/dependency skeleton: hypothesis nodes,
        # the status lifecycle (active/weakened/abandoned/resolved),
        # the dependency map Dep(), and attack/undermine edges.
        self.ablate_del_awareness = ablate_del_awareness
        # Acyclicity guard. Default
        # off: with check_acyclicity=False the update path is unchanged
        # (a single boolean test per edge-adding op). When on, every
        # edge-adding operation re-runs cycle detection and, if a cycle
        # appears, appends a flag to self.cycle_flags identifying the
        # affected subgraph so a deployment can recompute it offline.
        self.check_acyclicity: bool = False
        self.cycle_flags: list[dict] = []

    # --------------------------------------------------------
    # Epistemic Operations
    # --------------------------------------------------------

    def observe(self, obs_id: str, content: str, turn: str, speaker: str):
        """Record a factual observation. Eliminates inconsistent worlds."""
        self.observations[obs_id] = Observation(obs_id, content, turn, speaker)
        if not self.ablate_del_awareness:
            self.awareness.add(obs_id)
        self.history.append(EngineEvent("Observe", turn, {"id": obs_id, "content": content}))

    def hypothesize(self, hyp_id: str, content: str, turn: str, speaker: str,
                    explains: list[str] = None, depends_on: list[str] = None):
        """
        Propose a new hypothesis. Enters as belief (not knowledge).
        Implements plausibility upgrade from Definition 2.
        """
        explains = explains or []
        depends_on = depends_on or []

        self.hypotheses[hyp_id] = Hypothesis(
            id=hyp_id, content=content, turn=turn, speaker=speaker,
            explains=explains
        )
        if not self.ablate_del_awareness:
            self.awareness.add(hyp_id)

        if depends_on:
            self.set_dependencies(hyp_id, depends_on)

        # Link observations to their explanatory hypothesis
        for obs_id in explains:
            if obs_id in self.observations:
                self.observations[obs_id].explained_by = hyp_id

        self.history.append(EngineEvent("Hypothesize", turn, {
            "id": hyp_id, "content": content, "explains": explains, "depends_on": depends_on
        }))
        self._check_cycles_if_enabled("Hypothesize", turn)

    def support(self, hyp_id: str, evidence: str, turn: str, speaker: str):
        """Increase plausibility of an existing hypothesis."""
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.supports.append({"evidence": evidence, "turn": turn, "speaker": speaker})
            if h.status == HypothesisStatus.ACTIVE and not self.ablate_del_awareness:
                h.plausibility = "high"
        self.history.append(EngineEvent("Support", turn, {"target": hyp_id, "evidence": evidence}))

    def undermine(self, hyp_id: str, evidence: str, turn: str, speaker: str):
        """
        Decrease plausibility of an existing hypothesis.
        The hypothesis becomes less plausible but is not abandoned.
        """
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.undermines.append({"evidence": evidence, "turn": turn, "speaker": speaker})
            if h.status == HypothesisStatus.ACTIVE:
                h.status = HypothesisStatus.WEAKENED
                if not self.ablate_del_awareness:
                    h.plausibility = "low"
        self.history.append(EngineEvent("Undermine", turn, {"target": hyp_id, "evidence": evidence}))
        self._check_cycles_if_enabled("Undermine", turn)

    def revise(self, hyp_id: str, reason: str, turn: str, speaker: str):
        """
        Abandon a hypothesis entirely. This is a categorical change,
        not just weakening. Implements radical upgrade of ¬h.
        """
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.status = HypothesisStatus.ABANDONED
            if not self.ablate_del_awareness:
                h.plausibility = "none"
            h.abandoned_reason = reason
            h.abandoned_turn = turn
            # Un-explain observations that were explained by this hypothesis
            for obs_id in h.explains:
                if obs_id in self.observations and self.observations[obs_id].explained_by == hyp_id:
                    self.observations[obs_id].explained_by = None
        self.history.append(EngineEvent("Revise", turn, {"target": hyp_id, "reason": reason}))

    def expand_awareness(self, prop_id: str, content: str, turn: str, speaker: str):
        """
        Add a previously unconceived proposition to the awareness set.
        Implements awareness expansion from Definition 3.

        Under ablate_del_awareness this operation is a no-op: the awareness
        layer does not exist in the ablated system.
        """
        if self.ablate_del_awareness:
            return
        self.awareness.add(prop_id)
        self.history.append(EngineEvent("Expand-Awareness", turn, {"id": prop_id, "content": content}))

    def resolve(self, hyp_id: str, turn: str, speaker: str):
        """Elevate a hypothesis from belief to knowledge (accepted conclusion)."""
        if hyp_id in self.hypotheses:
            h = self.hypotheses[hyp_id]
            h.status = HypothesisStatus.RESOLVED
            if not self.ablate_del_awareness:
                h.plausibility = "certain"
        self.history.append(EngineEvent("Resolve", turn, {"target": hyp_id}))

    def question(self, content: str, turn: str, speaker: str):
        """Record an open question."""
        self.open_questions.append({"content": content, "turn": turn, "speaker": speaker, "resolved": False})
        self.history.append(EngineEvent("Question", turn, {"content": content}))

    # --------------------------------------------------------
    # Dependency Tracking (dependency soundness)
    # --------------------------------------------------------

    def get_affected(self, assumption_id: str) -> list[str]:
        """
        Compute Affected(p) = {a ∈ Args : p ∈ Dep(a)}.
        Returns all hypotheses whose justification depends on the given assumption.
        """
        affected = []
        for hyp_id, deps in self.dependencies.items():
            if assumption_id in deps:
                affected.append(hyp_id)
        return affected

    def set_dependencies(self, hyp_id: str, deps) -> None:
        """Write Dep(hyp_id) and keep the reverse index consistent. All Dep writes go through here."""
        for d in self.dependencies.get(hyp_id, ()):
            self._dependents.get(d, set()).discard(hyp_id)
        self.dependencies[hyp_id] = set(deps)
        for d in deps:
            self._dependents.setdefault(d, set()).add(hyp_id)

    def remove_dependencies(self, hyp_id: str) -> None:
        """Drop Dep(hyp_id) and its reverse-index entries (used when an argument retires)."""
        for d in self.dependencies.pop(hyp_id, set()):
            self._dependents.get(d, set()).discard(hyp_id)

    def get_affected_closure(self, assumption_id: str) -> list[str]:
        """
        Affected*(p): least fixpoint of Affected (Corollary dep-closure).
        BFS over the maintained reverse index: O(|Affected*| + out-edges).
        """
        seen, frontier = set(), [assumption_id]
        while frontier:
            q = frontier.pop()
            for a in self._dependents.get(q, ()):
                if a not in seen:
                    seen.add(a); frontier.append(a)
        return sorted(seen)

    def retract_assumption(self, assumption_id: str, transitive: bool = True) -> dict:
        """
        Simulate retracting an assumption and compute the impact.
        S' = S \ Affected(p) is conflict-free (dependency soundness).

        Returns a structured report of what changes and what doesn't.
        """
        affected = (self.get_affected_closure(assumption_id) if transitive
                    else self.get_affected(assumption_id))

        result = {
            "retracted": assumption_id,
            "affected": [],
            "unaffected": [],
        }

        for hyp_id, h in self.hypotheses.items():
            entry = {
                "id": hyp_id,
                "content": h.content,
                "status": h.status.value,
                "depends_on": list(self.dependencies.get(hyp_id, set())),
            }
            if hyp_id in affected:
                entry["reason"] = f"Depends on retracted assumption: {assumption_id}"
                result["affected"].append(entry)
            else:
                result["unaffected"].append(entry)

        return result

    # --------------------------------------------------------
    # Cycle Detection (acyclicity guard)
    # --------------------------------------------------------

    def attack_edge_view(self) -> list[tuple[str, str]]:
        """
        Generalized attack-edge view: one (source, target) edge per stored
        undermine record. In the current schema the source is always an
        evidence string (evidence -> hypothesis), which cannot form cycles;
        a record may optionally carry an explicit 'source' node id
        (e.g. another hypothesis), which is the generalized case this
        detector guards against.
        """
        edges = []
        for hyp_id, h in self.hypotheses.items():
            for u in h.undermines:
                src = u.get("source") or f"ev:{u['evidence'][:30]}"
                edges.append((src, hyp_id))
        return edges

    def dep_edge_view(self) -> list[tuple[str, str]]:
        """Dependency edges: (hypothesis, assumption) per Dep() entry."""
        return [(hyp_id, d)
                for hyp_id, deps in self.dependencies.items()
                for d in deps]

    @staticmethod
    def _find_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
        """
        Iterative DFS (three-color) over a directed edge list.
        Returns one cycle (as a node path closing on itself) per back
        edge found; empty list iff the graph is acyclic.
        """
        graph: dict[str, list[str]] = {}
        nodes: set[str] = set()
        for s, d in edges:
            graph.setdefault(s, []).append(d)
            nodes.add(s)
            nodes.add(d)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        cycles: list[list[str]] = []
        for root in nodes:
            if color[root] != WHITE:
                continue
            color[root] = GRAY
            stack = [(root, iter(graph.get(root, ())))]
            path = [root]
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if color[nxt] == GRAY:
                        i = path.index(nxt)
                        cycles.append(path[i:] + [nxt])
                    elif color[nxt] == WHITE:
                        color[nxt] = GRAY
                        stack.append((nxt, iter(graph.get(nxt, ()))))
                        path.append(nxt)
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    path.pop()
                    color[node] = BLACK
        return cycles

    def detect_attack_cycles(self) -> list[list[str]]:
        """Cycles in the generalized attack graph (empty iff acyclic)."""
        return self._find_cycles(self.attack_edge_view())

    def detect_dep_cycles(self) -> list[list[str]]:
        """Cycles in the dependency map Dep() (empty iff acyclic)."""
        return self._find_cycles(self.dep_edge_view())

    def _check_cycles_if_enabled(self, operation: str, turn: str):
        """Optional per-update acyclicity check. No-op when the flag is off."""
        if not self.check_acyclicity:
            return
        att = self.detect_attack_cycles()
        dep = self.detect_dep_cycles()
        if att or dep:
            self.cycle_flags.append({
                "turn": turn, "operation": operation,
                "attack_cycles": att, "dep_cycles": dep,
            })
            print(f"  ⚠ ENGINE: acyclicity violated after {operation} at {turn}: "
                  f"{len(att)} attack cycle(s), {len(dep)} dependency cycle(s) — "
                  f"affected subgraph flagged for offline recomputation")

    # --------------------------------------------------------
    # Consistency Checking
    # --------------------------------------------------------

    def check_consistency(self) -> list[dict]:
        """
        Run consistency checks on the current model state.
        The symbolic engine uses these to detect errors in LLM classification.
        """
        issues = []

        # Check: hypothesis undermined multiple times should be Revised
        for hyp_id, h in self.hypotheses.items():
            if h.status == HypothesisStatus.WEAKENED and len(h.undermines) >= 2:
                issues.append({
                    "type": "suggest_revise",
                    "hypothesis": hyp_id,
                    "message": f"{hyp_id} has been undermined {len(h.undermines)} times — "
                               f"consider upgrading to Revise (full abandonment)"
                })

        # Check: unexplained observations
        for obs_id, obs in self.observations.items():
            if obs.explained_by is None:
                # Check if any active hypothesis claims to explain it
                explained = False
                for h in self.hypotheses.values():
                    if h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.RESOLVED):
                        if obs_id in h.explains:
                            explained = True
                            break
                if not explained:
                    issues.append({
                        "type": "unexplained_observation",
                        "observation": obs_id,
                        "message": f"Observation {obs_id} ({obs.content}) has no active explanatory hypothesis"
                    })

        return issues

    # --------------------------------------------------------
    # State Summary
    # --------------------------------------------------------

    def get_state_summary(self) -> str:
        """Generate a human-readable summary of the current model state."""
        lines = [f"=== ENGINE STATE (after {self.current_turn} turns) ===\n"]

        # Hypotheses by status
        for status in [HypothesisStatus.ACTIVE, HypothesisStatus.WEAKENED,
                       HypothesisStatus.RESOLVED, HypothesisStatus.ABANDONED]:
            hyps = [(id, h) for id, h in self.hypotheses.items() if h.status == status]
            if hyps:
                lines.append(f"  {status.value.upper()} hypotheses:")
                for id, h in hyps:
                    lines.append(f"    {id}: {h.content}")
                    if h.undermines:
                        lines.append(f"      Undermined by: {'; '.join(u['evidence'] for u in h.undermines)}")
                    if h.abandoned_reason:
                        lines.append(f"      Abandoned (T{h.abandoned_turn}): {h.abandoned_reason}")
                    deps = self.dependencies.get(id)
                    if deps:
                        lines.append(f"      Dep({id}) = {{{', '.join(deps)}}}")

        # Unexplained observations
        unexplained = [(id, o) for id, o in self.observations.items() if o.explained_by is None]
        if unexplained:
            lines.append(f"\n  UNEXPLAINED observations:")
            for id, o in unexplained:
                lines.append(f"    {id}: {o.content}")

        # Open questions
        open_qs = [q for q in self.open_questions if not q["resolved"]]
        if open_qs:
            lines.append(f"\n  OPEN questions:")
            for q in open_qs:
                lines.append(f"    ({q['turn']}) {q['content']}")

        # Awareness (omitted entirely under the del_awareness ablation)
        if not self.ablate_del_awareness:
            lines.append(f"\n  Awareness set ({len(self.awareness)} items): {{{', '.join(sorted(self.awareness))}}}")

        return "\n".join(lines)


# ============================================================
# Phase 2 Scenario
# ============================================================

# Module-level Phase 2 turn list, referenced by run_phase2().
# Each tuple is (turn_id, speaker, text, apply_fn). apply_fn(engine) mutates
# the engine in place; its return value (a list of Nones) is discarded.
PHASE2_TURNS = [
        ("T1", "Carol", "P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation.",
         lambda e: [
             e.observe("o1", "auth failure rate elevated", "T1", "Carol"),
             e.observe("o2", "payment failure rate elevated", "T1", "Carol"),
             e.observe("o3", "database health degradation alert", "T1", "Carol"),
             e.observe("o4", "customer complaints from ~2:15am", "T1", "Carol"),
             e.question("What is causing the failures?", "T1", "Carol"),
         ]),
        ("T2", "Alice", "401 Unauthorized spikes. Tokens rejected as expired. Auth traffic 3x normal. Strange.",
         lambda e: [
             e.observe("o5", "401 'token expired' errors from ~2am", "T2", "Alice"),
             e.observe("o6", "auth traffic 3x normal at 2am", "T2", "Alice"),
             e.question("Why is auth traffic 3x at 2am?", "T2", "Alice"),
         ]),
        ("T3", "Bob", "Stripe returning 429s. Redis connection timeouts from Payment Service.",
         lambda e: [
             e.observe("o7", "Stripe 429 rate-limit errors", "T3", "Bob"),
             e.observe("o8", "Redis connection timeouts from Payment", "T3", "Bob"),
         ]),
        ("T4", "Carol", "Database alert is about Redis, not the primary DB.",
         lambda e: [
             e.expand_awareness("mis_monitor", "monitoring miscategorises Redis as 'database health'", "T4", "Carol"),
             e.question("Redis issues from Payment side?", "T4", "Carol"),
         ]),
        ("T5", "Bob", "Connection pool exhausted. Rate-limit checks failing. Requests go to Stripe unthrottled.",
         lambda e: [
             e.observe("o8b", "Redis connection pool exhausted", "T5", "Bob"),
             e.hypothesize("h1", "Redis pool exhaustion → rate-limit bypass → Stripe 429s", "T5", "Bob",
                           explains=["o7", "o8"], depends_on=["o8"]),
         ]),
        ("T6", "Carol", "Redis sick → Payment loses rate limiting → Stripe hammered. Redis → Auth too?",
         lambda e: [
             e.support("h1", "Carol restates the chain", "T6", "Carol"),
             e.hypothesize("h2", "Redis failure → Auth failures (via shared session cache)", "T6", "Carol",
                           explains=["o1", "o5"], depends_on=["o8"]),
             e.question("Does Auth use Redis?", "T6", "Carol"),
         ]),
        ("T7", "Alice", "Auth uses Redis for sessions. But error is 'token expired' (401), not 'session lookup failed' (503).",
         lambda e: [
             e.observe("o9", "error code is 'token expired' (401), NOT 'session lookup failed' (503)", "T7", "Alice"),
             e.undermine("h2", "error code inconsistent: h2 predicts 503, observed 401", "T7", "Alice"),
         ]),
        ("T8", "Carol", "Auth failures might not be caused by Redis after all?",
         lambda e: [
             e.question("Are auth failures caused by Redis?", "T8", "Carol"),
         ]),
        ("T9", "Alice", "Error type is wrong. Token expiry is a different problem from Redis.",
         lambda e: [
             e.revise("h2", "error code proves auth failures are independent of Redis", "T9", "Alice"),
         ]),
        ("T10", "Bob", "Why is auth traffic 3x if token expiry is independent of Redis?",
         lambda e: [
             e.question("Why is auth traffic 3x if not caused by Redis?", "T10", "Bob"),
         ]),
        ("T11", "Alice", "Frontend retrying on token expired. Retry loop generates amplified traffic.",
         lambda e: [
             e.hypothesize("h3", "token bug → frontend retry loop → 3x traffic amplification", "T11", "Alice",
                           explains=["o6"], depends_on=["o5"]),
         ]),
        ("T12", "Carol", "Retry storm from token bug exhausting Redis? Auth→Redis, not Redis→Auth?",
         lambda e: [
             e.hypothesize("h4", "token bug → retry storm → Redis pool exhaustion (CAUSAL REVERSAL)", "T12", "Carol",
                           explains=["o8"], depends_on=["h3", "o6"]),
         ]),
        ("T13", "Alice", "Plausible. 3x traffic × Redis lookups = pool exhaustion. Whole thing cascades from token bug.",
         lambda e: [
             e.support("h4", "mechanistic argument: 3x traffic × Redis lookups = pool exhaustion", "T13", "Alice"),
             e.resolve("h4", "T13", "Alice"),
             e.resolve("h1", "T13", "Alice"),
             e.resolve("h3", "T13", "Alice"),
             # Set dependencies for the unified chain
             e.set_dependencies("h4", {"o9", "o6", "h3"}),
             e.set_dependencies("h1", {"o8", "h4"}),
             e.set_dependencies("unified", {"o9", "o6", "o8", "h3", "h4"}),
         ]),
]


def run_phase2(engine: EpistemicEngine, verbose: bool = True):
    """
    Run the Phase 2 debugging scenario through the engine.
    Each turn applies the ground-truth epistemic operations.
    """
    for i, (turn_id, speaker, text, apply_fn) in enumerate(PHASE2_TURNS):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  {turn_id} — {speaker}: {text}")
        apply_fn(engine)
        engine.current_turn = i + 1

        # Run consistency checks
        issues = engine.check_consistency()
        if verbose and issues:
            for issue in issues:
                print(f"  ⚠ ENGINE: {issue['message']}")

    if verbose:
        print(f"\n{'='*60}")
        print(engine.get_state_summary())


def run_counterfactuals(engine: EpistemicEngine):
    """
    Demonstrate counterfactual reasoning via Dep() and Affected().
    This is the capability that requires the symbolic engine.
    """
    scenarios = [
        ("o9", "Retract error code evidence (o9)",
         "What if Alice's error code observation were wrong?"),
        ("o8", "Retract Redis timeout observation (o8)",
         "What if Bob's Redis timeouts were a misreading?"),
        ("o6", "Retract 3x traffic observation (o6)",
         "What if the 3x traffic reading were a monitoring glitch?"),
        ("o5", "Retract auth-failure-pattern observation (o5)",
         "What if the auth failure pattern (o5) were an artefact? (2-hop: o5 -> h3 -> h4)"),
    ]

    print(f"\n{'='*60}")
    print("COUNTERFACTUAL ANALYSIS")
    print(f"{'='*60}")

    for assumption_id, label, question in scenarios:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"  Q: {question}")

        result = engine.retract_assumption(assumption_id)

        print(f"\n  Affected({assumption_id}) = {{{', '.join(result['affected_ids'] if 'affected_ids' in result else [h['id'] for h in result['affected']])}}}")

        if result["affected"]:
            print(f"\n  AFFECTED (flagged for re-evaluation):")
            for h in result["affected"]:
                print(f"    {h['id']} [{h['status']}]: {h['content']}")
                print(f"      Dep = {{{', '.join(h['depends_on'])}}}")
                print(f"      → {h['reason']}")

        if result["unaffected"]:
            print(f"\n  UNAFFECTED (remain valid):")
            for h in result["unaffected"]:
                print(f"    {h['id']} [{h['status']}]: {h['content']}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    engine = EpistemicEngine()
    run_phase2(engine, verbose=True)

    if "--counterfactual" in sys.argv or True:  # Always run counterfactuals in demo
        run_counterfactuals(engine)

    # Print dependency map
    print(f"\n{'='*60}")
    print("DEPENDENCY MAP")
    print(f"{'='*60}")
    for hyp_id, deps in engine.dependencies.items():
        affected = engine.get_affected(list(deps)[0]) if deps else []
        print(f"  Dep({hyp_id}) = {{{', '.join(sorted(deps))}}}")

    print(f"\n{'='*60}")
    print("ENGINE STATISTICS")
    print(f"{'='*60}")
    print(f"  Observations: {len(engine.observations)}")
    print(f"  Hypotheses: {len(engine.hypotheses)}")
    print(f"    Active: {sum(1 for h in engine.hypotheses.values() if h.status == HypothesisStatus.ACTIVE)}")
    print(f"    Weakened: {sum(1 for h in engine.hypotheses.values() if h.status == HypothesisStatus.WEAKENED)}")
    print(f"    Abandoned: {sum(1 for h in engine.hypotheses.values() if h.status == HypothesisStatus.ABANDONED)}")
    print(f"    Resolved: {sum(1 for h in engine.hypotheses.values() if h.status == HypothesisStatus.RESOLVED)}")
    print(f"  Awareness set: {len(engine.awareness)} propositions")
    print(f"  Dependencies tracked: {len(engine.dependencies)}")
    print(f"  Operations applied: {len(engine.history)}")
