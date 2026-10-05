#!/usr/bin/env python3
"""
Retraction latency microbenchmark

Empirically validates the complexity bound's claim that the verifier's
Affected(p) operation scales with |Args_t| (the active dependency-graph size),
while a baseline that re-reads history scales with K (the number of turns).

Two scaling regimes:

  (1) "naive": |Args_t| grows linearly with K. This is the worst case the
      paper's analytical bound technically allows.

  (2) "bounded": |Args_t| saturates at a constant ARG_CAP, modelling the
      Phase-2-like deployment regime where hypotheses are resolved/abandoned
      and only a bounded working set remains active. This is the regime our
      scenarios actually satisfy (Phase 2: |Args|=4 at K=13).

Two cost metrics per regime:

  - Operation count: primitive comparisons, hardware-independent
  - Wall time: median over n_queries on this machine

This script is fully model-free; no LLM is required.

Usage:
    python experiments/scripts/run_retraction_latency.py --output retraction_latency.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import os as _os
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
from symbolic_engine import EpistemicEngine, EngineEvent  # noqa: E402


# ============================================================
# Synthetic Generators
# ============================================================

def build_engine_naive(K: int, seed: int = 0) -> EpistemicEngine:
    """Naive regime: |Args| ~ 0.3 * K (no resolution/abandonment)."""
    rng = random.Random(seed)
    engine = EpistemicEngine()

    n_obs = max(1, int(K * 0.50))
    n_hyp = max(1, int(K * 0.30))

    for i in range(1, n_obs + 1):
        engine.observe(f"o{i}", f"observation {i}", f"T{i}", "A")

    obs_ids = list(engine.observations.keys())
    prior_hyps: list[str] = []

    for j in range(1, n_hyp + 1):
        n_deps = rng.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]
        deps: list[str] = []
        if rng.random() < 0.15 and prior_hyps:
            deps.append(rng.choice(prior_hyps))
            n_deps -= 1
        for _ in range(n_deps):
            if obs_ids:
                deps.append(rng.choice(obs_ids))
        deps = list(dict.fromkeys(deps))

        hid = f"h{j}"
        engine.hypothesize(
            hyp_id=hid, content=f"hypothesis {j}",
            turn=f"T{n_obs + j}", speaker="B",
            explains=[rng.choice(obs_ids)] if obs_ids else [],
            depends_on=deps,
        )
        prior_hyps.append(hid)

    while len(engine.history) < K:
        engine.history.append(EngineEvent(
            operation="Question",
            turn=f"T{len(engine.history) + 1}",
            details={"content": f"q{len(engine.history)}"},
        ))

    return engine


def build_engine_bounded(K: int, seed: int = 0, arg_cap: int = 50) -> EpistemicEngine:
    """
    Bounded regime: |Args| saturates at arg_cap.

    Models real deployment: as new hypotheses are added past the cap,
    older ones are evicted from the active dependency map (representing
    Resolved/Abandoned hypotheses that no longer participate in
    grounding queries). History still grows linearly.
    """
    rng = random.Random(seed)
    engine = EpistemicEngine()

    n_obs = max(1, int(K * 0.50))
    n_hyp = max(1, int(K * 0.30))

    for i in range(1, n_obs + 1):
        engine.observe(f"o{i}", f"observation {i}", f"T{i}", "A")

    obs_ids = list(engine.observations.keys())
    active_hyps: list[str] = []  # FIFO order

    for j in range(1, n_hyp + 1):
        n_deps = rng.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]
        deps: list[str] = []
        if rng.random() < 0.15 and active_hyps:
            deps.append(rng.choice(active_hyps))
            n_deps -= 1
        for _ in range(n_deps):
            if obs_ids:
                deps.append(rng.choice(obs_ids))
        deps = list(dict.fromkeys(deps))

        hid = f"h{j}"
        engine.hypothesize(
            hyp_id=hid, content=f"hypothesis {j}",
            turn=f"T{n_obs + j}", speaker="B",
            explains=[rng.choice(obs_ids)] if obs_ids else [],
            depends_on=deps,
        )
        active_hyps.append(hid)

        while len(active_hyps) > arg_cap:
            evicted = active_hyps.pop(0)
            engine.remove_dependencies(evicted)

    while len(engine.history) < K:
        engine.history.append(EngineEvent(
            operation="Question",
            turn=f"T{len(engine.history) + 1}",
            details={"content": f"q{len(engine.history)}"},
        ))

    return engine


# ============================================================
# Algorithms (counted + plain timing variants)
# ============================================================

def _closure_from_index(index: dict, p: str) -> tuple[list[str], int]:
    """Affected*(p) by BFS over a dependents index; ops = nodes visited + edges scanned."""
    seen, frontier, ops = set(), [p], 0
    while frontier:
        q = frontier.pop(); ops += 1
        for a in index.get(q, ()):
            ops += 1
            if a not in seen:
                seen.add(a); frontier.append(a)
    return sorted(seen), ops


def verifier_affected_counted(engine: EpistemicEngine, p: str) -> tuple[list[str], int]:
    # Affected*: BFS over the engine's maintained reverse index (Cor. dep-closure)
    return _closure_from_index(engine._dependents, p)


def baseline_affected_counted(engine: EpistemicEngine, p: str,
                              tokens_per_turn: int = 50) -> tuple[list[str], int]:
    """
    Baseline: walk the K-turn history, reconstructing Dep(α) for each α
    encountered. Each turn pays tokens_per_turn ops (proxy for re-reading).
    """
    ops = 0
    reconstructed: dict[str, set[str]] = {}
    for ev in engine.history:
        ops += tokens_per_turn
        if ev.operation == "Hypothesize":
            hid = ev.details.get("id")
            if hid is not None:
                deps = ev.details.get("depends_on", []) or []
                reconstructed[hid] = set(deps)
    index: dict[str, set[str]] = {}
    for hid, deps in reconstructed.items():
        for d in deps:
            index.setdefault(d, set()).add(hid)
    affected, cops = _closure_from_index(index, p)
    return affected, ops + cops


def verifier_affected(engine: EpistemicEngine, p: str) -> list[str]:
    return engine.get_affected_closure(p)


def baseline_affected(engine: EpistemicEngine, p: str) -> list[str]:
    reconstructed: dict[str, set[str]] = {}
    for ev in engine.history:
        if ev.operation == "Hypothesize":
            hid = ev.details.get("id")
            if hid is not None:
                deps = ev.details.get("depends_on", []) or []
                reconstructed[hid] = set(deps)
    index: dict[str, set[str]] = {}
    for hid, deps in reconstructed.items():
        for d in deps:
            index.setdefault(d, set()).add(hid)
    return _closure_from_index(index, p)[0]


# ============================================================
# Driver
# ============================================================

def benchmark_one(engine: EpistemicEngine, n_queries: int, seed: int) -> dict:
    pool = list(engine.observations.keys()) + list(engine.hypotheses.keys())
    rng = random.Random(seed + 1)
    queries = [rng.choice(pool) for _ in range(n_queries)]

    v_ops_list, b_ops_list = [], []
    for p in queries:
        _, vops = verifier_affected_counted(engine, p)
        _, bops = baseline_affected_counted(engine, p, tokens_per_turn=50)
        v_ops_list.append(vops)
        b_ops_list.append(bops)

    v_times, b_times = [], []
    for p in queries:
        t0 = time.perf_counter_ns()
        _ = verifier_affected(engine, p)
        v_times.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        _ = baseline_affected(engine, p)
        b_times.append(time.perf_counter_ns() - t0)

    return {
        "n_args": len(engine.dependencies),
        "n_history": len(engine.history),
        "verifier_ops_median": statistics.median(v_ops_list),
        "baseline_ops_median": statistics.median(b_ops_list),
        "verifier_us_median": statistics.median(v_times) / 1000.0,
        "baseline_us_median": statistics.median(b_times) / 1000.0,
    }


def run_regime(name: str, builder, Ks: list[int], n_queries: int,
               seeds: list[int]) -> list[dict]:
    print(f"\n=== Regime: {name} ===")
    print(f"  {'K':>5}  {'|Args|':>7}  {'|Hist|':>7}  "
          f"{'V_ops':>8}  {'B_ops':>8}  {'B/V_ops':>9}  "
          f"{'V_us':>7}  {'B_us':>8}  {'B/V_us':>8}")

    out = []
    for K in Ks:
        per_seed = [benchmark_one(builder(K, seed=s), n_queries, s)
                    for s in seeds]

        v_ops = [r["verifier_ops_median"] for r in per_seed]
        b_ops = [r["baseline_ops_median"] for r in per_seed]
        v_us = [r["verifier_us_median"] for r in per_seed]
        b_us = [r["baseline_us_median"] for r in per_seed]

        agg = {
            "regime": name,
            "K": K,
            "n_args_avg": statistics.mean(r["n_args"] for r in per_seed),
            "n_history_avg": statistics.mean(r["n_history"] for r in per_seed),
            "verifier_ops": statistics.median(v_ops),
            "baseline_ops": statistics.median(b_ops),
            "verifier_us": statistics.median(v_us),
            "baseline_us": statistics.median(b_us),
            "per_seed": per_seed,
        }
        agg["ops_speedup"] = agg["baseline_ops"] / max(1, agg["verifier_ops"])
        agg["us_speedup"] = agg["baseline_us"] / max(0.001, agg["verifier_us"])

        print(f"  {K:>5}  {agg['n_args_avg']:>7.0f}  {agg['n_history_avg']:>7.0f}  "
              f"{agg['verifier_ops']:>8.0f}  {agg['baseline_ops']:>8.0f}  "
              f"{agg['ops_speedup']:>8.1f}x  "
              f"{agg['verifier_us']:>6.2f}  {agg['baseline_us']:>7.2f}  "
              f"{agg['us_speedup']:>7.1f}x")

        out.append(agg)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="retraction_latency.json")
    parser.add_argument("--Ks", default="13,26,50,100,200,500,1000,2000")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--arg-cap", type=int, default=50)
    args = parser.parse_args()

    Ks = [int(x) for x in args.Ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print("Retraction latency microbenchmark")
    print(f"  Ks = {Ks}")
    print(f"  seeds = {seeds}")
    print(f"  n_queries per (K, seed) = {args.n_queries}")
    print(f"  arg_cap (bounded regime) = {args.arg_cap}")

    naive = run_regime(
        "naive",
        lambda K, seed: build_engine_naive(K, seed),
        Ks, args.n_queries, seeds,
    )

    bounded = run_regime(
        f"bounded (cap={args.arg_cap})",
        lambda K, seed: build_engine_bounded(K, seed, arg_cap=args.arg_cap),
        Ks, args.n_queries, seeds,
    )

    out = {
        "experiment": "Retraction latency microbenchmark",
        "Ks": Ks,
        "seeds": seeds,
        "n_queries_per_seed": args.n_queries,
        "arg_cap_bounded": args.arg_cap,
        "regimes": {
            "naive": naive,
            "bounded": bounded,
        },
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
