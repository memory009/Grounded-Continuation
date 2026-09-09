"""
run_e5.py — E5: Robustness of `Verify` to extraction noise.

Originally implemented per ./exp_robustness.md
(advisor spec). The literal spec (drop + add over H×O) produced a perfectly
flat curve at 0.98 across all 16 cells — see
experiments/results/e5_robustness/E5_EDGE_NOISE_DIAGNOSIS.md and the frozen
results_spec_literal.json. The flatness is structural: 88% of test items are
decided by mechanisms (status check, null-resolution, observation/awareness
leaves) that run BEFORE walk_deps, so dep-graph perturbation has no surface
area to act on.

This file is the extended version. It runs three
clearly-labeled noise models offline:

  drop    — for each existing dep edge, drop with probability ε.
            (Spec literal; recall errors. Acts on the 6 walk-deps items but
            cannot flip them since shrinking Dep cannot introduce abandoned
            hits in this universe.)

  add     — universe is H × (O ∪ H), excluding self-edges, the synthetic
            'unified' key, and entities not yet present at the engine state.
            (EXTENDED from spec's H × O. Allows precision errors that target
            abandoned hypotheses, which trigger walk_deps short-circuit on
            the 6 walk-deps items.)

  status  — flip each hypothesis's status across the good-standing boundary
            (active/resolved ↔ abandoned/weakened) with probability ε.
            (NEW. Captures missed Revise / Resolve / lifecycle-tracking
            extraction errors. Drives the 15 stale items + transitively
            affects the walk-deps items.)

Each noise model is independent — sweeps run in three separate top-level keys
of results.json. No combined noise.

Per-item truncation (not all-at-T13): preserves the test set's t-relative
semantics so ε=0 reproduces e2_verify's verifier baseline (49/50 = 0.98).

Zero API spend; canonical-GT engine + canonical asserts_id (no LLM resolution).

Usage:
  python experiments/e5_robustness/run_e5.py            # full run
  python experiments/e5_robustness/run_e5.py --smoke    # 1 seed × 3 noise levels (debug)
"""

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from symbolic_engine import EpistemicEngine, HypothesisStatus, PHASE2_TURNS  # noqa: E402
from verify_experiment import verifier_judgement  # noqa: E402

GOOD_STANDING = (HypothesisStatus.ACTIVE, HypothesisStatus.RESOLVED)
NOT_IN_GOOD_STANDING = (HypothesisStatus.ABANDONED, HypothesisStatus.WEAKENED)

TEST_SET_PATH = PROJECT_ROOT / "experiments" / "e2_verify" / "verify_test_set_phase2.yaml"
OUT_PATH = PROJECT_ROOT / "experiments" / "results" / "e5_robustness" / "results.json"
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
N_SEEDS = 10
CATEGORIES = ("actual", "stale", "cross_conv", "counterfactual")


# ----------------------------------------------------------------------------
# Engine construction
# ----------------------------------------------------------------------------

def build_engine_at_t(t: int) -> EpistemicEngine:
    """Apply PHASE2_TURNS[0:t] to a fresh engine. t is 1-indexed.
    For t==13 this includes the unification override (h1→{o8,h4} etc.)."""
    engine = EpistemicEngine()
    for i in range(min(t, len(PHASE2_TURNS))):
        _, _, _, apply_fn = PHASE2_TURNS[i]
        apply_fn(engine)
        engine.current_turn = i + 1
    return engine


# ----------------------------------------------------------------------------
# Noise universe and perturbations
# ----------------------------------------------------------------------------

def get_perturbable_deps(engine):
    """Return {hyp_id: {assumption_ids}} restricted to keys that are real
    hypotheses. Filters out the synthetic 'unified' key (set by T13 lambda
    but not in engine.hypotheses; perturbing it has no observable effect on
    Verify but creates noisy diagnostic output)."""
    return {k: set(v) for k, v in engine.dependencies.items()
            if k in engine.hypotheses}


def all_possible_edges(engine):
    """Universe of candidate edges for the `add` noise model.

    Extended from the spec's H × O to H × (O ∪ H). Excludes:
      - self-edges (h, h) — extraction would never produce these
      - the synthetic 'unified' key — not in engine.hypotheses (auto-excluded
        as a SOURCE; targets restricted to engine.hypotheses keys exclude it)
      - entities not yet present at the engine state — auto-excluded because
        engine_at_t only contains entities created by turns 1..t

    Why extend: the original spec's H × O universe could not flip any
    walk-deps items (every added edge resolved to a leaf observation). H × H
    edges allow precision errors to target abandoned hypotheses, which
    trigger walk_deps short-circuit. See E5_EDGE_NOISE_DIAGNOSIS.md."""
    hyp_ids = list(engine.hypotheses.keys())
    obs_ids = list(engine.observations.keys())
    targets = obs_ids + hyp_ids
    return {(h, t) for h in hyp_ids for t in targets if h != t}


def perturb_drop(deps_gt, eps, rng):
    """For each existing edge in Dep_t, drop with probability eps.

    Models LLM extraction recall errors. Spec literal."""
    return {h: {a for a in asns if rng.random() >= eps}
            for h, asns in deps_gt.items()}


def perturb_add(deps_gt, all_edges, eps, rng):
    """Add non-existent edges, calibrated so E[added] ≈ eps × n_existing.

    Universe per `all_possible_edges` (H × (O ∪ H), no self-edges, no
    synthetic keys, no future entities). Models LLM extraction precision
    errors — including (post-extension) wrong hypothesis-to-hypothesis
    edges, which can target abandoned hypotheses and trigger walk_deps
    short-circuit on active-hyp items.

    On small graphs (low t with few existing edges), p_add becomes very small
    and the add-noise effect is weak. This is correct behaviour (precision
    errors scale with extraction volume) but means the `add` curve mainly
    contributes through high-t items."""
    n_existing = sum(len(s) for s in deps_gt.values())
    existing_edges = {(h, a) for h, asns in deps_gt.items() for a in asns}
    candidate_edges = all_edges - existing_edges
    out = {h: set(asns) for h, asns in deps_gt.items()}
    if not candidate_edges or n_existing == 0:
        return out
    p_add = min((eps * n_existing) / len(candidate_edges), 1.0)
    for (h, a) in candidate_edges:
        if rng.random() < p_add:
            out.setdefault(h, set()).add(a)
    return out


def perturb_status(engine, eps, rng):
    """In-place: for each hypothesis, with probability eps, flip its status
    across the good-standing boundary.

    Third noise model. Captures LLM
    extraction errors in Revise / Resolve / lifecycle-transition tracking.

    Mapping:
      ACTIVE   ↔ ABANDONED   (missed Revise / extra Revise)
      RESOLVED ↔ WEAKENED    (missed Resolve / spurious Undermine)

    The mapping crosses the good-standing/not-good-standing boundary, so
    walk_deps's status short-circuit is exercised on every flip.
    Caller must deepcopy the engine first; this mutates h.status."""
    for h in engine.hypotheses.values():
        if rng.random() >= eps:
            continue
        if h.status == HypothesisStatus.ACTIVE:
            h.status = HypothesisStatus.ABANDONED
        elif h.status == HypothesisStatus.ABANDONED:
            h.status = HypothesisStatus.ACTIVE
        elif h.status == HypothesisStatus.RESOLVED:
            h.status = HypothesisStatus.WEAKENED
        elif h.status == HypothesisStatus.WEAKENED:
            h.status = HypothesisStatus.RESOLVED


# ----------------------------------------------------------------------------
# Canonical-id resolution (GT engine, no LLM)
# ----------------------------------------------------------------------------

def resolve_canonical_gt(canonical_id, engine):
    """Map a canonical asserts_id to (id_in_engine, kind) for verifier_judgement.

    Because the E5 engine is the canonical GT engine (not an LLM-extracted one),
    canonical IDs equal LLM IDs and no content alignment is needed.

    Returns (None, None) for a null asserts_id (multi-entity / meta items in the
    test set; verifier_judgement returns 'ungrounded' for these). For canonical
    IDs not yet present in D_t at this item's truncation_t, returns (None, kind)
    so the verifier emits 'ungrounded' with the correct entity-class reason."""
    if canonical_id is None:
        return None, None
    if canonical_id.startswith("h"):
        return (canonical_id if canonical_id in engine.hypotheses else None,
                "hypothesis")
    if canonical_id.startswith("o"):
        return (canonical_id if canonical_id in engine.observations else None,
                "observation")
    if canonical_id in engine.awareness:
        return canonical_id, "awareness"
    return None, "awareness"


# ----------------------------------------------------------------------------
# Per-(model, eps, seed) cell
# ----------------------------------------------------------------------------

def run_one_seed(noise_model, eps, seed, items, base_engines_by_t):
    """Run all items at their truncation_t against an engine perturbed
    under (noise_model, eps, seed).

    noise_model ∈ {'drop', 'add', 'status'}:
      drop, add  — perturb engine.dependencies (deps mutate; status preserved).
      status     — perturb engine.hypotheses[*].status (status mutates;
                   deps preserved exactly as GT)."""
    rng = random.Random(seed)

    # Build a perturbed engine per t. deepcopy preserves state and history.
    perturbed_by_t = {}
    for t, base in base_engines_by_t.items():
        eng = deepcopy(base)
        if noise_model == "drop":
            deps_gt = get_perturbable_deps(base)
            deps_noisy = perturb_drop(deps_gt, eps, rng)
            for k in list(eng.dependencies.keys()):
                if k in eng.hypotheses:
                    eng.dependencies[k] = deps_noisy.get(k, set())
            for k, v in deps_noisy.items():
                if k not in eng.dependencies:
                    eng.dependencies[k] = v
        elif noise_model == "add":
            deps_gt = get_perturbable_deps(base)
            edges = all_possible_edges(base)
            deps_noisy = perturb_add(deps_gt, edges, eps, rng)
            for k in list(eng.dependencies.keys()):
                if k in eng.hypotheses:
                    eng.dependencies[k] = deps_noisy.get(k, set())
            for k, v in deps_noisy.items():
                if k not in eng.dependencies:
                    eng.dependencies[k] = v
        elif noise_model == "status":
            perturb_status(eng, eps, rng)
        else:
            raise ValueError(noise_model)
        perturbed_by_t[t] = eng

    pooled_correct = pooled_total = 0
    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in items:
        t = item["truncation_t"]
        engine = perturbed_by_t[t]
        resolved_id, kind = resolve_canonical_gt(item.get("asserts_id"), engine)
        pred, _ = verifier_judgement(engine, resolved_id, kind)
        ok = int(pred == item["label"])
        pooled_correct += ok
        pooled_total += 1
        per_cat[item["category"]]["correct"] += ok
        per_cat[item["category"]]["total"] += 1

    return {
        "pooled_acc": pooled_correct / pooled_total if pooled_total else 0.0,
        "per_category_acc": {
            c: (d["correct"] / d["total"] if d["total"] else 0.0)
            for c, d in per_cat.items()
        },
        "n_correct": pooled_correct,
        "n_total": pooled_total,
        "n_correct_by_cat": {c: d["correct"] for c, d in per_cat.items()},
        "n_total_by_cat": {c: d["total"] for c, d in per_cat.items()},
    }


def aggregate_seeds(seed_results):
    """Median + p25 + p75 over per-seed pooled accuracy and per-category accuracy."""
    pooled = [r["pooled_acc"] for r in seed_results]
    cat_accs = defaultdict(list)
    for r in seed_results:
        for c, a in r["per_category_acc"].items():
            cat_accs[c].append(a)
    return {
        "median": float(np.median(pooled)),
        "p25": float(np.percentile(pooled, 25)),
        "p75": float(np.percentile(pooled, 75)),
        "all": [float(a) for a in pooled],
        "per_category": {
            c: {
                "median": float(np.median(vs)),
                "p25": float(np.percentile(vs, 25)),
                "p75": float(np.percentile(vs, 75)),
            }
            for c, vs in cat_accs.items()
        },
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Run a quick subset (1 seed × 3 ε levels) for debug.")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    test_set = yaml.safe_load(TEST_SET_PATH.read_text())
    items = test_set["items"]
    print(f"Loaded {len(items)} items from {TEST_SET_PATH}")

    unique_ts = sorted({it["truncation_t"] for it in items})
    print(f"Unique truncation_t values: {unique_ts}")

    base_engines_by_t = {t: build_engine_at_t(t) for t in unique_ts}
    for t in sorted(base_engines_by_t):
        e = base_engines_by_t[t]
        n_deps = sum(len(v) for k, v in e.dependencies.items() if k in e.hypotheses)
        print(f"  t={t:>2}: {len(e.observations):>2} obs, "
              f"{len(e.hypotheses):>1} hyps, {n_deps:>2} dep-edges")

    noise_levels = [0.00, 0.20, 0.60] if args.smoke else NOISE_LEVELS
    n_seeds = 1 if args.smoke else N_SEEDS

    results = {"drop": {}, "add": {}, "status": {}}
    for noise_model in ("drop", "add", "status"):
        print(f"\n[{noise_model} noise]")
        for eps in noise_levels:
            seed_results = [
                run_one_seed(noise_model, eps, seed, items, base_engines_by_t)
                for seed in range(n_seeds)
            ]
            cell = aggregate_seeds(seed_results)
            results[noise_model][f"{eps:.2f}"] = cell
            cat_str = " ".join(
                f"{c[:5]}={cell['per_category'][c]['median']:.2f}"
                for c in CATEGORIES if c in cell["per_category"]
            )
            print(f"  ε={eps:.2f}: median={cell['median']:.3f} "
                  f"[p25={cell['p25']:.3f}, p75={cell['p75']:.3f}]  {cat_str}")

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    out = {
        "experiment": "E5 — Robustness of Verify to extraction noise (extended)",
        "spec": "exp_robustness.md",
        "spec_deviations": [
            "add-edge universe extended from H×O to H×(O∪H), no self-edges, "
            "no synthetic 'unified' key, no future entities.",
            "status corruption added as a third noise model (NEW). With "
            "probability ε flips each hypothesis status across the "
            "good-standing boundary (active/resolved ↔ abandoned/weakened).",
            "per-item truncation (not all-at-T13): preserves t-relative "
            "test-set semantics so ε=0 reproduces e2_verify baseline.",
        ],
        "noise_models": ["drop", "add", "status"],
        "noise_levels": noise_levels,
        "n_seeds": n_seeds,
        "n_items": len(items),
        "categories": list(CATEGORIES),
        "git_sha": git_sha,
        "smoke": args.smoke,
        "drop": results["drop"],
        "add": results["add"],
        "status": results["status"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
