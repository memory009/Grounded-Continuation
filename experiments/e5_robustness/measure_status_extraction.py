"""
measure_status_extraction.py — Step 2 for E5: empirical lifecycle/status extraction error rate.

Reads existing GPT-4o pipeline outputs (no API calls) from the C3 +
e1b_ablation runs. For each run × canonical hypothesis (h1, h2, h3, h4):
  1. Content-align LLM-extracted hypothesis to canonical via the same
     token-overlap matcher that produced the published Table 2 numbers
     (eval_dep_extraction.token_overlap, threshold 0.15 from
     verify_experiment._best_content_match).
  2. Compare LLM final status to GT.
  3. Score on two metrics:
     - exact_status_match: LLM status equals GT status (active/resolved/
       weakened/abandoned).
     - boundary_match: LLM and GT agree on good-standing vs
       not-good-standing partition. This is the metric that maps directly
       to E5's status-corruption noise model (which flips across the
       boundary, not within).

Outputs:
  - JSON: experiments/results/e5_robustness/status_extraction_observed.json
  - Console summary by run, by transition, aggregate error rates.

Recommendation logic at the end: whether the aggregate boundary-error rate
is defensible as the "empirical operating point" on E5's status curve.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

import eval_dep_extraction as ev  # token_overlap

PIPELINE_DEP_DIR = PROJECT_ROOT / "experiments" / "results" / "pipeline_dep_extraction"
E1B_DIR = PROJECT_ROOT / "experiments" / "results" / "e1b_ablation"
OUT_PATH = PROJECT_ROOT / "experiments" / "results" / "e5_robustness" / "status_extraction_observed.json"

# GT canonical hypothesis content (mirrors eval_dep_extraction.GT_H_CONTENT
# and the test-set author's canonical statements).
GT_HYP = {
    "h1": "Redis pool exhaustion → rate-limit bypass → Stripe 429s",
    "h2": "Redis failure → Auth failures (via shared session cache)",
    "h3": "token bug → frontend retry loop → 3x traffic amplification",
    "h4": "token bug → retry storm → Redis pool exhaustion (CAUSAL REVERSAL)",
}

# Ground-truth status at end of Phase 2 (post-T13).
# Source: symbolic_engine.run_phase2 / PHASE2_TURNS final state.
GT_STATUS = {
    "h1": "resolved",
    "h2": "abandoned",
    "h3": "resolved",
    "h4": "resolved",
}

GOOD_STANDING = {"active", "resolved"}
NOT_GOOD_STANDING = {"abandoned", "weakened"}


def boundary(status: str) -> str:
    if status in GOOD_STANDING:
        return "good"
    if status in NOT_GOOD_STANDING:
        return "not_good"
    return "unknown"


def best_match(target_text: str, candidates: dict, min_overlap: float = 0.15):
    """Many-to-one content match — same logic as verify_experiment._best_content_match."""
    best_id, best_score = None, 0.0
    for cid, ctext in candidates.items():
        score = ev.token_overlap(target_text, ctext)
        if score > best_score:
            best_score, best_id = score, cid
    if best_score < min_overlap:
        return None, best_score
    return best_id, best_score


def evaluate_run(label: str, hypotheses: dict, gt_status_map: dict):
    """For one run, score each canonical hypothesis against its content-aligned LLM hyp."""
    # Build LLM candidates: id → content.
    candidates = {hid: h["content"] for hid, h in hypotheses.items()}
    rows = []
    for canon_id, canon_content in GT_HYP.items():
        gt = gt_status_map[canon_id]
        match_id, score = best_match(canon_content, candidates)
        if match_id is None:
            llm_status = None
            exact = False
            boundary_ok = False  # missing = wrong (no entity to flip = no extraction)
        else:
            llm_status = hypotheses[match_id].get("status")
            exact = (llm_status == gt)
            boundary_ok = (boundary(llm_status) == boundary(gt))
        rows.append({
            "run": label,
            "canonical": canon_id,
            "gt_status": gt,
            "matched_llm_id": match_id,
            "match_score": round(score, 3),
            "llm_status": llm_status,
            "exact_status_match": exact,
            "boundary_match": boundary_ok,
        })
    return rows


def discover_gpt4o_runs():
    """Return [(label, hypotheses_dict)] for every GPT-4o pipeline output found."""
    runs = []
    # C3 single run.
    f = PIPELINE_DEP_DIR / "pipeline_output_gpt4o_v0.json"
    if f.exists():
        d = json.load(open(f))
        runs.append((f"C3/{f.stem}", d["hypotheses"]))
    # e1b_ablation: 4 prompt variants × 3 runs (and 1 self-consistency).
    for f in sorted(E1B_DIR.glob("pipeline_output_gpt4o_*.json")):
        if "_dep_eval" in f.name or "_pilot" in f.name:
            continue
        d = json.load(open(f))
        runs.append((f"e1b/{f.stem}", d["hypotheses"]))
    return runs


def main():
    runs = discover_gpt4o_runs()
    print(f"Discovered {len(runs)} GPT-4o pipeline runs:")
    for label, _ in runs:
        print(f"  {label}")

    all_rows = []
    for label, hyps in runs:
        all_rows.extend(evaluate_run(label, hyps, GT_STATUS))

    n_total = len(all_rows)  # 4 hyps × n_runs
    n_exact = sum(1 for r in all_rows if r["exact_status_match"])
    n_boundary = sum(1 for r in all_rows if r["boundary_match"])

    # Per-canonical-hypothesis aggregation.
    by_canon = defaultdict(lambda: {"n": 0, "exact": 0, "boundary": 0})
    for r in all_rows:
        c = r["canonical"]
        by_canon[c]["n"] += 1
        by_canon[c]["exact"] += int(r["exact_status_match"])
        by_canon[c]["boundary"] += int(r["boundary_match"])

    # Per-status-class (good vs not_good): h1/h3/h4 vs h2.
    by_class = {"good (h1, h3, h4)": {"n": 0, "exact": 0, "boundary": 0},
                "not_good (h2)": {"n": 0, "exact": 0, "boundary": 0}}
    for r in all_rows:
        cls = "not_good (h2)" if r["canonical"] == "h2" else "good (h1, h3, h4)"
        by_class[cls]["n"] += 1
        by_class[cls]["exact"] += int(r["exact_status_match"])
        by_class[cls]["boundary"] += int(r["boundary_match"])

    print()
    print("=" * 70)
    print("PER-RUN × PER-CANONICAL ROWS")
    print("=" * 70)
    for r in all_rows:
        ok_e = "✓" if r["exact_status_match"] else "✗"
        ok_b = "✓" if r["boundary_match"] else "✗"
        match_str = (f"{r['matched_llm_id']}({r['match_score']:.2f})"
                     if r["matched_llm_id"] else "(no match)")
        print(f"  {r['run']:<58} {r['canonical']} → {match_str:<22}  "
              f"gt={r['gt_status']:<10} llm={str(r['llm_status']):<10} "
              f"exact={ok_e} boundary={ok_b}")

    print()
    print("=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    print(f"  total observations: {n_total} (= 4 hypotheses × {len(runs)} runs)")
    print(f"  exact status match:    {n_exact}/{n_total} = {n_exact/n_total:.3f}")
    print(f"  boundary match:        {n_boundary}/{n_total} = {n_boundary/n_total:.3f}")
    print(f"  ε_status_observed (boundary error rate): {1 - n_boundary/n_total:.3f}")

    print()
    print("By canonical hypothesis:")
    for c in ("h1", "h2", "h3", "h4"):
        d = by_canon[c]
        print(f"  {c}: exact {d['exact']}/{d['n']} = {d['exact']/d['n']:.3f}  "
              f"boundary {d['boundary']}/{d['n']} = {d['boundary']/d['n']:.3f}  "
              f"(GT={GT_STATUS[c]})")

    print()
    print("By boundary class:")
    for cls, d in by_class.items():
        print(f"  {cls}: boundary {d['boundary']}/{d['n']} = {d['boundary']/d['n']:.3f}  "
              f"(error rate {1 - d['boundary']/d['n']:.3f})")

    # Persist.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_runs": len(runs),
        "runs_used": [r[0] for r in runs],
        "gt_status": GT_STATUS,
        "n_observations": n_total,
        "exact_status_match": n_exact,
        "boundary_match": n_boundary,
        "epsilon_status_observed_aggregate": round(1 - n_boundary / n_total, 4),
        "by_canonical": {c: {**d, "exact_acc": round(d["exact"] / d["n"], 4),
                             "boundary_acc": round(d["boundary"] / d["n"], 4)}
                         for c, d in by_canon.items()},
        "by_boundary_class": {cls: {**d,
                                    "boundary_acc": round(d["boundary"] / d["n"], 4),
                                    "boundary_err": round(1 - d["boundary"] / d["n"], 4)}
                              for cls, d in by_class.items()},
        "rows": all_rows,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
