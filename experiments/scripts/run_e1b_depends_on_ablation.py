"""
run_e1b_depends_on_ablation.py — depends_on prompt-schema ablation (E1b).

Tests 4 variants of how the `depends_on` field is prompted in
pipeline.CLASSIFY_SYSTEM, while keeping everything else fixed:
  (a) baseline   — current pipeline.CLASSIFY_SYSTEM unchanged
  (b) cot        — adds a CoT step before depends_on output
  (c) examples   — adds 2 worked examples showing cross-hypothesis Dep
  (d) self-cons  — runs baseline 5x at temperature 0.7, majority vote per edge

GPT-4o-only this round (per locked decision 2026-04-27).

Output strategy: for each variant we save a `pipeline_output_<variant>.json`
in the canonical shape (model / turns / dependency_map / hypotheses) so that
the existing battle-tested scorer
`experiments/scripts/eval_dep_extraction.py` can score each variant
identically to the published `tab:end-to-end` reference (P=1.00 R=0.43 F1=0.60
under the "fixed" content-matched observation alignment).

Honest caveat the script surfaces: the current pipeline schema sets
engine.dependencies[h] only at hypothesis CREATION time, so the post-T13
unification override (h1->h4 cross-hypothesis link) is structurally
unrecoverable through prompt variants alone. E1b makes this empirically
visible and will appear as a permanent FN in the post-T13 GT.

Usage:
  export OPENAI_API_KEY=...
  python run_e1b_depends_on_ablation.py                       # all variants
  python run_e1b_depends_on_ablation.py --variant baseline    # smoke
  python run_e1b_depends_on_ablation.py --score-only          # re-score existing outputs

After running, score with:
  for v in baseline cot examples self-consistency; do
    python experiments/scripts/eval_dep_extraction.py \\
      experiments/results/e1b_ablation/pipeline_output_gpt4o_${v}.json
  done
"""

import os
import sys
import json
import argparse
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from pipeline import EpistemicPipeline, PHASE2_CONVERSATION  # noqa: E402

# Reuse the battle-tested content-alignment matchers for both deterministic
# scoring and per-sample alignment in self-consistency voting.
# eval_dep_extraction.py is a sibling in experiments/scripts/.
import eval_dep_extraction as ev  # noqa: E402

# Two ground truths. Both are needed:
#   - GROUND_TRUTH_DEP_CREATION = upper bound for prompt engineering under the
#     current pipeline schema (engine.dependencies[h] is set once at hypothesis
#     creation, never updated).
#   - GROUND_TRUTH_DEP_POST_T13 = full reference state including the T13
#     unification override; recovering h1->h4 here is structurally impossible
#     under the current schema, no matter the prompt.
GROUND_TRUTH_DEP_CREATION = {
    "h1": {"o8"},
    "h2": {"o8"},
    "h3": {"o5"},
    "h4": {"h3", "o6"},
}
GROUND_TRUTH_DEP_POST_T13 = {
    "h1": {"h4", "o8"},
    "h2": {"o8"},
    "h3": {"o5"},
    "h4": {"h3", "o6", "o9"},
}

ORIGINAL_CLASSIFY_SYSTEM = pipeline.CLASSIFY_SYSTEM

DEP_HINT_BASELINE = (
    '- "depends_on": which observations/hypotheses does this depend on?\n'
    '  This is crucial for dependency tracking.\n'
)

DEP_HINT_COT = (
    '- "depends_on": which observations/hypotheses does this depend on?\n'
    '  This is crucial for dependency tracking. Before listing, think:\n'
    '  1. What evidence does the hypothesis rely on? (observations)\n'
    '  2. Does the hypothesis chain through any earlier hypothesis? (cross-hypothesis links)\n'
    '  3. List ALL such IDs in depends_on, including hypotheses if relevant.\n'
)

DEP_HINT_EXAMPLES = (
    '- "depends_on": which observations/hypotheses does this depend on?\n'
    '  This is crucial for dependency tracking. Examples:\n'
    '  Example A: hypothesis "Redis exhaustion -> rate-limit bypass -> 429s" rests\n'
    '    on observation o8 (Redis timeout). depends_on: ["o8"].\n'
    '  Example B: hypothesis "Token bug -> retry storm -> Redis exhaustion" rests\n'
    '    on hypothesis h3 (token bug -> retry loop) AND observation o6\n'
    '    (auth traffic 3x). depends_on: ["h3", "o6"]. Note BOTH the\n'
    '    hypothesis AND the observation are listed.\n'
)

VARIANT_HINTS = {
    "baseline":  DEP_HINT_BASELINE,
    "cot":       DEP_HINT_COT,
    "examples":  DEP_HINT_EXAMPLES,
}


def install_variant(variant_name):
    new_hint = VARIANT_HINTS[variant_name]
    new_system = ORIGINAL_CLASSIFY_SYSTEM.replace(DEP_HINT_BASELINE, new_hint)
    if new_system == ORIGINAL_CLASSIFY_SYSTEM and variant_name != "baseline":
        sys.exit(f"FAIL: depends_on hint substitution didn't change anything "
                 f"for variant {variant_name}. The baseline string in "
                 f"pipeline.CLASSIFY_SYSTEM has changed; please update "
                 f"DEP_HINT_BASELINE in this script.")
    pipeline.CLASSIFY_SYSTEM = new_system


def restore_classify_system():
    pipeline.CLASSIFY_SYSTEM = ORIGINAL_CLASSIFY_SYSTEM


def run_pipeline(api_key, model_name):
    """Run the canonical Phase 2 conversation through the pipeline. Uses
    pipeline.PHASE2_CONVERSATION (the same transcript every other experiment
    in this repo runs against), not a local abbreviated copy."""
    pipe = EpistemicPipeline(api_key, verbose=False)
    for turn in PHASE2_CONVERSATION:
        try:
            pipe.process_turn(turn["speaker"], turn["text"])
        except Exception as e:
            print(f"    WARN: turn failed: {e}")
        time.sleep(0.2)
    return pipe


def serialize_pipeline_output(pipe, model_name):
    """Build the canonical pipeline_output dict (model / turns /
    dependency_map / hypotheses) consumed by eval_dep_extraction.py."""
    deps = pipe.get_dependency_map()
    return {
        "model": model_name,
        "turns": pipe.classifications,
        "dependency_map": {h: list(d) for h, d in deps.items()},
        "hypotheses": {
            hid: {"content": h.content, "status": h.status.value}
            for hid, h in pipe.engine.hypotheses.items()
        },
    }


def run_self_consistency(api_key, model_name, n_samples=5, threshold=3):
    """Content-aligned self-consistency voting (replaces the pilot's
    raw-ID voter, which was unstable across temperature samples).

    Procedure:
      1. Run n_samples hot pipeline runs (temperature=0.7) under the
         baseline depends_on hint.
      2. For each sample, content-align its hypotheses to canonical
         h1..h4 via ev.match_llm_to_gt and its observations to canonical
         o5/o6/o8/o9 via ev.match_llm_obs_to_gt. Translate that sample's
         raw dep tuples to canonical (gt_hyp, gt_dep) tuples; drop any
         tuple whose hyp or dep doesn't align.
      3. Vote on canonical tuples; keep edges with >= threshold votes.
      4. Emit a synthetic pipeline_output keyed by canonical IDs and
         flagged sc_aligned=True. The scoring path identity-maps the
         canonical IDs (no further alignment needed).

    This eliminates the ID-instability the pilot SC voter suffered from:
    if h_auto_3 in sample 1 and h_auto_5 in sample 2 both content-align
    to canonical h4, their dep edges now vote together for canonical
    edges of h4, instead of being treated as different entities.
    """
    original_call_llm = pipeline.call_llm

    def hot_call_llm(system, user, api_key, temperature=0, max_retries=3):
        return original_call_llm(system, user, api_key,
                                 temperature=0.7, max_retries=max_retries)

    install_variant("baseline")
    sample_data = []
    pipeline.call_llm = hot_call_llm
    try:
        for i in range(n_samples):
            print(f"    sample {i+1}/{n_samples} (T=0.7)...")
            pipe = run_pipeline(api_key, model_name)
            engine = pipe.engine

            llm_hyps_for_match = {
                hid: {"content": h.content,
                      "status": (h.status.value if hasattr(h.status, "value") else str(h.status))}
                for hid, h in engine.hypotheses.items()
            }
            llm_obs_for_match = {oid: o.content for oid, o in engine.observations.items()}

            hyp_matches = ev.match_llm_to_gt(llm_hyps_for_match, ev.GT_H_CONTENT)
            obs_matches = ev.match_llm_obs_to_gt(llm_obs_for_match, ev.GT_O_CONTENT)

            llm_to_gt_hyp = {m[0]: gt for gt, m in hyp_matches.items() if m[0] is not None}
            llm_to_gt_obs = {m[0]: gt for gt, m in obs_matches.items() if m[0] is not None}

            raw_deps = {hid: list(d) for hid, d in pipe.get_dependency_map().items()}
            canonical_deps = {}
            for raw_h, raw_ds in raw_deps.items():
                canon_h = llm_to_gt_hyp.get(raw_h)
                if canon_h is None:
                    continue
                for raw_d in raw_ds:
                    if raw_d.startswith("h"):
                        canon_d = llm_to_gt_hyp.get(raw_d)
                    elif raw_d.startswith("o"):
                        canon_d = llm_to_gt_obs.get(raw_d)
                    else:
                        canon_d = None
                    if canon_d is None:
                        continue
                    canonical_deps.setdefault(canon_h, set()).add(canon_d)

            sample_data.append({
                "raw_deps": raw_deps,
                "canonical_deps": {h: sorted(ds) for h, ds in canonical_deps.items()},
                "hyp_matches": {gt: {"llm_id": m[0], "overlap": m[1]}
                                for gt, m in hyp_matches.items()},
                "obs_matches": {gt: {"llm_id": m[0], "overlap": m[1]}
                                for gt, m in obs_matches.items()},
            })
    finally:
        pipeline.call_llm = original_call_llm
        restore_classify_system()

    # Aggregate canonical edges across samples.
    edge_votes = Counter()
    for s in sample_data:
        for h, ds in s["canonical_deps"].items():
            for d in ds:
                edge_votes[(h, d)] += 1

    aggregated_canonical = {}
    for (h, d), votes in edge_votes.items():
        if votes >= threshold:
            aggregated_canonical.setdefault(h, []).append(d)
    # Sort dep lists for stable output
    aggregated_canonical = {h: sorted(ds) for h, ds in aggregated_canonical.items()}

    # Synthesize hypotheses dict using canonical IDs and reference content.
    # Include any canonical h that's either a key or appears as a dep value.
    referenced_hyps = set(aggregated_canonical.keys())
    for ds in aggregated_canonical.values():
        for d in ds:
            if d.startswith("h"):
                referenced_hyps.add(d)
    synthesized_hyps = {
        h: {"content": ev.GT_H_CONTENT.get(h, ""), "status": "active"}
        for h in sorted(referenced_hyps)
    }

    out = {
        "model": model_name,
        "sc_aligned": True,
        "turns": [],  # not used in canonical-aligned scoring path
        "dependency_map": aggregated_canonical,
        "hypotheses": synthesized_hyps,
        "sc_metadata": {
            "n_samples": n_samples,
            "vote_threshold": threshold,
            "temperature": 0.7,
            "alignment_mode": "per-sample content-match before voting (canonical-aligned)",
            "edge_votes_canonical": {f"{h}->{d}": v
                                     for (h, d), v in sorted(edge_votes.items())},
            "per_sample": sample_data,
        },
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",
                   choices=list(VARIANT_HINTS) + ["self-consistency", "all"],
                   default="all")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--base-url",
                   default="https://api.openai.com/v1/chat/completions")
    p.add_argument("--out-dir",
                   default="experiments/results/e1b_ablation")
    p.add_argument("--run-id", type=int, default=None,
                   help="Suffix output as _run<N>.json. Default: auto-pick next "
                        "free run id (0, 1, 2, ...) based on what exists in --out-dir.")
    p.add_argument("--score-only", action="store_true",
                   help="Skip running; just aggregate scoring across all "
                        "_run*.json files per variant in --out-dir.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = (list(VARIANT_HINTS) + ["self-consistency"]
                if args.variant == "all" else [args.variant])

    def existing_run_ids(variant_name):
        """Return sorted list of run-ids found on disk for a variant."""
        prefix = f"pipeline_output_gpt4o_{variant_name}_run"
        ids = []
        for f in out_dir.glob(f"{prefix}*.json"):
            if f.stem.endswith("_dep_eval"):
                continue
            tail = f.stem[len(prefix):]
            try:
                ids.append(int(tail))
            except ValueError:
                continue
        return sorted(ids)

    if not args.score_only:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("Set OPENAI_API_KEY (E1b uses GPT-4o).")

        pipeline.BACKEND = "openai"
        pipeline.MODEL = args.model
        pipeline.API_URL = args.base_url

        for v in variants:
            # Pick run-id: explicit arg, or next free id auto-detected.
            if args.run_id is not None:
                run_id = args.run_id
            else:
                existing = existing_run_ids(v)
                run_id = (existing[-1] + 1) if existing else 0

            print(f"\n=== Variant: {v} (run_id={run_id}) ===")
            if v == "self-consistency":
                out = run_self_consistency(api_key, args.model)
            else:
                install_variant(v)
                try:
                    pipe = run_pipeline(api_key, args.model)
                    out = serialize_pipeline_output(pipe, args.model)
                finally:
                    restore_classify_system()

            out["run_id"] = run_id
            out_path = out_dir / f"pipeline_output_gpt4o_{v}_run{run_id}.json"
            if out_path.exists():
                print(f"  WARN: overwriting existing {out_path}")
            out_path.write_text(json.dumps(out, indent=2))
            print(f"  Saved {out_path}")
            print(f"  hypotheses: {sorted(out['hypotheses'])}")
            print(f"  dependency_map (raw IDs): "
                  f"{ {k: sorted(vs) for k, vs in out['dependency_map'].items()} }")

    # Score each variant with the canonical scorer.
    print("\n" + "=" * 60)
    print("SCORING (using experiments/scripts/eval_dep_extraction.py)")
    print("=" * 60)

    eval_script = Path("experiments/scripts/eval_dep_extraction.py").resolve()
    if not eval_script.exists():
        sys.exit(f"eval_dep_extraction.py not found at {eval_script}")

    # Capture the scorer's default GT (post-T13) so we can swap it for the
    # creation-time scoring pass and restore it after.
    EV_GT_POST_T13 = ev.GT_DEPS

    def _score_against(gt_deps, llm_deps, hyp_matches, obs_matches):
        """ev.score_dep reads GT_DEPS from module scope; toggle it and
        restore."""
        ev.GT_DEPS = gt_deps
        try:
            naive = ev.score_dep(llm_deps, hyp_matches, obs_matches,
                                 passthrough_obs=True)
            fixed = ev.score_dep(llm_deps, hyp_matches, obs_matches,
                                 passthrough_obs=False)
        finally:
            ev.GT_DEPS = EV_GT_POST_T13
        return naive, fixed

    def _score_one_run(out_path):
        """Score a single pipeline_output JSON; returns the per-run dict."""
        d = json.loads(out_path.read_text())
        llm_hyps = d.get("hypotheses", {})
        llm_deps = {h: set(deps) for h, deps in d.get("dependency_map", {}).items()}
        llm_obs = ev._extract_llm_observations(d)
        if d.get("sc_aligned"):
            # Canonical-aligned SC output: hyps and obs are already in canonical
            # IDs. Use identity matches so ev.score_dep doesn't re-align.
            hyp_matches = {gt: (gt, 1.0) for gt in ev.GT_H_CONTENT}
            obs_matches = {gt: (gt, 1.0) for gt in ev.GT_O_CONTENT}
        else:
            hyp_matches = ev.match_llm_to_gt(llm_hyps, ev.GT_H_CONTENT)
            obs_matches = ev.match_llm_obs_to_gt(llm_obs, ev.GT_O_CONTENT)
        dep_naive_creat, dep_fixed_creat = _score_against(
            GROUND_TRUTH_DEP_CREATION, llm_deps, hyp_matches, obs_matches)
        dep_naive_post, dep_fixed_post = _score_against(
            GROUND_TRUTH_DEP_POST_T13, llm_deps, hyp_matches, obs_matches)
        aff_fixed = ev.score_affected(llm_deps, hyp_matches, obs_matches,
                                      passthrough_obs=False)
        h1_llm = next((m[0] for gt, m in hyp_matches.items() if gt == "h1" and m[0]), None)
        h4_llm = next((m[0] for gt, m in hyp_matches.items() if gt == "h4" and m[0]), None)
        h3_llm = next((m[0] for gt, m in hyp_matches.items() if gt == "h3" and m[0]), None)
        h1_to_h4 = (h1_llm is not None and h4_llm is not None
                    and h4_llm in llm_deps.get(h1_llm, set()))
        h4_to_h3 = (h4_llm is not None and h3_llm is not None
                    and h3_llm in llm_deps.get(h4_llm, set()))
        return {
            "path": str(out_path),
            "model": d.get("model"),
            "fixed_creation": dep_fixed_creat,
            "fixed_post_t13": dep_fixed_post,
            "naive_creation": dep_naive_creat,
            "naive_post_t13": dep_naive_post,
            "aff_fixed": aff_fixed,
            "h1_to_h4": h1_to_h4,
            "h4_to_h3": h4_to_h3,
            "hyp_matches": hyp_matches,
            "raw_data": d,
        }

    def _aggregate(per_run):
        """Aggregate per-run scores: median P/R/F1, min/max, modal cross-hyp."""
        from statistics import median
        if not per_run:
            return None
        agg = {}
        for kind in ("fixed_creation", "fixed_post_t13"):
            for metric in ("precision", "recall", "f1"):
                vs = [r[kind][metric] for r in per_run]
                agg[f"{kind}_{metric}_median"] = median(vs)
                agg[f"{kind}_{metric}_min"] = min(vs)
                agg[f"{kind}_{metric}_max"] = max(vs)
        # Modal (or "any"-style) cross-hypothesis recovery: report fraction
        # of runs that recovered each link.
        agg["h1_to_h4_recovered_frac"] = sum(r["h1_to_h4"] for r in per_run) / len(per_run)
        agg["h4_to_h3_recovered_frac"] = sum(r["h4_to_h3"] for r in per_run) / len(per_run)
        agg["aff_acc_median"] = median([r["aff_fixed"]["accuracy"] for r in per_run])
        agg["n_runs"] = len(per_run)
        return agg

    summary_rows = []
    for v in variants:
        # Find all runs for this variant.
        run_paths = sorted(out_dir.glob(f"pipeline_output_gpt4o_{v}_run*.json"))
        run_paths = [p for p in run_paths if not p.stem.endswith("_dep_eval")]
        if not run_paths:
            print(f"  SKIP {v}: no _run*.json found in {out_dir}")
            continue

        per_run = [_score_one_run(p) for p in run_paths]

        # Write per-run eval JSONs (overwrite if present).
        for run_data, p in zip(per_run, run_paths):
            sc_meta = run_data["raw_data"].get("sc_metadata")
            eval_path = p.with_name(p.stem + "_dep_eval.json")
            payload = {
                "variant": v,
                "run_id": run_data["raw_data"].get("run_id"),
                "model": run_data["model"],
                "hyp_matches": {gt: {"llm_id": m[0], "overlap": m[1]}
                                for gt, m in run_data["hyp_matches"].items()},
                "dep_scores_creation_fixed": {k: v_ for k, v_ in run_data["fixed_creation"].items()
                                              if k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                "dep_detail_creation_fixed": {k: v_ for k, v_ in run_data["fixed_creation"].items()
                                              if k in ("llm_tuples_in_gt_space", "gt_tuples",
                                                       "missed_tuples", "extra_tuples")},
                "dep_scores_post_t13_fixed": {k: v_ for k, v_ in run_data["fixed_post_t13"].items()
                                              if k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                "dep_detail_post_t13_fixed": {k: v_ for k, v_ in run_data["fixed_post_t13"].items()
                                              if k in ("llm_tuples_in_gt_space", "gt_tuples",
                                                       "missed_tuples", "extra_tuples")},
                "affected_fixed": run_data["aff_fixed"],
                "cross_hypothesis_links": {
                    "h1_to_h4_recovered": run_data["h1_to_h4"],
                    "h4_to_h3_recovered": run_data["h4_to_h3"],
                },
                "ground_truth_creation": {k: sorted(v_) for k, v_ in GROUND_TRUTH_DEP_CREATION.items()},
                "ground_truth_post_t13": {k: sorted(v_) for k, v_ in GROUND_TRUTH_DEP_POST_T13.items()},
            }
            if v == "self-consistency":
                payload["sc_alignment_mode"] = (
                    "Per-sample content-aligned voting: each of 5 hot "
                    "samples (T=0.7) is aligned to canonical h1..h4 / "
                    "o5/o6/o8/o9 via ev.match_llm_to_gt + match_llm_obs_to_gt "
                    "before voting. Edges aggregated in canonical (gt_hyp, "
                    "gt_dep) space, eliminating the auto-ID instability "
                    "the pilot raw-ID voter suffered from."
                )
                if sc_meta:
                    payload["sc_metadata"] = sc_meta
            eval_path.write_text(json.dumps(payload, indent=2, default=list))

        agg = _aggregate(per_run)
        summary_rows.append({
            "variant": v,
            "n_runs": len(per_run),
            "agg": agg,
            "per_run_f1_creation": [r["fixed_creation"]["f1"] for r in per_run],
            "per_run_f1_post_t13": [r["fixed_post_t13"]["f1"] for r in per_run],
            "per_run_h1_to_h4": [r["h1_to_h4"] for r in per_run],
            "per_run_h4_to_h3": [r["h4_to_h3"] for r in per_run],
            "per_run_aff": [r["aff_fixed"]["accuracy"] for r in per_run],
            "exploratory": False,
        })

        print(f"\n  [{v}] n_runs={len(per_run)}")
        print(f"    Per-run F1 (creation): "
              f"{[round(r['fixed_creation']['f1'], 3) for r in per_run]}")
        print(f"    Per-run F1 (post-T13): "
              f"{[round(r['fixed_post_t13']['f1'], 3) for r in per_run]}")
        print(f"    Median F1 creation: {agg['fixed_creation_f1_median']:.3f}  "
              f"[range {agg['fixed_creation_f1_min']:.3f}-{agg['fixed_creation_f1_max']:.3f}]")
        print(f"    Median F1 post-T13: {agg['fixed_post_t13_f1_median']:.3f}  "
              f"[range {agg['fixed_post_t13_f1_min']:.3f}-{agg['fixed_post_t13_f1_max']:.3f}]")
        print(f"    h1->h4 recovered in {sum(r['h1_to_h4'] for r in per_run)}/{len(per_run)} runs")
        print(f"    h4->h3 recovered in {sum(r['h4_to_h3'] for r in per_run)}/{len(per_run)} runs")
        if v == "self-consistency":
            print(f"    [SC EXPLORATORY]")

    # Aggregate summary table (median across runs per variant).
    print("\n" + "=" * 60)
    print("E1b AGGREGATE SUMMARY (fixed-mode, dual-GT, median across runs)")
    print("=" * 60)
    print(f"  CREATION-time GT (prompt-engineering ceiling, h1={{o8}}, h4={{h3,o6}}):")
    print(f"  {'Variant':<20} {'n':>3}  {'F1 med':>7} {'[min':>6} {'max]':>6}")
    for r in summary_rows:
        a = r["agg"]
        tag = " *" if r["exploratory"] else "  "
        print(f"  {r['variant']:<20}{tag}{r['n_runs']:>3}  "
              f"{a['fixed_creation_f1_median']:>7.3f} "
              f"{a['fixed_creation_f1_min']:>6.3f} {a['fixed_creation_f1_max']:>6.3f}")
    print()
    print(f"  POST-T13 GT (full reference, includes structural ceiling):")
    print(f"  {'Variant':<20} {'n':>3}  {'F1 med':>7} {'[min':>6} {'max]':>6}  "
          f"{'h1->h4':>7} {'h4->h3':>7}  {'Aff':>6}")
    for r in summary_rows:
        a = r["agg"]
        tag = " *" if r["exploratory"] else "  "
        h1h4_n = round(a['h1_to_h4_recovered_frac'] * r['n_runs'])
        h4h3_n = round(a['h4_to_h3_recovered_frac'] * r['n_runs'])
        h1h4_str = f"{h1h4_n}/{r['n_runs']}"
        h4h3_str = f"{h4h3_n}/{r['n_runs']}"
        print(f"  {r['variant']:<20}{tag}{r['n_runs']:>3}  "
              f"{a['fixed_post_t13_f1_median']:>7.3f} "
              f"{a['fixed_post_t13_f1_min']:>6.3f} {a['fixed_post_t13_f1_max']:>6.3f}  "
              f"{h1h4_str:>7} {h4h3_str:>7}  "
              f"{a['aff_acc_median']:>6.3f}")
    print(f"\n  Reference (committed v0 GPT-4o pipeline, post-T13 fixed, single run): "
          f"P=1.000 R=0.429 F1=0.600  h1->h4: no  h4->h3: YES")
    print(f"\n  Self-consistency uses content-aligned voting over 5 hot samples")
    print(f"  (per-sample alignment to canonical IDs before majority vote).")
    print(f"\nNote: h1->h4 (post-T13 unification override) is structurally")
    print(f"  unrecoverable through prompt-engineering alone — the current pipeline")
    print(f"  schema sets engine.dependencies[h] only at hypothesis CREATION time")
    print(f"  and has no mechanism to update existing hypotheses' deps. This is a")
    print(f"  positive scoping result for the paper, not a model failure.")

    aggregate_path = out_dir / "e1b_summary.json"
    aggregate_path.write_text(json.dumps({
        "experiment": "E1b — depends_on prompt-schema ablation",
        "model": args.model,
        "backend": "openai",
        "results": summary_rows,
    }, indent=2, default=list))
    print(f"\nAggregate summary saved to {aggregate_path}")


if __name__ == "__main__":
    main()
