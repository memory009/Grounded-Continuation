#!/usr/bin/env python3
"""
End-to-end dependency extraction evaluation.

Ground truth (paper tex:362-365 + symbolic_engine.py Phase 2 demo):
  Dep(h1) = {h4, o8}    h1: Redis exhaustion -> rate-limit bypass -> Stripe 429s
  Dep(h2) = {o8}        h2: Redis failure -> Auth failures (abandoned)
  Dep(h3) = {o5}        h3: token bug -> retry loop -> 3x traffic
  Dep(h4) = {h3,o6,o9}  h4: token bug -> retry storm -> Redis exhaustion
Paper's retraction queries (tex:362-365):
  Affected(o9) = {h4}
  Affected(o8) = {h1, h2}
  Affected(o6) = {h4}

This script:
  1. Loads pipeline_output.json produced by `python pipeline.py`.
  2. Matches LLM-extracted hypotheses to the 4 ground-truth hypotheses by
     token overlap on the content strings.
  3. Reports per-GT-hypothesis Dep precision/recall + micro-averaged F1.
  4. Computes Affected(p) with the LLM-extracted deps (transitive closure
     over the matched dep graph) for p in {o9, o8, o6} and compares to
     paper's GT answers.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

GT_DEPS = {
    "h1": {"h4", "o8"},
    "h2": {"o8"},
    "h3": {"o5"},
    "h4": {"h3", "o6", "o9"},
}
GT_AFFECTED = {
    "o9": {"h4"},
    "o8": {"h1", "h2"},
    "o6": {"h4"},
}
# Canonical GT hypothesis descriptions distilled from the paper (tex:236-246)
# and the symbolic_engine Phase 2 demo. Used only for content-matching
# LLM-generated hypothesis IDs to the canonical GT IDs.
GT_H_CONTENT = {
    "h1": "redis connection pool exhaustion rate limit bypass stripe 429 payment",
    "h2": "redis failure auth failures via shared session cache",
    "h3": "token bug frontend retry loop 3x traffic amplification auth volume",
    "h4": "token bug retry storm redis pool exhaustion causal reversal",
}
# Ground-truth observation content hints (optional, for informational purposes).
GT_O_CONTENT = {
    "o5": "token expired 401 unauthorized",
    "o6": "auth traffic 3x normal anomalous volume",
    "o8": "redis connection timeouts pool exhausted",
    "o9": "error code 401 token expired distinct from 503 session lookup",
}


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return [w for w in text.split() if len(w) > 2]


def token_overlap(a: str, b: str) -> float:
    """Symmetric Jaccard-ish: |A∩B| / max(|A|,|B|)."""
    ta, tb = set(normalize(a)), set(normalize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def match_llm_to_gt(llm_hyps: dict, gt_contents: dict,
                    min_overlap: float = 0.15) -> dict:
    """Each GT hypothesis claims the LLM hypothesis with the highest token
    overlap (>= min_overlap). Returns {gt_id: llm_id or None}.
    """
    matches: dict = {}
    used: set = set()
    # Score every pair once
    scored = []
    for gt_id, gt_txt in gt_contents.items():
        for llm_id, llm_h in llm_hyps.items():
            s = token_overlap(gt_txt, llm_h["content"])
            scored.append((s, gt_id, llm_id))
    scored.sort(reverse=True)
    for s, gt_id, llm_id in scored:
        if gt_id in matches or llm_id in used:
            continue
        if s < min_overlap:
            break
        matches[gt_id] = (llm_id, round(s, 3))
        used.add(llm_id)
    for gt_id in gt_contents:
        if gt_id not in matches:
            matches[gt_id] = (None, 0.0)
    return matches


def match_llm_obs_to_gt(llm_obs: dict, gt_contents: dict,
                        min_overlap: float = 0.15) -> dict:
    """Content-based matcher for observations. Mirrors match_llm_to_gt.

    llm_obs: {llm_o_id: content_string}
    gt_contents: GT_O_CONTENT (GT id -> short content hint string)
    Returns {gt_o_id: (llm_o_id or None, overlap)}.

    Needed because `pipeline.py` numbers observations in order of LLM
    discovery (o1, o2, ...), while the paper's GT uses its own numbering.
    Without this, o*-pass-through in translate_llm_dep() systematically
    mis-aligns correct LLM dependencies as false positives.
    """
    matches: dict = {}
    used: set = set()
    scored = []
    for gt_id, gt_txt in gt_contents.items():
        for llm_id, llm_txt in llm_obs.items():
            s = token_overlap(gt_txt, llm_txt)
            scored.append((s, gt_id, llm_id))
    scored.sort(reverse=True)
    for s, gt_id, llm_id in scored:
        if gt_id in matches or llm_id in used:
            continue
        if s < min_overlap:
            break
        matches[gt_id] = (llm_id, round(s, 3))
        used.add(llm_id)
    for gt_id in gt_contents:
        if gt_id not in matches:
            matches[gt_id] = (None, 0.0)
    return matches


def translate_llm_dep(llm_dep_id: str,
                      gt_from_llm_hyp: dict,
                      gt_from_llm_obs: dict,
                      passthrough_obs: bool) -> Optional[str]:
    """Convert an LLM dep id to its canonical GT id.
    - LLM hypothesis ids (h*) are translated via the hypothesis match table.
    - LLM observation ids (o*):
        * passthrough_obs=True  -> return id unchanged (old naive mode,
          kept for before/after comparison in output);
        * passthrough_obs=False -> translate via gt_from_llm_obs
          (content-matched), return None if no match.
    """
    if llm_dep_id.startswith("o"):
        if passthrough_obs:
            return llm_dep_id
        return gt_from_llm_obs.get(llm_dep_id)
    return gt_from_llm_hyp.get(llm_dep_id)


def score_dep(llm_deps: dict, hyp_matches: dict,
              obs_matches: dict, passthrough_obs: bool) -> dict:
    """Micro-averaged precision/recall/F1 of Dep tuples (hyp, dep) under
    the LLM->GT matching. Two modes selected by passthrough_obs."""
    # Invert matches: llm_id -> gt_id
    llm_to_gt_hyp = {v[0]: gt for gt, v in hyp_matches.items() if v[0] is not None}
    llm_to_gt_obs = ({v[0]: gt for gt, v in obs_matches.items() if v[0] is not None}
                     if (obs_matches and not passthrough_obs) else {})

    # Project LLM tuples into GT space where possible
    llm_tuples_in_gt = set()
    for llm_hyp, deps in llm_deps.items():
        gt_hyp = llm_to_gt_hyp.get(llm_hyp)
        if gt_hyp is None:
            continue
        for d in deps:
            tr = translate_llm_dep(d, llm_to_gt_hyp, llm_to_gt_obs, passthrough_obs)
            if tr is None:
                continue
            llm_tuples_in_gt.add((gt_hyp, tr))

    gt_tuples = set((h, d) for h, deps in GT_DEPS.items() for d in deps)

    tp = len(llm_tuples_in_gt & gt_tuples)
    fp = len(llm_tuples_in_gt - gt_tuples)
    fn = len(gt_tuples - llm_tuples_in_gt)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "llm_tuples_in_gt_space": sorted(llm_tuples_in_gt),
        "gt_tuples": sorted(gt_tuples),
        "missed_tuples": sorted(gt_tuples - llm_tuples_in_gt),
        "extra_tuples": sorted(llm_tuples_in_gt - gt_tuples),
    }


def affected_from_llm_deps(llm_deps: dict, hyp_matches: dict,
                           retract_gt: str, obs_matches: dict,
                           passthrough_obs: bool) -> set:
    """Compute Affected(retract) using the LLM-extracted dep graph, then
    project hypothesis IDs to GT canonical IDs.

    retract_gt is always expressed in GT space (e.g., 'o9'). Under
    passthrough_obs we assume LLM uses the same id; under the fixed mode
    we look up which LLM obs id corresponds to that GT id (unmatched ->
    nothing is affected, so returns an empty set).
    """
    llm_to_gt_hyp = {v[0]: gt for gt, v in hyp_matches.items() if v[0] is not None}

    if passthrough_obs:
        retract_llm = retract_gt  # old naive assumption
    else:
        gt_to_llm_obs = ({gt: v[0] for gt, v in obs_matches.items() if v[0] is not None}
                         if obs_matches else {})
        retract_llm = gt_to_llm_obs.get(retract_gt)
        if retract_llm is None:
            return set()

    affected_llm: set = set()
    changed = True
    while changed:
        changed = False
        for h, deps in llm_deps.items():
            if h in affected_llm:
                continue
            if retract_llm in deps or (deps & affected_llm):
                affected_llm.add(h)
                changed = True
    # Project to GT ids; drop unmatched hypotheses.
    return {llm_to_gt_hyp[h] for h in affected_llm if h in llm_to_gt_hyp}


def score_affected(llm_deps: dict, hyp_matches: dict,
                   obs_matches: dict, passthrough_obs: bool) -> dict:
    results = []
    correct = 0
    for retract, gt_set in GT_AFFECTED.items():
        llm_deps_sets = {h: set(d) for h, d in llm_deps.items()}
        llm_affected = affected_from_llm_deps(
            llm_deps_sets, hyp_matches, retract, obs_matches, passthrough_obs)
        is_correct = llm_affected == gt_set
        correct += int(is_correct)
        results.append({
            "retract": retract,
            "gt_affected": sorted(gt_set),
            "llm_affected": sorted(llm_affected),
            "correct": is_correct,
        })
    return {
        "per_query": results,
        "accuracy": round(correct / len(GT_AFFECTED), 4),
        "correct": correct,
        "total": len(GT_AFFECTED),
    }


def _extract_llm_observations(d: dict) -> dict:
    """Walk pipeline_output.json's turns and collect observation
    content keyed by LLM observation id (first-occurrence wins)."""
    obs: dict = {}
    for turn in d.get("turns", []) or []:
        classif = turn.get("classification", {}) or {}
        for o in classif.get("observations") or []:
            oid = o.get("id")
            content = o.get("content", "")
            if oid and oid not in obs:
                obs[oid] = content
    return obs


def _print_dep_block(label: str, dep_score: dict) -> None:
    print("=" * 60)
    print(f"DEP TUPLE PRECISION / RECALL ({label})")
    print("=" * 60)
    print(f"  precision={dep_score['precision']}  "
          f"recall={dep_score['recall']}  f1={dep_score['f1']}")
    print(f"  tp={dep_score['tp']} fp={dep_score['fp']} fn={dep_score['fn']}")
    print(f"  LLM tuples projected to GT space: {dep_score['llm_tuples_in_gt_space']}")
    print(f"  GT tuples:                        {dep_score['gt_tuples']}")
    print(f"  MISSED (GT \\ LLM):                {dep_score['missed_tuples']}")
    print(f"  EXTRA  (LLM \\ GT):                {dep_score['extra_tuples']}")


def _print_affected_block(label: str, aff: dict) -> None:
    print("=" * 60)
    print(f"AFFECTED(p) RETRACTION QUERIES ({label})")
    print("=" * 60)
    for r in aff["per_query"]:
        mark = "OK" if r["correct"] else "WRONG"
        print(f"  [{mark}] Affected({r['retract']}): "
              f"GT={r['gt_affected']} vs LLM={r['llm_affected']}")
    print(f"  Accuracy: {aff['correct']}/{aff['total']} = {aff['accuracy']}")


def main(path: str):
    d = json.loads(Path(path).read_text())
    llm_hyps = d.get("hypotheses", {})
    llm_deps = {h: set(deps) for h, deps in d.get("dependency_map", {}).items()}
    llm_obs = _extract_llm_observations(d)

    # Only observations referenced by the GT dep graph matter for scoring.
    gt_dep_obs = {o for deps in GT_DEPS.values() for o in deps if o.startswith("o")}
    relevant_gt_obs = set(GT_AFFECTED.keys()) | gt_dep_obs

    print(f"Model: {d.get('model')}")
    print(f"LLM hypotheses: {len(llm_hyps)}")
    print(f"LLM observations: {len(llm_obs)}")
    print(f"LLM dep entries: {len(llm_deps)}")
    print()

    # Hypothesis matching (unchanged logic).
    print("=" * 60)
    print("MATCHING LLM hypotheses -> ground truth")
    print("=" * 60)
    hyp_matches = match_llm_to_gt(llm_hyps, GT_H_CONTENT)
    for gt_id, gt_txt in GT_H_CONTENT.items():
        llm_id, score = hyp_matches[gt_id]
        print(f"  GT {gt_id}: {gt_txt}")
        if llm_id is not None:
            h = llm_hyps[llm_id]
            print(f"    -> matched to LLM {llm_id} (overlap={score})")
            print(f"       content: {h['content']}")
        else:
            print(f"    -> NO MATCH (best overlap below threshold)")
    print()

    # NEW: observation matching (only report slots referenced by GT_DEPS /
    # GT_AFFECTED, since other o* don't affect scoring).
    print("=" * 60)
    print("MATCHING LLM observations -> ground truth  [FIXED MODE ONLY]")
    print("=" * 60)
    obs_matches = match_llm_obs_to_gt(llm_obs, GT_O_CONTENT)
    for gt_id in sorted(GT_O_CONTENT):
        if gt_id not in relevant_gt_obs:
            continue
        llm_id, score = obs_matches[gt_id]
        print(f"  GT {gt_id}: {GT_O_CONTENT[gt_id]}")
        if llm_id is not None:
            print(f"    -> matched to LLM {llm_id} (overlap={score})")
            print(f"       content: {llm_obs[llm_id]}")
        else:
            print(f"    -> NO MATCH (best overlap below threshold)")
    print()

    # Score twice: naive (old behaviour, obs id pass-through) and
    # fixed (new, content-matched observations).
    dep_naive = score_dep(llm_deps, hyp_matches, obs_matches,
                          passthrough_obs=True)
    dep_fixed = score_dep(llm_deps, hyp_matches, obs_matches,
                          passthrough_obs=False)
    aff_naive = score_affected(llm_deps, hyp_matches, obs_matches,
                               passthrough_obs=True)
    aff_fixed = score_affected(llm_deps, hyp_matches, obs_matches,
                               passthrough_obs=False)

    _print_dep_block("naive: obs id pass-through", dep_naive)
    print()
    _print_dep_block("fixed: content-matched obs", dep_fixed)
    print()
    _print_affected_block("naive", aff_naive)
    print()
    _print_affected_block("fixed", aff_fixed)
    print()

    # Before/after summary (compact).
    print("=" * 60)
    print("BEFORE / AFTER SUMMARY")
    print("=" * 60)
    print(f"  Dep F1:   naive={dep_naive['f1']:.4f}  fixed={dep_fixed['f1']:.4f}  "
          f"(delta={dep_fixed['f1'] - dep_naive['f1']:+.4f})")
    print(f"  Dep prec: naive={dep_naive['precision']:.4f}  fixed={dep_fixed['precision']:.4f}")
    print(f"  Dep rec:  naive={dep_naive['recall']:.4f}  fixed={dep_fixed['recall']:.4f}")
    print(f"  Affected acc: naive={aff_naive['accuracy']:.4f}  fixed={aff_fixed['accuracy']:.4f}")

    # Emit machine-readable summary with BOTH modes.
    out = {
        "model": d.get("model"),
        "n_llm_hypotheses": len(llm_hyps),
        "n_llm_observations": len(llm_obs),
        "n_llm_dep_entries": len(llm_deps),
        "hyp_matches": {gt: {"llm_id": m[0], "overlap": m[1]}
                        for gt, m in hyp_matches.items()},
        "obs_matches": {gt: {"llm_id": m[0], "overlap": m[1],
                             "llm_content": llm_obs.get(m[0]) if m[0] else None}
                        for gt, m in obs_matches.items()
                        if gt in relevant_gt_obs},
        "dep_scores_naive": {k: v for k, v in dep_naive.items()
                             if k in ("tp", "fp", "fn", "precision", "recall", "f1")},
        "dep_detail_naive": {k: v for k, v in dep_naive.items()
                             if k in ("llm_tuples_in_gt_space", "gt_tuples",
                                      "missed_tuples", "extra_tuples")},
        "dep_scores_fixed": {k: v for k, v in dep_fixed.items()
                             if k in ("tp", "fp", "fn", "precision", "recall", "f1")},
        "dep_detail_fixed": {k: v for k, v in dep_fixed.items()
                             if k in ("llm_tuples_in_gt_space", "gt_tuples",
                                      "missed_tuples", "extra_tuples")},
        "affected_naive": aff_naive,
        "affected_fixed": aff_fixed,
    }
    out_path = Path(path).with_name(
        Path(path).stem + "_dep_eval.json")
    out_path.write_text(json.dumps(out, indent=2, default=list))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python experiments/scripts/eval_dep_extraction.py "
              "<pipeline_output_file.json>",
              file=sys.stderr)
        print("  Example: python experiments/scripts/eval_dep_extraction.py "
              "experiments/results/pipeline_dep_extraction/"
              "pipeline_output_qwen7b_v0.json",
              file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
