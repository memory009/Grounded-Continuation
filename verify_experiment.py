"""
verify_experiment.py — direct verifier evaluation harness (E2).

For each (truncation_t, candidate, asserts_id) item in
experiments/e2_verify/verify_test_set_phase2.yaml, this script:

  1. Builds D_t by replaying the canonical pipeline.PHASE2_CONVERSATION
     turns 1..t through the LLM pipeline (cached per t).
  2. Resolves the canonical asserts_id (h1..h4 / o1..o9 / mis_monitor)
     to the LLM-run-specific ID by content alignment, reusing the
     same matchers from experiments/scripts/eval_dep_extraction.py.
  3. Runs Verify(c, D_t): structural reachability over engine.dependencies,
     starting from the resolved asserts_id, returning grounded iff every
     leaf is an observation or active/resolved hypothesis (or awareness
     entry for the mis_monitor case).
  4. Runs an LLM-only baseline: full transcript 1..t + candidate, ask
     for a one-word grounded/ungrounded judgement.
  5. Scores both against the test-set author's label and aggregates
     accuracy by category, including the headline pooled
     stale+counterfactual delta.

GPT-4o-only this round (per locked decision 2026-04-27).

Smoke-test caveat: as long as Cohen's κ from the independent annotator
has not been computed, treat results as code-correctness validation
only — the test-set labels are author-only at that point.

Usage:
  export OPENAI_API_KEY=...

  # Smoke (5 items, ~3-5 min, ~$0.50): one item per category + an extra
  python verify_experiment.py --max-items 5 --output-suffix _smoke

  # Full (50 items, ~30-40 min, ~$4 incl. baseline)
  python verify_experiment.py
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

import pipeline
from pipeline import EpistemicPipeline, PHASE2_CONVERSATION

# Reuse the battle-tested content-alignment matchers from the existing
# eval pipeline (these produced the published GPT-4o tab:end-to-end nums).
EVAL_DIR = Path(__file__).resolve().parent / "experiments" / "scripts"
sys.path.insert(0, str(EVAL_DIR))
import eval_dep_extraction as ev  # noqa: E402

# Extended canonical-observation content map — broader than
# eval_dep_extraction.GT_O_CONTENT (which only covers o5/o6/o8/o9, the
# subset referenced by GT_DEPS). The E2 test set references all nine
# Phase-2 observations plus the T4 awareness item, so we need fuller
# coverage. This dict is used for asserts_id resolution only; the existing
# eval_dep_extraction matchers stay untouched.
EXTENDED_OBS_CONTENT = {
    "o1": "auth failure rate alert spike",
    "o2": "payment failure rate alert spike",
    "o3": "database health degradation alert",
    "o4": "customer complaints around 2am 215 incident start",
    "o5": "token expired 401 unauthorized",
    "o6": "auth traffic 3x normal anomalous volume request",
    "o7": "stripe 429 too many requests rate limit",
    "o8": "redis connection timeouts pool exhausted payment",
    "o9": "error code 401 token expired distinct 503 session lookup failed",
}

# Canonical awareness items. Used to content-match Expand-Awareness events
# in engine.history. Lowered to a min-score-1 fuzzy match because awareness
# events tend to use short phrasings.
CANONICAL_AWARENESS_CONTENT = {
    "mis_monitor": "database alert redis primary monitoring misclassified",
}
AWARENESS_MIN_SCORE = 1


def build_d_t(api_key, truncate_at_turn, ablate_layers=None):
    """Replay PHASE2_CONVERSATION turns 1..truncate_at_turn through the
    pipeline and return the EpistemicPipeline (engine inside).

    ablate_layers="del_awareness" builds D_t with the DEL plausibility and
    awareness layers disabled (ablation): the extraction
    interface loses Support and Expand-Awareness, the engine's awareness set
    stays empty, and plausibility is never soft-upgraded. The verify-time
    decision rule is otherwise unchanged; because engine.awareness is empty
    in the ablated D_t, the awareness clauses in resolve_canonical_to_llm /
    verifier_judgement / walk_deps can never fire, i.e. Verify runs on the
    pure argument/dependency skeleton (nodes + status + Dep edges)."""
    pipe = EpistemicPipeline(
        api_key, verbose=False,
        ablate_del_awareness=(ablate_layers == "del_awareness"))
    n = min(truncate_at_turn, len(PHASE2_CONVERSATION))
    for i in range(n):
        turn = PHASE2_CONVERSATION[i]
        try:
            pipe.process_turn(turn["speaker"], turn["text"])
        except Exception as e:
            print(f"    WARN: turn {i+1} failed: {e}")
    return pipe


def _best_content_match(target_text, candidates_id_to_text, min_overlap=0.15):
    """Many-to-one content match: given target text and a dict of
    {entity_id: text}, return the (entity_id, score) with highest
    Jaccard-ish overlap (per ev.token_overlap), or (None, 0.0) if none
    exceeds min_overlap. Independent per call — does not consume entities.
    """
    best_id, best_score = None, 0.0
    for cid, ctext in candidates_id_to_text.items():
        score = ev.token_overlap(target_text, ctext)
        if score > best_score:
            best_score, best_id = score, cid
    if best_score < min_overlap:
        return None, best_score
    return best_id, best_score


def resolve_canonical_to_llm(canonical_id, engine):
    """Map a canonical asserts_id (from the test set) to the LLM-run-specific
    ID present in `engine`. Returns (llm_id_or_None, kind) where kind is one
    of {'observation', 'hypothesis', 'awareness', None}.

    Many-to-one resolution: each canonical id resolves independently to its
    best content-overlap match in `engine`, ignoring whether other canonical
    ids also claim the same LLM entity. This matters because the LLM
    pipeline routinely collapses several canonical entities into a single
    LLM observation (e.g. T1's three alerts → one LLM obs); under the
    bipartite matcher in eval_dep_extraction.py one of the canonical ids
    would be lost. For verifier judgement we want every canonical id to
    have a chance to find its content in D_t.

    For asserts_id like 'mis_monitor' (an awareness expansion), we also
    search engine.observations and engine.hypotheses content, because
    LLM classifications vary turn-by-turn (e.g. T4 may be tagged as Observe
    even though the canonical operation is Expand-Awareness).
    """
    if canonical_id is None:
        return None, None

    if canonical_id.startswith("h"):
        target = ev.GT_H_CONTENT.get(canonical_id, "")
        hyp_texts = {hid: h.content for hid, h in engine.hypotheses.items()}
        m_id, _ = _best_content_match(target, hyp_texts, min_overlap=0.15)
        return m_id, "hypothesis"

    if canonical_id.startswith("o"):
        target = EXTENDED_OBS_CONTENT.get(canonical_id, "")
        obs_texts = {oid: o.content for oid, o in engine.observations.items()}
        m_id, _ = _best_content_match(target, obs_texts, min_overlap=0.10)
        return m_id, "observation"

    # Awareness item: search Expand-Awareness events first, then fall back
    # to observations + hypotheses (the LLM may have classified the
    # awareness expansion as Observe or Hypothesize instead).
    if canonical_id in CANONICAL_AWARENESS_CONTENT:
        target = CANONICAL_AWARENESS_CONTENT[canonical_id]
        aw_texts = {}
        for h_event in getattr(engine, "history", []):
            if getattr(h_event, "operation", None) != "Expand-Awareness":
                continue
            details = getattr(h_event, "details", {}) or {}
            aw_id = details.get("id")
            content = details.get("content", "")
            if aw_id in engine.awareness:
                aw_texts[aw_id] = content
        m_id, score = _best_content_match(target, aw_texts, min_overlap=0.10)
        if m_id is not None:
            return m_id, "awareness"
        # Fall back: maybe the LLM classified it as Observe or Hypothesize.
        all_texts = {oid: o.content for oid, o in engine.observations.items()}
        all_texts.update({hid: h.content for hid, h in engine.hypotheses.items()})
        m_id, score = _best_content_match(target, all_texts, min_overlap=0.15)
        if m_id is not None:
            kind = "observation" if m_id in engine.observations else "hypothesis"
            return m_id, kind
        return None, "awareness"

    return None, None


def walk_deps(engine, node, visited=None):
    """BFS through engine.dependencies starting at `node`. Return the list
    of nodes traversed if every leaf is an observation, awareness entry,
    or active/resolved hypothesis. Return None if the chain hits an
    abandoned/weakened hypothesis or a missing entity.
    """
    if visited is None:
        visited = set()
    if node in visited:
        return []
    visited.add(node)

    if node in engine.observations:
        return [node]

    if node in engine.hypotheses:
        h = engine.hypotheses[node]
        status = h.status.value if hasattr(h.status, "value") else str(h.status)
        if status in ("abandoned", "weakened"):
            return None
        deps = engine.dependencies.get(node, set())
        chain = [node]
        for dep in deps:
            sub = walk_deps(engine, dep, visited)
            if sub is None:
                return None
            chain.extend(sub)
        return chain

    if node in getattr(engine, "awareness", set()):
        return [node]

    return None


def verifier_judgement(engine, llm_asserts_id, kind):
    """Run Verify(c, D_t). Return ('grounded'|'ungrounded', reason_chain)."""
    if llm_asserts_id is None:
        if kind is None:
            return "ungrounded", ["asserts_id is null (no entity in D_t matches the claim)"]
        return "ungrounded", [f"asserted {kind} not present in D_t (no content-aligned LLM ID)"]

    if llm_asserts_id in engine.observations:
        return "grounded", [f"{llm_asserts_id} (observation) — leaf in D_t"]

    if llm_asserts_id in engine.hypotheses:
        h = engine.hypotheses[llm_asserts_id]
        status = h.status.value if hasattr(h.status, "value") else str(h.status)
        if status in ("abandoned", "weakened"):
            return ("ungrounded",
                    [f"{llm_asserts_id} status={status} (hypothesis no longer in good standing)"])
        chain = walk_deps(engine, llm_asserts_id)
        if chain is None:
            return ("ungrounded",
                    [f"{llm_asserts_id} dep chain reaches an abandoned entity"])
        return "grounded", chain

    if llm_asserts_id in getattr(engine, "awareness", set()):
        return "grounded", [f"{llm_asserts_id} (awareness) — present in D_t"]

    return "ungrounded", [f"asserted entity '{llm_asserts_id}' not present in D_t"]


def baseline_judgement(api_key, truncate_at_turn, candidate):
    """LLM-only baseline: full transcript 1..t + candidate; one-word answer."""
    transcript = "\n".join(
        f"T{i+1} {turn['speaker']}: {turn['text']}"
        for i, turn in enumerate(PHASE2_CONVERSATION[:truncate_at_turn])
    )
    user_prompt = (
        f"Below is a multi-turn debugging conversation up to turn {truncate_at_turn}. "
        f"After it, a candidate next utterance is proposed. Decide whether the "
        f"candidate is GROUNDED (its claim follows from the prior conversation, "
        f"including respecting any hypotheses that were abandoned or revised) "
        f"or UNGROUNDED (its claim contradicts the prior conversation, asserts "
        f"something abandoned, or comes from outside the conversation).\n\n"
        f"=== CONVERSATION ===\n{transcript}\n=== END CONVERSATION ===\n\n"
        f"Candidate next utterance: {candidate}\n\n"
        f"Answer with one word: GROUNDED or UNGROUNDED."
    )
    resp = pipeline.call_llm(
        system="You are a careful conversation analyst.",
        user=user_prompt, api_key=api_key, temperature=0,
    )
    if not resp:
        return "ungrounded", "(no response)"
    s = resp.strip().upper()
    # Prioritize the first explicit label so a later "...not grounded..."
    # in a free-form explanation does not flip an UNGROUNDED verdict.
    # Check UNGROUNDED before GROUNDED at every level (UNGROUNDED's prefix
    # is "U", not "G", so prefix order does not matter, but order matters in
    # the token-fallback loop).
    if s.startswith("UNGROUNDED"):
        return "ungrounded", resp.strip()
    if s.startswith("GROUNDED"):
        return "grounded", resp.strip()
    for tok in s.split():
        if tok.startswith("UNGROUNDED"):
            return "ungrounded", resp.strip()
        if tok.startswith("GROUNDED"):
            return "grounded", resp.strip()
    return "ungrounded", resp.strip()


def select_smoke_subset(items, max_items):
    """Pick a stratified subset for smoke testing: at least one item per
    category, then fill in by item-id order."""
    if max_items is None or max_items >= len(items):
        return items
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    pick = []
    seen_ids = set()
    # Round 1: one per category, in fixed category order
    for cat in ("actual", "stale", "cross_conv", "counterfactual"):
        if by_cat[cat] and len(pick) < max_items:
            chosen = by_cat[cat][0]
            pick.append(chosen)
            seen_ids.add(chosen["id"])
    # Round 2: fill in remaining slots by source order
    for it in items:
        if len(pick) >= max_items:
            break
        if it["id"] not in seen_ids:
            pick.append(it)
            seen_ids.add(it["id"])
    return pick[:max_items]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-set",
                   default="experiments/e2_verify/verify_test_set_phase2.yaml")
    p.add_argument("--max-items", type=int, default=None,
                   help="Smoke-test on a stratified subset of N items.")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-verifier", action="store_true")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--base-url",
                   default="https://api.openai.com/v1/chat/completions")
    p.add_argument("--out-dir", default="experiments/results/e2_verify")
    p.add_argument("--output-suffix", default="",
                   help="Append suffix to output filename "
                        "(e.g. '_smoke' for code-correctness tests).")
    p.add_argument("--ablate-layers", choices=["del_awareness"], default=None,
                   help="Ablation: 'del_awareness' disables "
                        "the DEL plausibility layer (Support soft-upgrades) "
                        "and the awareness layer (Expand-Awareness / "
                        "awareness set) in BOTH the extraction interface and "
                        "the engine, keeping only the argument/dependency "
                        "skeleton (hypothesis nodes, status lifecycle, Dep "
                        "map, undermine edges).")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY (E2 uses GPT-4o).")

    pipeline.BACKEND = "openai"
    pipeline.MODEL = args.model
    pipeline.API_URL = args.base_url

    test_set = yaml.safe_load(Path(args.test_set).read_text())
    items = test_set.get("items") if isinstance(test_set, dict) else test_set
    if not items:
        sys.exit(f"No items found in {args.test_set}")

    items = select_smoke_subset(items, args.max_items)
    print(f"Loaded {len(items)} items from {args.test_set}"
          + (f" (smoke subset of {test_set.get('n_items', '?')})" if args.max_items else ""))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"e2_verify_results{args.output_suffix}.json"

    pipe_cache = {}
    results = []

    for item in items:
        t = item["truncation_t"]
        cand = item["candidate"]
        asserts_id = item.get("asserts_id")
        gt = item["label"]
        cat = item["category"]
        item_id = item["id"]

        if t not in pipe_cache and not args.skip_verifier:
            print(f"\nBuilding D_t for t={t} (one-time per truncation"
                  + (f", ablate={args.ablate_layers}" if args.ablate_layers else "")
                  + ")...")
            pipe_cache[t] = build_d_t(api_key, t, ablate_layers=args.ablate_layers)
            engine = pipe_cache[t].engine
            print(f"  D_{t}: {len(engine.observations)} obs, "
                  f"{len(engine.hypotheses)} hyps "
                  f"({sum(1 for h in engine.hypotheses.values() if h.status.value == 'abandoned')} abandoned), "
                  f"{len(engine.awareness)} awareness")

        verifier_ans = verifier_chain = None
        resolved_llm = resolved_kind = None
        if not args.skip_verifier:
            engine = pipe_cache[t].engine
            resolved_llm, resolved_kind = resolve_canonical_to_llm(asserts_id, engine)
            verifier_ans, verifier_chain = verifier_judgement(engine, resolved_llm, resolved_kind)

        baseline_ans = baseline_raw = None
        if not args.skip_baseline:
            baseline_ans, baseline_raw = baseline_judgement(api_key, t, cand)

        v_correct = (verifier_ans == gt) if verifier_ans else None
        b_correct = (baseline_ans == gt) if baseline_ans else None

        results.append({
            "id": item_id,
            "category": cat,
            "truncation_t": t,
            "candidate": cand,
            "canonical_asserts_id": asserts_id,
            "resolved_llm_asserts_id": resolved_llm,
            "resolved_kind": resolved_kind,
            "ground_truth_label": gt,
            "verifier": verifier_ans,
            "verifier_chain": verifier_chain,
            "verifier_correct": v_correct,
            "baseline": baseline_ans,
            "baseline_raw_response": baseline_raw,
            "baseline_correct": b_correct,
        })

        marker_v = "OK " if v_correct else ("X  " if v_correct is False else "-- ")
        marker_b = "OK " if b_correct else ("X  " if b_correct is False else "-- ")
        print(f"  [{item_id}/{cat}/t={t}] gt={gt:<10}  "
              f"V={str(verifier_ans):<10}{marker_v}  "
              f"B={str(baseline_ans):<10}{marker_b}")

    # Aggregate by category.
    by_cat = defaultdict(lambda: {"v_correct": 0, "v_total": 0,
                                  "b_correct": 0, "b_total": 0})
    for r in results:
        c = r["category"]
        if r["verifier"] is not None:
            by_cat[c]["v_total"] += 1
            by_cat[c]["v_correct"] += int(r["verifier_correct"])
        if r["baseline"] is not None:
            by_cat[c]["b_total"] += 1
            by_cat[c]["b_correct"] += int(r["baseline_correct"])

    summary = {}
    for c, d in by_cat.items():
        v_acc = d["v_correct"] / d["v_total"] if d["v_total"] else 0.0
        b_acc = d["b_correct"] / d["b_total"] if d["b_total"] else 0.0
        summary[c] = {
            "n": max(d["v_total"], d["b_total"]),
            "verifier_correct": d["v_correct"],
            "verifier_total": d["v_total"],
            "verifier_acc": v_acc,
            "baseline_correct": d["b_correct"],
            "baseline_total": d["b_total"],
            "baseline_acc": b_acc,
            "delta": v_acc - b_acc,
        }

    # Pooled stale + counterfactual (the headline analysis from EXPERIMENTS.md).
    pooled_v = pooled_v_total = pooled_b = pooled_b_total = 0
    for r in results:
        if r["category"] not in ("stale", "counterfactual"):
            continue
        if r["verifier"] is not None:
            pooled_v_total += 1
            pooled_v += int(r["verifier_correct"])
        if r["baseline"] is not None:
            pooled_b_total += 1
            pooled_b += int(r["baseline_correct"])
    if pooled_v_total or pooled_b_total:
        v_acc = pooled_v / pooled_v_total if pooled_v_total else 0.0
        b_acc = pooled_b / pooled_b_total if pooled_b_total else 0.0
        summary["stale+counterfactual_pooled"] = {
            "n": max(pooled_v_total, pooled_b_total),
            "verifier_correct": pooled_v,
            "verifier_total": pooled_v_total,
            "verifier_acc": v_acc,
            "baseline_correct": pooled_b,
            "baseline_total": pooled_b_total,
            "baseline_acc": b_acc,
            "delta": v_acc - b_acc,
        }

    out = {
        "experiment": "E2 — direct verifier evaluation harness",
        "model": args.model,
        "backend": "openai",
        "ablate_layers": args.ablate_layers,
        "test_set": args.test_set,
        "n_items_run": len(results),
        "n_items_in_test_set": len(test_set.get("items", [])) if isinstance(test_set, dict) else len(test_set),
        "smoke_caveat": (
            "If --max-items was used, this is a smoke run for code "
            "correctness only. Final paper-quality results require the "
            "full test set AND an independent annotator's labels with "
            "Cohen's κ ≥ 0.7 on the 20-item overlap (see "
            "experiments/e2_verify/blinded_to_source_mapping.yaml)."
        ),
        "per_item": results,
        "summary": summary,
    }
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 78)
    print("E2 SUMMARY")
    print("=" * 78)
    print(f"{'Category':<35} {'n':>4} {'Verifier':>14} {'LLM-only':>14} {'Δ':>8}")
    cat_order = ["actual", "stale", "cross_conv", "counterfactual",
                 "stale+counterfactual_pooled"]
    for c in cat_order:
        if c not in summary:
            continue
        s = summary[c]
        v_str = f"{s['verifier_correct']}/{s['verifier_total']}={s['verifier_acc']:.2%}"
        b_str = f"{s['baseline_correct']}/{s['baseline_total']}={s['baseline_acc']:.2%}"
        print(f"{c:<35} {s['n']:>4} {v_str:>14} {b_str:>14} {s['delta']:>+8.2%}")

    if args.max_items:
        print("\n  SMOKE: code-correctness validation only.")
        print("  Final E2 result requires the full test set and an")
        print("  independent annotator's κ ≥ 0.7 on the 20-item overlap.")


if __name__ == "__main__":
    main()
