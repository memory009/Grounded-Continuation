"""
run_e1c_resolve_update_probe.py — E1c isolated probe.

Question: can a generic Resolve-stage retrospective dependency-update step
recover the post-T13 unification edge h1 → h4, which E1b empirically showed
is structurally unrecoverable through hypothesis-creation-time prompt
engineering alone (0/10 across 4 prompt variants × 3 seeds; see
experiments/results/e1b_ablation/)?

The paper's §4.2 currently asserts (without empirical validation) that
"extending Resolve to update existing Dep tuples (Algorithm 1, Resolve case)"
is the visible fix. This probe directly tests that claim.

Protocol:
  1. Run the existing pipeline on PHASE2_CONVERSATION (all 13 turns).
  2. After each turn whose classification contains at least one Resolve op,
     issue an additional LLM call asking for a generic
     `dependency_updates` over already-existing hypotheses (additions or
     removals to their `depends_on` tuples). The prompt is generic — uses a
     non-Phase-2 example to illustrate format, with no Phase-2 hint.
  3. Record proposed updates per turn. Apply them to a *copy* of the
     pipeline's final engine.dependencies (does not mutate the canonical
     pipeline run).
  4. Score original deps and updated deps separately against the post-T13
     GT (eval_dep_extraction.GT_DEPS), using the same content-aligned
     matching machinery. Report:
       - h1 → h4 recovered before vs after (per seed)
       - Dep precision / recall / F1 before vs after
       - any new false-positive edges introduced by the update step
       - the LLM's proposed updates verbatim (for inspection)
  5. 3 seeds; deterministic pipeline (T=0) + deterministic update prompt (T=0).

GPT-4o only this round. Output to experiments/results/e1c_resolve_update_probe/.
Does not modify pipeline.py, run_experiments.py, or any prior-run JSON.

Usage:
  set -a; source .env.local; set +a
  python run_e1c_resolve_update_probe.py                    # full 3-seed run
  python run_e1c_resolve_update_probe.py --n-seeds 1 --smoke  # 1 seed quick
"""

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from pipeline import EpistemicPipeline, PHASE2_CONVERSATION, call_llm  # noqa: E402
import eval_dep_extraction as ev  # noqa: E402  (sibling in experiments/scripts/)

OUT_DIR_PHASE2 = PROJECT_ROOT / "experiments" / "results" / "e1c_resolve_update_probe"
OUT_DIR_PHASE3 = PROJECT_ROOT / "experiments" / "results" / "e1c_resolve_update_probe_phase3"


# ----------------------------------------------------------------------------
# Phase 3 conversation (parsed from run_experiments.PHASE3_CONVERSATION)
# Phase 3 is a deliberation conversation that ends in a single Resolve at T18.
# No unification edge is expected; this is a negative-sanity check that the
# Resolve-update probe does NOT hallucinate retrospective deps when no
# unification structure is present in the conversation.
# ----------------------------------------------------------------------------

def _load_phase3_conversation():
    """Import PHASE3_CONVERSATION (string form) and parse to [{speaker, text}]."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from run_experiments import PHASE3_CONVERSATION as P3_RAW
    turns = []
    for line in P3_RAW.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: "T<n> -- <Speaker>: <text>"
        m = re.match(r"^T\d+\s*--\s*([^:]+):\s*(.*)$", line)
        if not m:
            continue
        speaker = m.group(1).strip()
        text = m.group(2).strip()
        turns.append({"speaker": speaker, "text": text})
    return turns


PHASE3_CONVERSATION = _load_phase3_conversation()


# ----------------------------------------------------------------------------
# Resolve-stage retrospective dependency-update prompt
# ----------------------------------------------------------------------------

UPDATE_SYSTEM = """\
You are auditing a multi-turn debugging conversation that has been processed by a symbolic epistemic engine. The engine tracks observations, hypotheses, and a dependency map: Dep(hypothesis) -> set of assumption IDs (other hypotheses or observations) that the hypothesis depends on.

A Resolve operation has just been applied to one or more hypotheses on the most recent turn. Your job is to decide whether resolving these hypotheses retrospectively reveals dependency edges that should be added to or removed from any *existing* hypothesis (not necessarily the resolved one). For example, if resolving hypothesis A makes it newly clear that an earlier hypothesis B was implicitly relying on A (or on something A relies on), you would add A to Dep(B). Likewise, if a Resolve obviates a previously-recorded dependency, you may remove an edge.

Most Resolves do NOT require any retrospective updates; an empty list is the correct and common answer. Be conservative — only propose an update if the conversation up to and including this turn provides clear textual support for it.

Return strict JSON, no markdown fences, no commentary:

  {"updates": [
      {"target": "h_id_of_existing_hypothesis",
       "added":   ["assumption_id", ...],
       "removed": ["assumption_id", ...]}
  ], "rationale": "<one sentence explaining the reasoning, or 'no update warranted'>"}

EXAMPLE (a completely different scenario, illustrating the format only):
A team was deciding on an API style. Their earlier hypothesis h_rest ("we should use REST") had Dep(h_rest) = {o_existing_clients}. Later, hypothesis h_readability ("the team prioritises readability over performance") was raised. On a later turn the team Resolved h_rest after the chair pointed out that the earlier h_rest argument was only justified once the readability hypothesis was accepted. The correct retrospective update is:
  {"updates": [{"target": "h_rest", "added": ["h_readability"], "removed": []}], "rationale": "the resolution discussion made the readability hypothesis a load-bearing assumption of h_rest."}

If no retrospective updates are warranted on this turn, return:
  {"updates": [], "rationale": "no retrospective dependency updates warranted; the Resolve simply elevated existing reasoning to accepted conclusion."}
"""


def build_update_prompt(transcript_so_far: str, engine, resolved_targets: list[str],
                        last_turn_speaker: str, last_turn_text: str) -> str:
    """User prompt for the Resolve-stage update probe.

    Provides the conversation so far, the engine state summary, the most
    recent turn's text, and the targets that were just Resolve'd. Asks
    for retrospective updates to existing hypotheses' depends_on.
    """
    # Render engine state compactly.
    obs_lines = []
    for oid, o in engine.observations.items():
        obs_lines.append(f"  {oid} ({o.turn}): {o.content}")
    hyp_lines = []
    for hid, h in engine.hypotheses.items():
        deps = engine.dependencies.get(hid, set())
        deps_str = ("{" + ", ".join(sorted(deps)) + "}") if deps else "{}"
        status = h.status.value if hasattr(h.status, "value") else str(h.status)
        hyp_lines.append(f"  {hid} ({h.turn}, {status}): {h.content}\n     Dep({hid}) = {deps_str}")

    awareness = sorted(engine.awareness - set(engine.observations) - set(engine.hypotheses))
    aw_str = ", ".join(awareness) if awareness else "(none beyond obs/hyp)"

    return f"""\
=== CONVERSATION SO FAR (most recent turn last) ===
{transcript_so_far}

=== ENGINE STATE (after applying the most recent turn's classification) ===
Observations:
{chr(10).join(obs_lines) if obs_lines else '  (none)'}

Hypotheses (with current depends_on):
{chr(10).join(hyp_lines) if hyp_lines else '  (none)'}

Awareness-only entries: {aw_str}

=== JUST-APPLIED RESOLVE EVENT ===
On the most recent turn ({last_turn_speaker}: "{last_turn_text}"), the engine applied Resolve to: {", ".join(resolved_targets)}.

=== TASK ===
Decide whether this resolution retrospectively reveals dependency edges that should be added to or removed from any *existing* hypothesis (target may be the resolved hypothesis itself or any other earlier hypothesis). Return JSON per the system instructions.
"""


def parse_update_response(raw: str) -> dict | None:
    """Parse the LLM JSON for {updates: [...], rationale: ...}."""
    if not raw:
        return None
    raw = raw.strip()
    # Strip markdown fences if any.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the outermost JSON object.
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(d, dict) or "updates" not in d:
        return None
    if not isinstance(d["updates"], list):
        return None
    return d


# ----------------------------------------------------------------------------
# Main per-seed run
# ----------------------------------------------------------------------------

def run_one_seed(api_key: str, model: str, seed: int, verbose: bool = False,
                 conversation: list = None, temperature: float = 0) -> dict:
    """Run pipeline + Resolve-stage update probe; return per-seed result dict.

    When temperature > 0 the seed is a genuine sample: `temperature` is passed
    to every LLM call (pipeline classification AND update probe), and `seed`
    is passed through to the OpenAI sampling-seed parameter so each seed index
    draws a distinct (and individually reproducible-in-expectation) sample.
    At temperature == 0 this reduces to the original deterministic protocol.
    """
    pipeline.BACKEND = "openai"
    pipeline.MODEL = model
    pipeline.API_URL = "https://api.openai.com/v1/chat/completions"

    if conversation is None:
        conversation = PHASE2_CONVERSATION

    pipe = EpistemicPipeline(api_key, verbose=False,
                             temperature=temperature, llm_seed=seed)
    transcript_lines: list[str] = []
    resolve_events: list[dict] = []

    for i, turn in enumerate(conversation):
        turn_id = f"T{i + 1}"
        transcript_lines.append(f"{turn_id} {turn['speaker']}: {turn['text']}")

        result = pipe.process_turn(turn["speaker"], turn["text"])
        if "error" in result:
            return {"seed": seed, "error": f"pipeline failure at {turn_id}: {result['error']}"}

        # Inspect classification: did this turn contain Resolve ops?
        resolutions = result.get("classification", {}).get("resolutions", [])
        if not resolutions:
            continue
        resolved_targets = [r.get("target") for r in resolutions if r.get("target")]
        if not resolved_targets:
            continue

        # Snapshot the engine state immediately AFTER this turn's classification
        # has been applied. We deepcopy hypotheses/observations/dependencies so
        # later turns don't perturb the snapshot we send to the LLM.
        engine_for_prompt = pipe.engine

        transcript_so_far = "\n".join(transcript_lines)
        user_prompt = build_update_prompt(
            transcript_so_far, engine_for_prompt, resolved_targets,
            turn["speaker"], turn["text"],
        )
        raw = call_llm(UPDATE_SYSTEM, user_prompt, api_key,
                       temperature=temperature, seed=seed)
        parsed = parse_update_response(raw or "")

        # Snapshot deps at this resolve moment for the record (deepcopy).
        deps_at_resolve = {h: set(d) for h, d in pipe.engine.dependencies.items()}

        resolve_events.append({
            "turn": turn_id,
            "resolved_targets": resolved_targets,
            "engine_deps_at_resolve": {h: sorted(d) for h, d in deps_at_resolve.items()},
            "llm_raw_response": raw[:2000] if raw else None,
            "llm_parsed_updates": parsed,
        })

        if verbose:
            n_updates = len(parsed["updates"]) if parsed and parsed.get("updates") else 0
            print(f"  [seed {seed}] {turn_id}: Resolve({', '.join(resolved_targets)}) -> "
                  f"{n_updates} update(s) proposed")

    # Apply all proposed updates to a COPY of the final engine deps.
    deps_before = {h: set(d) for h, d in pipe.engine.dependencies.items()}
    deps_after = copy.deepcopy(deps_before)

    valid_hyp_ids = set(pipe.engine.hypotheses.keys())
    valid_obs_ids = set(pipe.engine.observations.keys())
    valid_targets = valid_hyp_ids | valid_obs_ids

    applied_updates: list[dict] = []
    skipped_updates: list[dict] = []
    for ev_entry in resolve_events:
        parsed = ev_entry.get("llm_parsed_updates")
        if not parsed or not parsed.get("updates"):
            continue
        for upd in parsed["updates"]:
            target = upd.get("target")
            added = upd.get("added") or []
            removed = upd.get("removed") or []
            if not target:
                continue
            if target not in valid_hyp_ids:
                # The LLM is only allowed to update existing hypotheses.
                skipped_updates.append({
                    "turn": ev_entry["turn"], "reason": "target not an existing hypothesis",
                    "update": upd,
                })
                continue
            cur = deps_after.setdefault(target, set())
            for a in removed:
                cur.discard(a)
            for a in added:
                if a in valid_targets and a != target:
                    cur.add(a)
                else:
                    skipped_updates.append({
                        "turn": ev_entry["turn"],
                        "reason": "added id not in engine or self-ref",
                        "target": target, "added_id": a,
                    })
            applied_updates.append({
                "turn": ev_entry["turn"], "target": target,
                "added_applied": [a for a in added if a in valid_targets and a != target],
                "removed_applied": removed,
            })

    return {
        "seed": seed,
        "model": model,
        "temperature": temperature,
        "deps_before": {h: sorted(d) for h, d in deps_before.items()},
        "deps_after": {h: sorted(d) for h, d in deps_after.items()},
        "resolve_events": resolve_events,
        "applied_updates": applied_updates,
        "skipped_updates": skipped_updates,
        "n_hypotheses_extracted": len(pipe.engine.hypotheses),
        "hypotheses": {hid: {"content": h.content,
                             "status": h.status.value if hasattr(h.status, "value") else str(h.status)}
                       for hid, h in pipe.engine.hypotheses.items()},
        "observations": {oid: o.content for oid, o in pipe.engine.observations.items()},
    }


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def score_seed(seed_result: dict) -> dict:
    """Score deps_before and deps_after against post-T13 GT, with content alignment."""
    if "error" in seed_result:
        return {"error": seed_result["error"]}

    # Wrap hypotheses to look like {hid: {"content": ...}} as expected by match_llm_to_gt.
    llm_hyps = {hid: {"content": h["content"]} for hid, h in seed_result["hypotheses"].items()}
    hyp_matches = ev.match_llm_to_gt(llm_hyps, ev.GT_H_CONTENT)
    obs_matches = ev.match_llm_obs_to_gt(seed_result["observations"], ev.GT_O_CONTENT)

    def deps_to_set(deps: dict[str, list[str]]) -> dict[str, set[str]]:
        return {h: set(d) for h, d in deps.items()}

    score_before = ev.score_dep(deps_to_set(seed_result["deps_before"]),
                                hyp_matches, obs_matches, passthrough_obs=False)
    score_after = ev.score_dep(deps_to_set(seed_result["deps_after"]),
                               hyp_matches, obs_matches, passthrough_obs=False)

    # h1 -> h4 recovery: check if (h1, h4) tuple is in the GT-projected tuple set.
    def has_h1_h4(score: dict) -> bool:
        return ("h1", "h4") in {tuple(t) for t in score.get("llm_tuples_in_gt_space", [])}

    return {
        "hyp_matches": {gt: {"llm_id": v[0], "score": v[1]}
                        for gt, v in hyp_matches.items()},
        "obs_matches": {gt: {"llm_id": v[0], "score": v[1]}
                        for gt, v in obs_matches.items()},
        "before": {**score_before, "h1_to_h4_recovered": has_h1_h4(score_before)},
        "after":  {**score_after,  "h1_to_h4_recovered": has_h1_h4(score_after)},
        "delta": {
            "f1": round(score_after["f1"] - score_before["f1"], 4),
            "precision": round(score_after["precision"] - score_before["precision"], 4),
            "recall": round(score_after["recall"] - score_before["recall"], 4),
            "tp_delta": score_after["tp"] - score_before["tp"],
            "fp_delta": score_after["fp"] - score_before["fp"],
            "fn_delta": score_after["fn"] - score_before["fn"],
            "new_extra_tuples": sorted(set(map(tuple, score_after["extra_tuples"]))
                                       - set(map(tuple, score_before["extra_tuples"]))),
            "newly_recovered_tuples": sorted(set(map(tuple, score_after["llm_tuples_in_gt_space"]))
                                             - set(map(tuple, score_before["llm_tuples_in_gt_space"]))),
            "newly_lost_tuples": sorted(set(map(tuple, score_before["llm_tuples_in_gt_space"]))
                                        - set(map(tuple, score_after["llm_tuples_in_gt_space"]))),
        },
    }


def aggregate(per_seed: list[dict], scored: list[dict]) -> dict:
    """Aggregate across seeds: median F1 before/after, h1→h4 recovery counts, all FP edges added."""
    n = len(per_seed)
    h1_h4_before = sum(1 for s in scored if s.get("before", {}).get("h1_to_h4_recovered"))
    h1_h4_after = sum(1 for s in scored if s.get("after", {}).get("h1_to_h4_recovered"))

    f1_before = sorted(s["before"]["f1"] for s in scored if "before" in s)
    f1_after = sorted(s["after"]["f1"] for s in scored if "after" in s)
    rec_before = sorted(s["before"]["recall"] for s in scored if "before" in s)
    rec_after = sorted(s["after"]["recall"] for s in scored if "after" in s)
    prec_before = sorted(s["before"]["precision"] for s in scored if "before" in s)
    prec_after = sorted(s["after"]["precision"] for s in scored if "after" in s)

    def median(xs):
        if not xs:
            return None
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else round((xs[m-1] + xs[m]) / 2, 4)

    new_fps = []
    for s in scored:
        for t in s.get("delta", {}).get("new_extra_tuples", []):
            new_fps.append(tuple(t))
    fp_counter = defaultdict(int)
    for fp in new_fps:
        fp_counter[fp] += 1

    return {
        "n_seeds": n,
        "h1_to_h4_recovered_before": f"{h1_h4_before}/{n}",
        "h1_to_h4_recovered_after": f"{h1_h4_after}/{n}",
        "median_dep_f1_before": median(f1_before),
        "median_dep_f1_after": median(f1_after),
        "median_dep_precision_before": median(prec_before),
        "median_dep_precision_after": median(prec_after),
        "median_dep_recall_before": median(rec_before),
        "median_dep_recall_after": median(rec_after),
        "all_dep_f1_before": f1_before,
        "all_dep_f1_after": f1_after,
        "new_false_positive_edges_by_count": [{"edge": list(e), "n_seeds": c}
                                              for e, c in sorted(fp_counter.items(),
                                                                 key=lambda x: -x[1])],
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def aggregate_phase3_negative(per_seed: list[dict]) -> dict:
    """Phase-3 negative-sanity aggregate: count empty-update events and any
    new edges added (each is by construction a false positive since no
    canonical retrospective edge exists in Phase 3)."""
    n = len(per_seed)
    n_resolve_events = 0
    n_empty_update_events = 0
    n_nonempty_update_events = 0
    n_seeds_clean = 0  # seeds with no edges added at all
    n_edges_added_total = 0
    edges_added_per_seed: list[list] = []
    for sr in per_seed:
        if "error" in sr:
            edges_added_per_seed.append([])
            continue
        before = {h: set(d) for h, d in sr["deps_before"].items()}
        after = {h: set(d) for h, d in sr["deps_after"].items()}
        # Edges added in the LLM namespace.
        added = []
        for h, d in after.items():
            for x in d - before.get(h, set()):
                added.append([h, x])
        edges_added_per_seed.append(sorted(added))
        if not added:
            n_seeds_clean += 1
        n_edges_added_total += len(added)
        for ev_entry in sr["resolve_events"]:
            n_resolve_events += 1
            parsed = ev_entry.get("llm_parsed_updates")
            if parsed and parsed.get("updates"):
                n_nonempty_update_events += 1
            else:
                n_empty_update_events += 1
    return {
        "n_seeds": n,
        "n_seeds_clean_no_edges_added": f"{n_seeds_clean}/{n}",
        "n_resolve_events_total": n_resolve_events,
        "n_empty_update_events": n_empty_update_events,
        "n_nonempty_update_events": n_nonempty_update_events,
        "n_edges_added_total_across_seeds": n_edges_added_total,
        "edges_added_per_seed": edges_added_per_seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--smoke", action="store_true",
                   help="Quick 1-seed run with verbose output.")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--phase", type=int, choices=[2, 3], default=2,
                   help="2 = Phase 2 (default; existing E1c probe with GT scoring). "
                        "3 = Phase 3 negative sanity (no GT, just check empty updates / no edges added).")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory. Default: phase-specific dir.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for ALL LLM calls (pipeline "
                        "classification + update probe). Default 0 = original "
                        "deterministic E1c protocol. Use 0.7 for the scale-up "
                        "so seeds are genuine samples, not pseudo-replicates.")
    p.add_argument("--seed-start", type=int, default=0,
                   help="First seed index (inclusive). Runs seeds "
                        "[seed_start, seed_start + n_seeds).")
    p.add_argument("--output", default=None,
                   help="Explicit output JSON file path (overrides "
                        "--out-dir + default filename).")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY (E1c uses GPT-4o).")

    n_seeds = 1 if args.smoke else args.n_seeds

    if args.phase == 2:
        conversation = PHASE2_CONVERSATION
        default_out_dir = OUT_DIR_PHASE2
        result_filename = f"e1c_resolve_update_probe{'_smoke' if args.smoke else ''}.json"
    else:
        conversation = PHASE3_CONVERSATION
        default_out_dir = OUT_DIR_PHASE3
        result_filename = f"e1c_resolve_update_probe_phase3{'_smoke' if args.smoke else ''}.json"

    if args.output:
        out_path_override = Path(args.output)
        out_dir = out_path_override.parent
    else:
        out_path_override = None
        out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + n_seeds))

    print(f"E1c — Resolve-stage retrospective dep-update probe (Phase {args.phase})")
    print(f"  model: {args.model}, seeds: {seeds}, smoke: {args.smoke}, "
          f"temperature: {args.temperature}")
    print(f"  conversation turns: {len(conversation)}")
    print(f"  out: {out_path_override or out_dir}")
    print()

    per_seed_results: list[dict] = []
    scored: list[dict] = []
    for seed in seeds:
        print(f"[seed {seed}] running pipeline + resolve-update probe...")
        sr = run_one_seed(api_key, args.model, seed, verbose=args.smoke,
                          conversation=conversation,
                          temperature=args.temperature)
        per_seed_results.append(sr)
        if "error" in sr:
            print(f"  ERROR: {sr['error']}")
            scored.append({"error": sr["error"]})
            continue

        if args.phase == 2:
            sc = score_seed(sr)
            scored.append(sc)
            b, a = sc.get("before", {}), sc.get("after", {})
            print(f"  before: F1={b.get('f1')}, P={b.get('precision')}, R={b.get('recall')}, "
                  f"h1->h4 recovered={b.get('h1_to_h4_recovered')}")
            print(f"  after:  F1={a.get('f1')}, P={a.get('precision')}, R={a.get('recall')}, "
                  f"h1->h4 recovered={a.get('h1_to_h4_recovered')}")
            nft = sc.get("delta", {}).get("new_extra_tuples", [])
            if nft:
                print(f"  new FP edges introduced by update: {nft}")
        else:
            # Phase 3: no canonical GT for retrospective deps. Report only
            # raw before/after edge sets and counts.
            n_resolves = len(sr["resolve_events"])
            n_nonempty = sum(1 for e in sr["resolve_events"]
                             if e.get("llm_parsed_updates") and e["llm_parsed_updates"].get("updates"))
            edges_before = sum(len(d) for d in sr["deps_before"].values())
            edges_after = sum(len(d) for d in sr["deps_after"].values())
            edges_added = edges_after - edges_before
            print(f"  resolve_events: {n_resolves}; nonempty_updates: {n_nonempty}; "
                  f"edges_before: {edges_before}; edges_after: {edges_after}; added: {edges_added}")
            scored.append({"phase": 3, "edges_before": edges_before,
                           "edges_after": edges_after, "edges_added": edges_added,
                           "n_resolve_events": n_resolves,
                           "n_nonempty_update_events": n_nonempty})

    print()
    print("=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    if args.phase == 2:
        summary = aggregate(per_seed_results, scored)
    else:
        summary = aggregate_phase3_negative(per_seed_results)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = out_path_override if out_path_override else (out_dir / result_filename)
    out_path.write_text(json.dumps({
        "experiment": f"E1c — Resolve-stage retrospective dep-update probe (Phase {args.phase})",
        "phase": args.phase,
        "model": args.model,
        "n_seeds": n_seeds,
        "seed_start": args.seed_start,
        "seeds": seeds,
        "temperature": args.temperature,
        "smoke": args.smoke,
        "summary": summary,
        "usage_log": pipeline.USAGE_LOG,
        "per_seed_results": per_seed_results,
        "per_seed_scores": scored,
    }, indent=2, default=str))
    print(f"\nusage: {json.dumps(pipeline.USAGE_LOG)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
