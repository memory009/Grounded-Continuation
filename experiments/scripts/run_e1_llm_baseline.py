"""
run_e1_llm_baseline.py — Direct-LLM-prompted dependency extraction baseline (E1).

For each of 4 Phase-2 hypotheses, ask the LLM to list the prior turns the
hypothesis depends on, in four prompt variants:
  (a) zero-shot
  (b) few-shot (one Phase-1 worked example)
  (c) chain-of-thought
  (d) self-consistency over 5 samples (CoT, majority vote per dependency edge)

GPT-4o-only this round (per locked decision 2026-04-27).

Outputs experiments/results/e1_llm_baseline/e1_llm_baseline.json with
per-variant precision/recall/F1 against ground-truth Dep tuples.

Usage:
  export OPENAI_API_KEY=...
  python run_e1_llm_baseline.py                          # all variants
  python run_e1_llm_baseline.py --variant zero-shot      # smoke
"""

import os
import sys
import json
import argparse
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline as plib  # noqa: E402

PHASE2_TRANSCRIPT = """T1 Carol: P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation.
T2 Alice: 401 Unauthorized spikes. Tokens rejected as expired. Auth traffic 3x normal. Strange.
T3 Bob: Stripe returning 429s. Redis connection timeouts from Payment Service.
T4 Carol: Database alert is about Redis, not the primary DB.
T5 Bob: Connection pool exhausted. Rate-limit checks failing. Requests go to Stripe unthrottled.
T6 Carol: Redis sick -> Payment loses rate limiting -> Stripe hammered. Redis -> Auth too?
T7 Alice: Auth uses Redis for sessions. But error is 'token expired' (401), not 'session lookup failed' (503).
T8 Carol: Auth failures might not be caused by Redis after all?
T9 Alice: Error type is wrong. Token expiry is a different problem from Redis.
T10 Bob: Why is auth traffic 3x if token expiry is independent of Redis?
T11 Alice: Frontend retrying on token expired. Retry loop generates amplified traffic.
T12 Carol: Retry storm from token bug exhausting Redis? Auth->Redis, not Redis->Auth?
T13 Alice: Plausible. 3x traffic x Redis lookups = pool exhaustion. Whole thing cascades from token bug."""

GROUND_TRUTH_DEP = {
    "h1": {"o8", "h4"},
    "h2": {"o8"},
    "h3": {"o5"},
    "h4": {"o9", "o6", "h3"},
}

HYPOTHESIS_TEXTS = {
    "h1": "Redis pool exhaustion causes rate-limit bypass causing Stripe 429s",
    "h2": "Redis failure causes auth failures via shared session cache",
    "h3": "Token bug causes a frontend retry loop amplifying traffic 3x",
    "h4": "Token bug causes retry storm causing Redis pool exhaustion (causal reversal of h2)",
}

TURN_TO_OBS = {
    "T1":  ["o1", "o2", "o3", "o4"],
    "T2":  ["o5", "o6"],
    "T3":  ["o7", "o8"],
    "T4":  ["mis_monitor"],
    "T5":  ["o8b", "h1"],
    "T6":  ["h2"],
    "T7":  ["o9"],
    "T11": ["h3"],
    "T12": ["h4"],
}

PROMPT_BASE = """You are analyzing a multi-turn debugging conversation. Given a hypothesis the team reached, list the prior conversation turns whose content this hypothesis directly depends on.

CONVERSATION:
{transcript}

HYPOTHESIS: {hypothesis_text}

Return ONLY a JSON object of the form:
{{"depends_on_turns": ["T5", "T7", ...]}}

Include only turns whose content the hypothesis directly relies on. Do not include indirect or chained dependencies."""

PROMPT_COT_SUFFIX = """

Think step by step before giving your answer:
1. What does the hypothesis claim?
2. Which turns introduced the evidence the hypothesis rests on?
3. Which turns does the hypothesis NOT depend on?

Then output the JSON."""

FEW_SHOT_EXAMPLE = """EXAMPLE (from a different conversation):
Conversation:
T1 Alice: I see error 503 on the gateway.
T2 Bob: Let me check the load balancer.
T3 Bob: LB is healthy. Check the upstream service.
T4 Alice: Upstream is at 80% CPU.
T5 Bob: That's normal for this time of day. The 503 must be from somewhere else.

HYPOTHESIS: The 503 errors are not caused by upstream CPU.

Answer: {"depends_on_turns": ["T1", "T4", "T5"]}
Reasoning: T1 introduces the 503; T4 introduces the CPU observation; T5 is the conclusion that links them. T2 and T3 (the LB check) are not dependencies of THIS hypothesis.

---

Now do the same for the conversation below.

"""


def parse_response(text):
    if not text:
        return None
    m = re.search(r'"depends_on_turns"\s*:\s*\[([^\]]*)\]', text)
    if not m:
        return None
    try:
        items = json.loads("[" + m.group(1) + "]")
        return [str(x).strip() for x in items if str(x).strip()]
    except json.JSONDecodeError:
        return None


def turns_to_dep_set(turns):
    deps = set()
    for t in turns:
        if t in TURN_TO_OBS:
            deps.update(TURN_TO_OBS[t])
    return deps


def score_one(predicted_turns, hyp_id):
    truth = GROUND_TRUTH_DEP[hyp_id]
    if predicted_turns is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "predicted_turns": None,
                "predicted_deps": [],
                "ground_truth": sorted(truth),
                "h1_to_h4_recovered": False}
    predicted = turns_to_dep_set(predicted_turns)
    tp = len(predicted & truth)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(truth) if truth else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    h1_to_h4 = (hyp_id == "h1" and "h4" in predicted)
    return {"precision": p, "recall": r, "f1": f1,
            "predicted_turns": predicted_turns,
            "predicted_deps": sorted(predicted),
            "ground_truth": sorted(truth),
            "h1_to_h4_recovered": h1_to_h4}


def run_variant(variant, api_key):
    print(f"\n=== Variant: {variant} ===")
    results = {}
    for hyp_id, hyp_text in HYPOTHESIS_TEXTS.items():
        prompt = PROMPT_BASE.format(transcript=PHASE2_TRANSCRIPT,
                                    hypothesis_text=hyp_text)
        if variant == "few-shot":
            prompt = FEW_SHOT_EXAMPLE + prompt
        if variant in ("cot", "self-consistency"):
            prompt = prompt + PROMPT_COT_SUFFIX

        if variant == "self-consistency":
            votes = Counter()
            samples = []
            for i in range(5):
                resp = plib.call_llm(
                    system="You are a careful logical analyst.",
                    user=prompt, api_key=api_key, temperature=0.7)
                turns = parse_response(resp) or []
                samples.append(turns)
                for t in turns:
                    votes[t] += 1
            predicted_turns = [t for t, c in votes.items() if c >= 3]
            score = score_one(predicted_turns, hyp_id)
            score["sc_samples"] = samples
            score["sc_votes"] = dict(votes)
        else:
            resp = plib.call_llm(
                system="You are a careful logical analyst.",
                user=prompt, api_key=api_key, temperature=0)
            predicted_turns = parse_response(resp)
            score = score_one(predicted_turns, hyp_id)

        results[hyp_id] = score
        print(f"  {hyp_id}: P={score['precision']:.2f} R={score['recall']:.2f} "
              f"F1={score['f1']:.2f}  pred_turns={score['predicted_turns']}  "
              f"pred_deps={score['predicted_deps']}  gt={score['ground_truth']}"
              f"  h1->h4={'YES' if score['h1_to_h4_recovered'] else 'no'}")

    avg_p = sum(r["precision"] for r in results.values()) / len(results)
    avg_r = sum(r["recall"] for r in results.values()) / len(results)
    avg_f1 = sum(r["f1"] for r in results.values()) / len(results)
    h1_to_h4_any = any(r["h1_to_h4_recovered"] for r in results.values())
    print(f"  AVG: P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}  "
          f"h1->h4_recovered={'YES' if h1_to_h4_any else 'no'}")
    return {"per_hypothesis": results,
            "average": {"precision": avg_p, "recall": avg_r, "f1": avg_f1},
            "h1_to_h4_recovered": h1_to_h4_any}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",
                   choices=["zero-shot", "few-shot", "cot",
                            "self-consistency", "all"],
                   default="all")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--base-url",
                   default="https://api.openai.com/v1/chat/completions")
    p.add_argument("--output",
                   default="experiments/results/e1_llm_baseline/e1_llm_baseline.json")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY (E1 uses GPT-4o).")

    plib.BACKEND = "openai"
    plib.MODEL = args.model
    plib.API_URL = args.base_url

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    variants = (["zero-shot", "few-shot", "cot", "self-consistency"]
                if args.variant == "all" else [args.variant])

    all_results = {
        "experiment": "E1 — LLM-prompted dependency-extraction baseline",
        "model": args.model,
        "backend": "openai",
        "ground_truth": "post-T13 (full reference incl. unification override)",
        "variants": {},
    }
    for v in variants:
        all_results["variants"][v] = run_variant(v, api_key)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n=== SUMMARY ===")
    print(f"{'Variant':<20} {'P':>8} {'R':>8} {'F1':>8}  {'h1->h4':>7}")
    for v in variants:
        a = all_results["variants"][v]["average"]
        h1h4 = all_results["variants"][v]["h1_to_h4_recovered"]
        print(f"{v:<20} {a['precision']:>8.3f} {a['recall']:>8.3f} "
              f"{a['f1']:>8.3f}  {'YES' if h1h4 else 'no':>7}")
    print("\nVerifier-pipeline reference (current tab:end-to-end, GPT-4o row):")
    print(f"{'verifier-pipeline':<20} {1.000:>8.3f} {0.430:>8.3f} {0.600:>8.3f}  "
          f"{'YES':>7}")
    print("\nDecision gate:")
    print("  - If any LLM-prompted variant recovers h1->h4 with comparable F1,")
    print("    STOP and surface to advisor before E1b/E2/E3 (per EXPERIMENTS.md).")


if __name__ == "__main__":
    main()
