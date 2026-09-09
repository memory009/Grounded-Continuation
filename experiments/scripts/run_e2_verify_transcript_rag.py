"""
run_e2_verify_transcript_rag.py — transcript-RAG baseline for E2.

Question: the verifier's E2 stale-premise win (15/15 vs 14/15 LLM-only) and
pooled stale+counterfactual win (25/25 vs 24/25) — is it attributable to
the symbolic engine's lifecycle/status reasoning, or could a transcript-RAG
baseline match it?

E2 is a stress test for retrieval-as-substitute-for-engine. Phase 2 has only
13 turns total, so the LLM-only baseline already gets the *full* truncated
transcript (turns 1..t). With top-k=20 retrieval, the retrieval block is
trivially the entire window — so we expect transcript-RAG ≈ baseline.

The architecturally-meaningful comparison: top-k=5 transcript-RAG.
This forces retrieval to be selective, mirroring the verifier's dep-walk
(which also selectively traces upstream commitments). If transcript-RAG
top-k=5 still fails on stale-premise (where the abandoned-hypothesis
turn is high-similarity to the re-asserting candidate), the verifier's
win is attributable to lifecycle reasoning, not retrieval.

Setup:
  - Same 50 candidate continuations + truncation_t labels
  - For each item, build per-turn chunks of pipeline.PHASE2_CONVERSATION[:t]
  - Embed turns + candidate; retrieve top-k by cosine similarity to candidate
  - Same grounded/ungrounded prompt as baseline_judgement, body now is
    retrieved turns rather than full window
  - Same GPT-4o, temperature=0
  - Score against author labels (κ=0.733 blinded overlap)

Cost: ~$0.30 per top-k variant (50 items × ~$0.006).

Usage:
  set -a; source .env.local; set +a
  python run_e2_verify_transcript_rag.py --top-k 5
  python run_e2_verify_transcript_rag.py --top-k 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from pipeline import PHASE2_CONVERSATION  # noqa: E402

EMBED_MODEL = "text-embedding-3-small"
TEST_SET_PATH = (PROJECT_ROOT / "experiments" / "e2_verify"
                 / "verify_test_set_phase2.yaml")
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "e2_verify_transcript_rag"
EXISTING_RESULTS = (PROJECT_ROOT / "experiments" / "results" / "e2_verify"
                    / "e2_verify_results.json")


def embed_batch(client, texts, batch_size=200, model=EMBED_MODEL):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        for e in resp.data:
            out.append(np.asarray(e.embedding, dtype=np.float32))
    return out


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def transcript_rag_judgement(api_key, openai_client, truncate_at_turn,
                             candidate, top_k):
    """Retrieve top-k turns by similarity to the candidate, then ask the
    LLM grounded/ungrounded with only those retrieved turns visible."""
    available_turns = PHASE2_CONVERSATION[:truncate_at_turn]
    turn_texts = [
        f"T{i+1} {turn['speaker']}: {turn['text']}"
        for i, turn in enumerate(available_turns)
    ]
    n_avail = len(turn_texts)
    effective_k = min(top_k, n_avail)

    embeds = embed_batch(openai_client, turn_texts + [candidate])
    turn_embs, cand_emb = embeds[:-1], embeds[-1]
    sims = [cosine_sim(e, cand_emb) for e in turn_embs]
    ranked_indices = sorted(range(n_avail), key=lambda i: -sims[i])
    chosen = sorted(ranked_indices[:effective_k])  # restore chronological order

    retrieval_block = "\n".join(
        f"{turn_texts[i]}    [sim={sims[i]:.3f}]"
        for i in chosen
    )

    user_prompt = (
        f"Below is a multi-turn debugging conversation up to turn {truncate_at_turn}, "
        f"with only the most relevant turns retrieved (top-{effective_k} of {n_avail} "
        f"by similarity to the candidate). After it, a candidate next utterance "
        f"is proposed. Decide whether the candidate is GROUNDED (its claim follows "
        f"from the prior conversation, including respecting any hypotheses that "
        f"were abandoned or revised) or UNGROUNDED (its claim contradicts the "
        f"prior conversation, asserts something abandoned, or comes from outside "
        f"the conversation).\n\n"
        f"=== RETRIEVED CONVERSATION TURNS ===\n{retrieval_block}\n=== END ===\n\n"
        f"Candidate next utterance: {candidate}\n\n"
        f"Answer with one word: GROUNDED or UNGROUNDED."
    )
    resp = pipeline.call_llm(
        system="You are a careful conversation analyst.",
        user=user_prompt, api_key=api_key, temperature=0,
    )
    if not resp:
        return "ungrounded", "(no response)", chosen, sims
    s = resp.strip().upper()
    if s.startswith("UNGROUNDED"):
        return "ungrounded", resp.strip(), chosen, sims
    if s.startswith("GROUNDED"):
        return "grounded", resp.strip(), chosen, sims
    for tok in s.split():
        if tok.startswith("UNGROUNDED"):
            return "ungrounded", resp.strip(), chosen, sims
        if tok.startswith("GROUNDED"):
            return "grounded", resp.strip(), chosen, sims
    return "ungrounded", resp.strip(), chosen, sims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-k turns to retrieve. Use 5 for selective; "
                        "20 (>=full window) is equivalent to LLM-only baseline.")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        sys.exit("Set OPENAI_API_KEY")

    pipeline.BACKEND = "openai"
    pipeline.MODEL = "gpt-4o"
    pipeline.API_URL = "https://api.openai.com/v1/chat/completions"

    test_set = yaml.safe_load(TEST_SET_PATH.read_text())
    items = test_set.get("items") if isinstance(test_set, dict) else test_set
    if args.max_items:
        items = items[:args.max_items]
    print(f"Loaded {len(items)} items, top_k={args.top_k}")

    if args.dry_run:
        est = len(items) * 0.006
        print(f"  Estimated cost: ~${est:.2f}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    openai_client = OpenAI(api_key=api_key)

    # Load existing E2 results for verifier + baseline comparison
    existing = json.load(open(EXISTING_RESULTS)) if EXISTING_RESULTS.exists() else {"per_item": []}
    existing_by_id = {r["id"]: r for r in existing.get("per_item", [])}

    results = []
    for item in items:
        t = item["truncation_t"]
        cand = item["candidate"]
        gt = item["label"]
        cat = item["category"]
        item_id = item["id"]

        try:
            tr_ans, tr_raw, chosen_idx, sims = transcript_rag_judgement(
                api_key, openai_client, t, cand, args.top_k
            )
        except Exception as e:
            print(f"  [{item_id}] ERROR: {e}")
            results.append({"id": item_id, "category": cat, "error": str(e)})
            continue

        tr_correct = (tr_ans == gt)
        existing_r = existing_by_id.get(item_id, {})
        verifier_ans = existing_r.get("verifier")
        baseline_ans = existing_r.get("baseline")
        v_correct = existing_r.get("verifier_correct")
        b_correct = existing_r.get("baseline_correct")

        results.append({
            "id": item_id,
            "category": cat,
            "truncation_t": t,
            "candidate": cand,
            "ground_truth_label": gt,
            "transcript_rag": tr_ans,
            "transcript_rag_correct": tr_correct,
            "transcript_rag_raw": tr_raw,
            "transcript_rag_top_k": args.top_k,
            "transcript_rag_n_available": min(t, len(PHASE2_CONVERSATION)),
            "transcript_rag_chosen_turn_indices": [i + 1 for i in chosen_idx],
            "transcript_rag_top_sims": sorted([round(s, 4) for s in sims], reverse=True)[:args.top_k],
            "verifier (existing)": verifier_ans,
            "verifier_correct (existing)": v_correct,
            "baseline (existing)": baseline_ans,
            "baseline_correct (existing)": b_correct,
        })

        markers = (
            ("OK" if tr_correct else "X "),
            ("OK" if v_correct else ("X" if v_correct is False else "-")),
            ("OK" if b_correct else ("X" if b_correct is False else "-")),
        )
        print(f"  [{item_id}/{cat}/t={t}] gt={gt:<10}  "
              f"TR={tr_ans:<10}{markers[0]}  V={str(verifier_ans):<10}{markers[1]}  "
              f"B={str(baseline_ans):<10}{markers[2]}")

    # Aggregate by category
    by_cat = defaultdict(lambda: {"tr_correct": 0, "v_correct": 0, "b_correct": 0, "n": 0})
    for r in results:
        if "error" in r:
            continue
        c = r["category"]
        by_cat[c]["n"] += 1
        by_cat[c]["tr_correct"] += int(r["transcript_rag_correct"])
        by_cat[c]["v_correct"] += int(bool(r.get("verifier_correct (existing)")))
        by_cat[c]["b_correct"] += int(bool(r.get("baseline_correct (existing)")))

    print(f"\n{'='*70}")
    print(f"E2 TRANSCRIPT-RAG vs VERIFIER vs BASELINE  (top_k={args.top_k}, n={len(results)})")
    print(f"{'='*70}")
    cat_summary = {}
    for c in ("actual", "stale", "cross_conv", "counterfactual"):
        d = by_cat.get(c, {"n": 0})
        if d["n"] == 0:
            continue
        n = d["n"]
        cat_summary[c] = {
            "n": n,
            "transcript_rag": f"{d['tr_correct']}/{n} = {100*d['tr_correct']/n:.1f}%",
            "verifier":       f"{d['v_correct']}/{n} = {100*d['v_correct']/n:.1f}%",
            "baseline":       f"{d['b_correct']}/{n} = {100*d['b_correct']/n:.1f}%",
            "delta_TR_minus_V_pp":   round(100*(d['tr_correct']-d['v_correct'])/n, 2),
            "delta_TR_minus_B_pp":   round(100*(d['tr_correct']-d['b_correct'])/n, 2),
        }
        print(f"  {c:<16} n={n:>2}  TR={cat_summary[c]['transcript_rag']}  "
              f"V={cat_summary[c]['verifier']}  B={cat_summary[c]['baseline']}")

    # Pooled stale + counterfactual (the paper's headline)
    stale_cf = [r for r in results if r["category"] in ("stale", "counterfactual") and "error" not in r]
    if stale_cf:
        n = len(stale_cf)
        tr_c = sum(int(r["transcript_rag_correct"]) for r in stale_cf)
        v_c = sum(int(bool(r.get("verifier_correct (existing)"))) for r in stale_cf)
        b_c = sum(int(bool(r.get("baseline_correct (existing)"))) for r in stale_cf)
        pooled = {
            "n": n,
            "transcript_rag": f"{tr_c}/{n}",
            "verifier": f"{v_c}/{n}",
            "baseline": f"{b_c}/{n}",
            "delta_TR_minus_V_pp": round(100*(tr_c-v_c)/n, 2),
            "delta_TR_minus_B_pp": round(100*(tr_c-b_c)/n, 2),
        }
        print(f"\n  POOLED stale+counterfactual (paper headline): "
              f"TR={tr_c}/{n}, V={v_c}/{n}, B={b_c}/{n}")
        print(f"    Δ_engine (V−TR) = {-pooled['delta_TR_minus_V_pp']:+.1f} pp")
        print(f"    TR vs B         = {pooled['delta_TR_minus_B_pp']:+.1f} pp")
    else:
        pooled = {}

    out = {
        "experiment": "E2 transcript-RAG baseline (no engine, retrieval over Phase 2 transcript)",
        "purpose": "Baseline: isolate symbolic engine vs retrieval on E2",
        "model": "gpt-4o",
        "embed_model": EMBED_MODEL,
        "top_k": args.top_k,
        "by_category": cat_summary,
        "pooled_stale_counterfactual": pooled,
        "per_item": results,
    }
    out_path = OUT_DIR / f"e2_verify_transcript_rag_k{args.top_k}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
