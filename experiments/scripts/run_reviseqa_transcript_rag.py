#!/usr/bin/env python3
"""ReviseQA transcript-RAG comparator arm (third arm).

Config-aligned with the completed two-arm run
(experiments/results/reviseqa/reviseqa_full_qwen7b_explicit_no_corr_no_reasoning.json):
  - same tested model:   Qwen/Qwen2.5-7B-Instruct via local vLLM
                         (http://localhost:8000/v1/chat/completions)
  - same task setting:   explicit_no_correction_no_reasoning
  - same QA prompt shape: REVISEQA_SYSTEM + REVISEQA_PROMPT_TEMPLATE,
                          growing per-chain dialogue, 0-shot demo turn
  - same closed-form scoring (True/False/Uncertain) and LCATA@2/4/7
  - same 930 verified scenarios in the same order (load_reviseqa,
    max_scenarios=930; asserted against the original run's
    completed_scenario_ids at startup)

Third-arm design (transcript RAG):
  At each edit step, instead of the engine dep-map block (hybrid arm)
  or nothing (baseline arm), the user message is prefixed with the
  top-k context statements retrieved from the CURRENT POST-EDIT
  context (`edits[i].edited_natural_language_context`, the benchmark's
  full NL context after the edit is applied) ranked by TF-IDF cosine
  similarity against the conclusion being evaluated at that step.

  Faithfulness note (documented in output config as well): the
  retrieval arm retrieves from the current post-edit context, i.e. it
  has access to the same edited facts the baseline sees in text form —
  it merely SELECTS a similarity-ranked subset. The comparison
  therefore isolates "similarity selection" vs "maintained dependency
  structure" (hybrid arm) at matched content access.

  k = 22: matched to the information budget of the dep-map arm. In the
  completed 930-scenario run, per-step Dep(h1) size has median 22.0 /
  mean 22.16 (n = 6510 steps; initial median 19). The retrieval pool
  (post-edit context) has median 18 statements, so at k = 22 the
  retrieval arm frequently receives the ENTIRE current context ranked
  by similarity — an intentionally strong comparator: it can never see
  less content than the dep-map arm's budget.

  Retriever: TF-IDF cosine (scikit-learn; local, deterministic).
  sentence-transformers is NOT installed in the `ecm` env and paid
  embedding APIs are ruled out for this comparator, so TF-IDF is used
  (documented in output config as `retriever`).

Usage:
  python experiments/scripts/run_reviseqa_transcript_rag.py \
      --max-scenarios 10 --output .../reviseqa_tr_smoke.json   # smoke
  python experiments/scripts/run_reviseqa_transcript_rag.py    # full 930
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = os.environ.get("RQ_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_DIR)

from benchmark_adapter import (  # noqa: E402
    load_reviseqa,
    _reviseqa_format_explicit_context,
    _reviseqa_parse_answer,
    _reviseqa_chat_call,
    _reviseqa_aggregate_lcata,
    _reviseqa_atomic_save,
    _reviseqa_load_checkpoint,
    REVISEQA_SYSTEM,
    REVISEQA_PROMPT_TEMPLATE,
)

DATA_DIR = os.environ.get("RQ_DATA_DIR", os.path.join(PROJECT_DIR, "data", "reviseqa", "reviseqa_data", "nl", "verified"))
REFERENCE_RUN = os.path.join(
    PROJECT_DIR, "experiments/results/reviseqa/"
    "reviseqa_full_qwen7b_explicit_no_corr_no_reasoning.json")
DEFAULT_OUT = os.path.join(
    PROJECT_DIR,
    "experiments/results/reviseqa_tr_baseline/reviseqa_tr_full.json")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "http://localhost:8000/v1/chat/completions"
TOP_K = 22
K_RATIONALE = (
    "k=22 matches the dep-map arm's per-step information budget: median "
    "dep_h1_size 22.0 / mean 22.16 across all 6510 edit steps of the "
    "completed 930-scenario two-arm run (initial Dep(h1) median 19). "
    "Retrieval pool (post-edit context) median is 18 statements, so the "
    "RAG arm frequently sees the entire current context similarity-ranked "
    "— it never has a smaller content budget than the dep-map arm.")


def retrieve_top_k(pool: list[str], query: str, k: int) -> list[int]:
    """Deterministic TF-IDF cosine retrieval. Returns pool indices in
    descending-similarity order (stable tie-break by pool index)."""
    if not pool:
        return []
    try:
        vec = TfidfVectorizer(lowercase=True)
        mat = vec.fit_transform(pool + [query])
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    except ValueError:
        # Degenerate vocabulary (e.g., all stop tokens): fall back to
        # original order.
        sims = np.zeros(len(pool))
    order = sorted(range(len(pool)), key=lambda i: (-float(sims[i]), i))
    return order[:k]


def format_rag_block(retrieved: list[str], k_used: int, pool_size: int) -> str:
    lines = [
        "## Retrieved context (transcript RAG)",
        f"Top-{k_used} statements from the current context most similar "
        "to the conclusion under evaluation "
        f"(pool size {pool_size}, similarity-ranked):",
    ]
    if not retrieved:
        lines.append("  (none)")
    else:
        for txt in retrieved:
            lines.append(f"  - {txt}")
    return "\n".join(lines) + "\n"


def evaluate_scenario_tr(scenario: dict, api_key: str, base_url: str,
                         model: str, k: int) -> dict:
    """One scenario chain (1 demo + 7 edits) under
    explicit_no_correction_no_reasoning, transcript-RAG arm only."""
    scenario_id = scenario["id"]
    edits = scenario.get("edits", [])
    orig_ctx_text = "\n".join(scenario.get("original_context", []) or [])
    conclusion = scenario.get("conclusion", "")
    orig_answer = scenario.get("answer", "Uncertain")
    base_question = f"Does the context entail the conclusion '{conclusion}'?"

    # no_reasoning: demo carries an empty reasoning (mirrors two-arm run)
    demo_assistant = json.dumps({"reasoning": "", "answer": orig_answer})
    messages = [
        {"role": "system", "content": REVISEQA_SYSTEM},
        {"role": "user", "content": REVISEQA_PROMPT_TEMPLATE.format(
            context=orig_ctx_text, question=base_question)},
        {"role": "assistant", "content": demo_assistant},
    ]

    per_step: list[dict] = []
    trace: list[bool] = []
    for step_idx, edit in enumerate(edits, start=1):
        gt = edit.get("answer", "")
        step_conclusion = edit.get("conclusion", conclusion)
        step_question = (f"Does the context entail the conclusion "
                         f"'{step_conclusion}'?")
        delta_ctx = _reviseqa_format_explicit_context(edit)

        # Retrieval pool: CURRENT post-edit context (benchmark-provided).
        pool = edit.get("edited_natural_language_context") or []
        idxs = retrieve_top_k(pool, step_conclusion, k)
        retrieved = [pool[i] for i in idxs]
        rag_block = format_rag_block(retrieved, len(retrieved), len(pool))

        user_msg = (rag_block + "\n" + REVISEQA_PROMPT_TEMPLATE.format(
            context=delta_ctx, question=step_question))
        messages.append({"role": "user", "content": user_msg})
        raw = _reviseqa_chat_call(messages, api_key, base_url, model)
        messages.append({"role": "assistant", "content": raw or ""})
        pred = _reviseqa_parse_answer(raw)
        correct = (pred == gt)
        trace.append(bool(correct))
        # no_correction: never append a CORRECTION turn.

        per_step.append({
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "gt": gt,
            "transcript_rag": {
                "raw": raw, "parsed": pred, "correct": correct,
                "rag_diag": {
                    "pool_size": len(pool),
                    "k_used": len(retrieved),
                    "retrieved_indices": idxs,
                },
            },
        })

    def _lcata_ks(tr):
        return {
            "k2": int(all(tr[:2])) if len(tr) >= 2 else 0,
            "k4": int(all(tr[:4])) if len(tr) >= 4 else 0,
            "k7": int(all(tr[:7])) if len(tr) >= 7 else 0,
        }

    return {
        "scenario_id": scenario_id,
        "task_setting": "explicit_no_correction_no_reasoning",
        "n_edits": len(edits),
        "modification_types": [e.get("modification_type") for e in edits],
        "transcript_rag_trace": trace,
        "transcript_rag_lcata": _lcata_ks(trace),
        "per_step": per_step,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-scenarios", type=int, default=930)
    ap.add_argument("--output", type=str, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--force-restart", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    api_key = os.environ.get("VLLM_API_KEY", "dummy")

    scenarios = load_reviseqa(DATA_DIR, max_scenarios=args.max_scenarios)
    print(f"Loaded {len(scenarios)} verified scenarios from {DATA_DIR}")

    # Scenario-set alignment guard against the original two-arm run.
    if os.path.isfile(REFERENCE_RUN):
        ref = json.load(open(REFERENCE_RUN))
        ref_ids = ref.get("completed_scenario_ids", [])
        ids = [s["id"] for s in scenarios]
        if args.max_scenarios == 930 and ids != ref_ids:
            print("ERROR: scenario set/order mismatch vs reference run; "
                  "aborting to preserve config alignment.")
            sys.exit(2)
        if args.max_scenarios < 930 and ids != ref_ids[:len(ids)]:
            print("ERROR: smoke scenario prefix mismatch vs reference run.")
            sys.exit(2)
        print("Scenario set/order verified against reference two-arm run.")

    checkpoint = None
    if not args.force_restart:
        try:
            checkpoint = _reviseqa_load_checkpoint(args.output)
        except json.JSONDecodeError as e:
            print(f"[warn] malformed checkpoint ({e}); fresh run.")
    records: list[dict] = (checkpoint or {}).get("records", []) or []
    done_ids = {r["scenario_id"] for r in records}
    if done_ids:
        print(f"[resume] {len(done_ids)} scenarios already complete.")

    config = {
        "arm": "transcript_rag",
        "tested_backend": "openai",
        "tested_model": MODEL,
        "tested_url": BASE_URL,
        "data_dir": DATA_DIR,
        "include_reasoning": False,
        "include_correction": False,
        "max_scenarios": args.max_scenarios,
        "task_setting": "explicit_no_correction_no_reasoning",
        "retriever": ("TF-IDF cosine (scikit-learn TfidfVectorizer + "
                      "cosine_similarity); sentence-transformers not "
                      "installed in ecm env; no paid embedding APIs used"),
        "retrieval_pool": ("edits[i].edited_natural_language_context — the "
                           "benchmark-provided FULL post-edit NL context at "
                           "each step; the RAG arm sees the same edited "
                           "facts the baseline sees in text and selects a "
                           "similarity-ranked subset, isolating similarity "
                           "selection vs maintained dependency structure "
                           "at matched content access"),
        "retrieval_query": "the conclusion under evaluation at that step",
        "top_k": args.top_k,
        "top_k_rationale": K_RATIONALE,
        "prompt_alignment": ("same REVISEQA_SYSTEM / REVISEQA_PROMPT_TEMPLATE"
                             ", growing per-chain dialogue and 0-shot demo "
                             "as the two-arm run; the RAG block replaces "
                             "the hybrid arm's dep-map block"),
    }

    def save(status: str):
        agg = _reviseqa_aggregate_lcata(records, "transcript_rag_lcata") \
            if records else {}
        _reviseqa_atomic_save(args.output, {
            "status": status,
            "task_setting": "explicit_no_correction_no_reasoning",
            "completed_scenario_ids": [r["scenario_id"] for r in records],
            "n_completed": len(records),
            "n_total_target": len(scenarios),
            "aggregate_transcript_rag_lcata": agg,
            "records": records,
            "config": config,
        })

    remaining = [s for s in scenarios if s["id"] not in done_ids]
    t0 = datetime.datetime.now()
    for j, scenario in enumerate(remaining):
        idx = len(scenarios) - len(remaining) + j + 1
        print(f"  [{idx}/{len(scenarios)}] {scenario['id']}", flush=True)
        rec = evaluate_scenario_tr(scenario, api_key, BASE_URL, MODEL,
                                   args.top_k)
        records.append(rec)
        print(f"    tr_trace={rec['transcript_rag_trace']} "
              f"LCATA={rec['transcript_rag_lcata']}", flush=True)
        save("in_progress")

    agg = _reviseqa_aggregate_lcata(records, "transcript_rag_lcata")
    print(f"\n{'='*60}\nTRANSCRIPT-RAG SUMMARY ({len(records)} scenarios)"
          f"\n{'='*60}")
    for k in ("k2", "k4", "k7"):
        print(f"  LCATA@{k[1:]}: {agg[k]}")
    save("completed")
    dt = datetime.datetime.now() - t0
    print(f"Elapsed this session: {dt}. Results saved to {args.output}")


if __name__ == "__main__":
    main()
