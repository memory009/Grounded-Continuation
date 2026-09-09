#!/usr/bin/env python3
"""ReviseQA multi-model sweep: run any of the four arms for one QA model.

Arms (same protocol as the published Qwen2.5-7B runs, one output file
per arm so the existing analysis code keeps working):

  two_arm       LLM-only baseline + engine (benchmark-native updates), the
                original evaluate_reviseqa_scenario chain
                -> records with baseline_trace / hybrid_dep_only_trace
  transcript_rag  TF-IDF top-k=22 over the post-edit context
                -> records with transcript_rag_trace
  e2e_replay    end-to-end arm: the GPT-4o Interpreter's per-step ops are
                REPLAYED from the cached shards of the published run
                (experiments/results/reviseqa_interp_full/interp_shard*.json),
                so a new QA model costs no Interpreter calls
                -> records with interp_trace

Common protocol: explicit_no_correction_no_reasoning task setting,
REVISEQA_SYSTEM / REVISEQA_PROMPT_TEMPLATE, growing per-chain dialogue,
0-shot demo turn, temperature 0, closed-form True/False/Uncertain scoring,
LCATA@2/4/7 (the benchmark's own easy/medium/hard cut-offs), the same 930
verified scenarios in the same order (asserted against the reference run).

New here: scenarios are independent chains, so they run on a thread pool
(--workers); vLLM batches the concurrent requests. Records are re-sorted
into canonical order on every checkpoint save.

Usage:
  # smoke: 10 scenarios, all arms, local vLLM on port 8001
  python experiments/scripts/run_reviseqa_multimodel.py \\
      --tag qwen2.5-14b --model Qwen/Qwen2.5-14B-Instruct \\
      --base-url http://localhost:8001/v1/chat/completions \\
      --arms two_arm transcript_rag e2e_replay --max-scenarios 10

  # GPT-4o pilot on the first 200 scenarios, two arms only
  OPENAI_API_KEY=... python experiments/scripts/run_reviseqa_multimodel.py \\
      --tag gpt-4o --model gpt-4o \\
      --base-url https://api.openai.com/v1/chat/completions \\
      --api-key-env OPENAI_API_KEY --arms two_arm --max-scenarios 200 \\
      --workers 6
"""

import argparse
import datetime
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default is the original server path; override with RQ_PROJECT_DIR so the
# script also runs from a local checkout (the imports below need it set
# before argparse runs).
PROJECT_DIR = os.environ.get("RQ_PROJECT_DIR",
                             ".")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "experiments", "scripts"))

import benchmark_adapter as BA  # noqa: E402
import run_reviseqa_transcript_rag as TR  # noqa: E402
import run_reviseqa_interp_full as IF  # noqa: E402
from benchmark_adapter import (  # noqa: E402
    load_reviseqa,
    evaluate_reviseqa_scenario,
    _reviseqa_init_scenario_state,
    _reviseqa_engine_block_dep_only,
    _reviseqa_dep_map_stats_from_engine,
    _reviseqa_format_explicit_context,
    _reviseqa_parse_answer,
    _reviseqa_aggregate_lcata,
    _reviseqa_atomic_save,
    _reviseqa_load_checkpoint,
    REVISEQA_SYSTEM,
    REVISEQA_PROMPT_TEMPLATE,
)

DATA_DIR = os.environ.get(
    "RQ_DATA_DIR",
    "datasets_cache/reviseqa/reviseqa_data/nl/verified")
REFERENCE_RUN = os.path.join(
    PROJECT_DIR, "experiments/results/reviseqa/"
    "reviseqa_full_qwen7b_explicit_no_corr_no_reasoning.json")
INTERP_CACHE_DIR = os.path.join(
    PROJECT_DIR, "experiments/results/reviseqa_interp_full")
OUT_ROOT = os.path.join(PROJECT_DIR, "experiments/results/reviseqa_multimodel")

ARMS = ("two_arm", "transcript_rag", "e2e_replay")
TRACE_KEY = {"two_arm": "hybrid_dep_only_lcata",
             "transcript_rag": "transcript_rag_lcata",
             "e2e_replay": "interp_lcata"}


# --------------------------------------------------------------------------
# Chat call with 429-aware backoff + usage accounting (shared by all arms via
# monkeypatch of the helper the imported runners call).
# --------------------------------------------------------------------------

USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
         "failed_calls": 0, "empty_content": 0}
USAGE_LOCK = threading.Lock()


def chat_call(messages, api_key, url, model, temperature=0,
              max_tokens=None, max_retries=8):
    import requests
    if max_tokens is None:
        max_tokens = int(os.environ.get("REVISEQA_MAX_TOKENS", "800"))
    backoff = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json={"model": model, "temperature": temperature,
                      "max_tokens": max_tokens, "messages": messages},
                timeout=600)
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            data = resp.json()
            if "choices" in data and data["choices"]:
                u = data.get("usage", {}) or {}
                content = data["choices"][0]["message"].get("content")
                with USAGE_LOCK:
                    USAGE["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                    USAGE["completion_tokens"] += \
                        u.get("completion_tokens", 0) or 0
                    USAGE["calls"] += 1
                    if not content:
                        USAGE["empty_content"] += 1
                return content
            print(f"  [chat] API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', data)}",
                  flush=True)
        except Exception as e:
            print(f"  [chat] request error (attempt {attempt+1}): {e}",
                  flush=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, 64)
    with USAGE_LOCK:
        USAGE["failed_calls"] += 1
    return None


# route every arm's QA call through chat_call
BA._reviseqa_chat_call = chat_call
TR._reviseqa_chat_call = chat_call
IF._reviseqa_chat_call = chat_call


# --------------------------------------------------------------------------
# e2e replay arm
# --------------------------------------------------------------------------

def load_interp_cache() -> dict:
    """scenario_id -> list of per-step cached op lists (from the shards)."""
    cache = {}
    for f in sorted(glob.glob(os.path.join(INTERP_CACHE_DIR,
                                           "interp_shard*.json"))):
        d = json.load(open(f))
        for r in d.get("records", []):
            cache[r["scenario_id"]] = [
                (s.get("interp") or {}).get("ops") or []
                for s in r.get("per_step", [])]
    return cache


def evaluate_scenario_replay(scenario, cached_ops, api_key, url, model):
    """Same as run_reviseqa_interp_full.evaluate_scenario_interp but the
    Interpreter ops come from the cache instead of a live GPT-4o call."""
    sid = scenario["id"]
    edits = scenario.get("edits", [])
    conclusion = scenario.get("conclusion", "")
    orig_ctx_text = "\n".join(scenario.get("original_context", []) or [])
    base_q = f"Does the context entail the conclusion '{conclusion}'?"
    demo_assistant = json.dumps({"reasoning": "",
                                 "answer": scenario.get("answer",
                                                        "Uncertain")})
    pipe_i, _, active_i, dep_i, counter_i, init_diag = \
        _reviseqa_init_scenario_state(scenario)
    eng_i = pipe_i.engine

    messages = [
        {"role": "system", "content": REVISEQA_SYSTEM},
        {"role": "user", "content": REVISEQA_PROMPT_TEMPLATE.format(
            context=orig_ctx_text, question=base_q)},
        {"role": "assistant", "content": demo_assistant},
    ]
    trace, per_step = [], []
    for step_idx, edit in enumerate(edits, start=1):
        gt = edit.get("answer", "")
        step_conclusion = edit.get("conclusion", conclusion)
        step_q = f"Does the context entail the conclusion '{step_conclusion}'?"
        delta_ctx = _reviseqa_format_explicit_context(edit)

        ops = cached_ops[step_idx - 1] if step_idx - 1 < len(cached_ops) \
            else []
        n_bad_ops = 0
        for op in ops:
            if not isinstance(op, dict):
                n_bad_ops += 1
                continue
            if op.get("op") == "remove" and op.get("id"):
                txt = IF._interp_retract_by_id(eng_i, active_i, dep_i,
                                               str(op["id"]))
                if not txt:
                    n_bad_ops += 1
            elif (op.get("op") == "add"
                  and op.get("kind") in ("fact", "rule")
                  and op.get("text")):
                IF._interp_add(eng_i, active_i, dep_i, counter_i,
                               op["kind"], op["text"])
            else:
                n_bad_ops += 1
        eng_i.dependencies["h1"] = {o for o in dep_i if o in active_i}

        engine_block = _reviseqa_engine_block_dep_only(eng_i, "h1")
        user_msg = (engine_block + "\n" + REVISEQA_PROMPT_TEMPLATE.format(
            context=delta_ctx, question=step_q))
        messages.append({"role": "user", "content": user_msg})
        qa_raw = chat_call(messages, api_key, url, model)
        messages.append({"role": "assistant", "content": qa_raw or ""})
        pred = _reviseqa_parse_answer(qa_raw)
        correct = (pred == gt)
        trace.append(bool(correct))
        per_step.append({
            "step_idx": step_idx,
            "edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "gt": gt,
            "interp": {"ops": ops, "n_bad_ops": n_bad_ops,
                       "source": "cached_gpt4o_ops"},
            "qa": {"raw": qa_raw, "parsed": pred, "correct": correct,
                   "dep_stats": _reviseqa_dep_map_stats_from_engine(
                       eng_i, "h1")},
        })

    def _lcata(tr):
        return {"k2": int(all(tr[:2])) if len(tr) >= 2 else 0,
                "k4": int(all(tr[:4])) if len(tr) >= 4 else 0,
                "k7": int(all(tr[:7])) if len(tr) >= 7 else 0}

    return {
        "scenario_id": sid,
        "task_setting": "explicit_no_correction_no_reasoning",
        "init_diag": init_diag,
        "interp_trace": trace,
        "interp_lcata": _lcata(trace),
        "per_step": per_step,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_arm(arm, scenarios, args, api_key, out_path, interp_cache):
    checkpoint = None
    if not args.force_restart:
        try:
            checkpoint = _reviseqa_load_checkpoint(out_path)
        except json.JSONDecodeError as e:
            print(f"[warn] malformed checkpoint ({e}); fresh run.")
    records = (checkpoint or {}).get("records", []) or []
    done = {r["scenario_id"] for r in records}
    if done:
        print(f"[{arm}] resume: {len(done)} scenarios already complete.")
    if checkpoint and checkpoint.get("usage"):
        for k, v in checkpoint["usage"].items():
            USAGE[k] = v

    order = {s["id"]: i for i, s in enumerate(scenarios)}
    lock = threading.Lock()

    config = {
        "arm": arm,
        "tested_backend": "openai",
        "tested_model": args.model,
        "tested_url": args.base_url,
        "data_dir": DATA_DIR,
        "task_setting": "explicit_no_correction_no_reasoning",
        "include_reasoning": False,
        "include_correction": False,
        "max_scenarios": args.max_scenarios,
        "max_tokens": int(os.environ.get("REVISEQA_MAX_TOKENS", "800")),
        "workers": args.workers,
        "temperature": 0,
        "notes": args.notes,
    }
    if arm == "two_arm":
        config.update({"engine_block_mode": "dep_map_only_no_state_summary",
                       "structured_update_source": "benchmark_native"})
    elif arm == "transcript_rag":
        config.update({"retriever": "TF-IDF cosine (scikit-learn), same "
                                    "as the published Qwen-7B TR arm",
                       "top_k": args.top_k,
                       "top_k_rationale": TR.K_RATIONALE})
    else:
        config.update({"interpreter_model": "gpt-4o (cached ops from "
                                            "experiments/results/"
                                            "reviseqa_interp_full)",
                       "interpreter_cache_dir": INTERP_CACHE_DIR})

    def save(status):
        recs = sorted(records, key=lambda r: order.get(r["scenario_id"],
                                                        1 << 30))
        agg = {}
        if recs:
            if arm == "two_arm":
                agg = {"aggregate_baseline_lcata":
                       _reviseqa_aggregate_lcata(recs, "baseline_lcata"),
                       "aggregate_hybrid_dep_only_lcata":
                       _reviseqa_aggregate_lcata(recs,
                                                 "hybrid_dep_only_lcata")}
            else:
                agg = {f"aggregate_{TRACE_KEY[arm]}":
                       _reviseqa_aggregate_lcata(recs, TRACE_KEY[arm])}
        _reviseqa_atomic_save(out_path, {
            "status": status,
            "task_setting": "explicit_no_correction_no_reasoning",
            "completed_scenario_ids": [r["scenario_id"] for r in recs],
            "n_completed": len(recs),
            "n_total_target": len(scenarios),
            **agg,
            "usage": dict(USAGE),
            "records": recs,
            "config": config,
        })

    def work(scenario):
        if arm == "two_arm":
            return evaluate_reviseqa_scenario(
                scenario, tested_api_key=api_key, base_url=args.base_url,
                model=args.model, include_reasoning=False,
                include_correction=False, run_baseline=True,
                run_hybrid=True)
        if arm == "transcript_rag":
            return TR.evaluate_scenario_tr(scenario, api_key, args.base_url,
                                           args.model, args.top_k)
        return evaluate_scenario_replay(
            scenario, interp_cache.get(scenario["id"], []), api_key,
            args.base_url, args.model)

    remaining = [s for s in scenarios if s["id"] not in done]
    print(f"[{arm}] {len(remaining)} scenarios to run with "
          f"{args.workers} workers -> {out_path}", flush=True)
    t0 = datetime.datetime.now()
    n_done_session = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s["id"] for s in remaining}
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # keep the sweep alive; log and go on
                print(f"  [{arm}] {sid} FAILED: {e!r}", flush=True)
                continue
            with lock:
                records.append(rec)
                n_done_session += 1
                if n_done_session % args.checkpoint_every == 0:
                    save("in_progress")
            key = TRACE_KEY[arm]
            print(f"  [{arm}] {len(records)}/{len(scenarios)} {sid} "
                  f"LCATA={rec.get(key)}", flush=True)
    save("completed" if len(records) >= len(scenarios) else "in_progress")
    dt = datetime.datetime.now() - t0
    recs = records
    print(f"\n[{arm}] DONE {len(recs)}/{len(scenarios)} in {dt}; "
          f"usage={USAGE}", flush=True)
    if arm == "two_arm":
        print("  baseline:", _reviseqa_aggregate_lcata(recs, "baseline_lcata"))
        print("  engine  :", _reviseqa_aggregate_lcata(recs,
                                                        "hybrid_dep_only_lcata"))
    else:
        print(f"  {arm}:", _reviseqa_aggregate_lcata(recs, TRACE_KEY[arm]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True,
                    help="output subdir name, e.g. qwen2.5-14b")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="VLLM_API_KEY",
                    help="env var holding the API key (default: dummy key "
                         "for local vLLM)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--max-scenarios", type=int, default=930)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=TR.TOP_K)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--force-restart", action="store_true")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "dummy")
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    scenarios = load_reviseqa(DATA_DIR, max_scenarios=args.max_scenarios)
    if os.path.isfile(REFERENCE_RUN):
        ref = json.load(open(REFERENCE_RUN))
        ref_ids = ref.get("completed_scenario_ids", [])
        ids = [s["id"] for s in scenarios]
        if ids != ref_ids[:len(ids)]:
            print("ERROR: scenario set/order mismatch vs reference run.")
            sys.exit(2)
        print(f"Loaded {len(scenarios)} scenarios (order verified vs reference)")
    else:
        print(f"Loaded {len(scenarios)} scenarios (no reference run at "
              f"{REFERENCE_RUN}; order check skipped)")
    print(f"model={args.model} url={args.base_url} workers={args.workers} "
          f"max_tokens={os.environ.get('REVISEQA_MAX_TOKENS', '800')}")

    interp_cache = load_interp_cache() if "e2e_replay" in args.arms else {}
    if "e2e_replay" in args.arms:
        missing = [i for i in ids if i not in interp_cache]
        print(f"interp cache: {len(interp_cache)} scenarios; "
              f"missing for {len(missing)} of the requested ones")
        if missing:
            print("ERROR: cached Interpreter ops missing; cannot replay.")
            sys.exit(3)

    for arm in args.arms:
        for k in USAGE:
            USAGE[k] = 0
        run_arm(arm, scenarios, args, api_key,
                os.path.join(out_dir, f"{arm}.json"), interp_cache)


if __name__ == "__main__":
    main()
