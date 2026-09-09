#!/usr/bin/env python3
"""ReviseQA interpreter-in-the-loop FULL run (930 scenarios, sharded).

Identical config to the 15-scenario smoke
(experiments/scripts/run_reviseqa_interp_smoke.py /
 experiments/results/reviseqa_interp_smoke/):
  - Interpreter: gpt-4o (api.openai.com, OPENAI_API_KEY from .env.local),
    input = engine's tracked premises (id: text) + the edit's explicit NL
    delta text (_reviseqa_format_explicit_context — same text the QA
    model sees); output = {"ops": [...]} remove-by-id / add-(kind,text).
  - QA: Qwen/Qwen2.5-7B-Instruct on local vLLM, hybrid-style growing
    dialogue with dep-map-only engine block built from the
    interpreter-maintained engine.
  - Benchmark-native initialization (interpreter only in the loop for
    the 7 edits), explicit_no_correction_no_reasoning, closed-form
    True/False/Uncertain scoring, LCATA@2/4/7.
  - Edit-translation accuracy vs benchmark-native structured updates as
    ground truth (resolved on an identically-seeded native shadow).

Scope: ALL 930 verified scenarios (same list/order as the two-arm full
run, asserted at startup), INCLUDING the 15 smoke scenarios — re-run
fresh here so every scenario in this output comes from one uniform run
(the smoke used the identical config; its copies live separately in
experiments/results/reviseqa_interp_smoke/).

Sharding: shard i of N takes scenarios[i::N] (interleaved, preserves
global order coverage and balances load). Each shard checkpoints and
resumes independently via its own --output file.

Token usage / cost: every gpt-4o response's `usage` field is
accumulated per shard and written to the output; cost is estimated at
the assumed gpt-4o rates USD 2.50 / 1M input tokens and USD 10.00 / 1M
output tokens (documented in the output; local Qwen QA calls are free
and not counted).

429 handling: exponential backoff (2,4,8,16,32,64 s) on rate-limit and
transient errors, up to 7 attempts per call.
"""

import argparse
import datetime
import json
import os
import random
import re
import sys
import time

PROJECT_DIR = "."
sys.path.insert(0, PROJECT_DIR)

from benchmark_adapter import (  # noqa: E402
    load_reviseqa,
    _reviseqa_init_scenario_state,
    _reviseqa_apply_edit,
    _reviseqa_engine_block_dep_only,
    _reviseqa_dep_map_stats_from_engine,
    _reviseqa_format_explicit_context,
    _reviseqa_fol_key,
    _reviseqa_parse_answer,
    _reviseqa_chat_call,
    _reviseqa_atomic_save,
    _reviseqa_load_checkpoint,
    REVISEQA_SYSTEM,
    REVISEQA_PROMPT_TEMPLATE,
)

DATA_DIR = "datasets_cache/reviseqa/reviseqa_data/nl/verified"
REFERENCE_RUN = os.path.join(
    PROJECT_DIR, "experiments/results/reviseqa/"
    "reviseqa_full_qwen7b_explicit_no_corr_no_reasoning.json")
OUT_DIR = os.path.join(PROJECT_DIR, "experiments/results/reviseqa_interp_full")

QA_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QA_URL = "http://localhost:8000/v1/chat/completions"
INTERP_MODEL = "gpt-4o"
INTERP_URL = "https://api.openai.com/v1/chat/completions"

# Assumed gpt-4o pricing (USD per 1M tokens) for cost estimation.
PRICE_IN_PER_M = 2.50
PRICE_OUT_PER_M = 10.00

INTERPRETER_SYSTEM = (
    "You are the update Interpreter of an epistemic state engine. Given "
    "the engine's currently tracked premises (each with an id) and a "
    "natural-language edit to the underlying context, translate the edit "
    "into engine operations.\n"
    "Output ONLY a JSON object of the form {\"ops\": [...]} — no "
    "markdown, no commentary. Each op is one of:\n"
    "  {\"op\": \"remove\", \"id\": \"<id of the tracked premise that "
    "the edit removes>\"}\n"
    "  {\"op\": \"add\", \"kind\": \"fact\" or \"rule\", \"text\": "
    "\"<the added statement, copied verbatim>\"}\n"
    "A 'rule' is a conditional or quantified statement (if/then, "
    "for-all, either/or constraints); a 'fact' is an atomic statement "
    "about an individual. Match each removed statement to the tracked "
    "premise id whose text has the same meaning. Emit one op per "
    "removed or added statement, and nothing for unchanged premises."
)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().rstrip(".")).lower()


def chat_with_usage(messages, api_key, url, model, temperature=0,
                    max_tokens=1500, max_retries=7):
    """OpenAI-compatible chat call returning (content, usage). 429-aware
    exponential backoff."""
    import requests
    backoff = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json={"model": model, "temperature": temperature,
                      "max_tokens": max_tokens, "messages": messages},
                timeout=120)
            if resp.status_code == 429:
                print(f"  [interp] 429 rate-limited (attempt {attempt+1}); "
                      f"backing off {backoff}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            data = resp.json()
            if "choices" in data and data["choices"]:
                usage = data.get("usage", {}) or {}
                return data["choices"][0]["message"]["content"], usage
            print(f"  [interp] API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', data)}", flush=True)
        except Exception as e:
            print(f"  [interp] request error (attempt {attempt+1}): {e}",
                  flush=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, 64)
    return None, {}


def _interp_retract_by_id(engine, active: set, dep: set, oid: str) -> str:
    txt = ""
    if oid in getattr(engine, "observations", {}):
        txt = engine.observations[oid].content
        del engine.observations[oid]
    if oid in getattr(engine, "awareness", set()):
        engine.awareness.discard(oid)
    for _, deps in list(getattr(engine, "dependencies", {}).items()):
        deps.discard(oid)
    active.discard(oid)
    dep.discard(oid)
    return txt


def _interp_add(engine, active: set, dep: set, counter: dict,
                kind: str, text: str) -> str:
    prefix = "f" if kind == "fact" else "r"
    counter[prefix] = counter.get(prefix, 0) + 1
    oid = f"{prefix}_{counter[prefix]}_i"
    engine.observe(oid, text, turn="reviseqa", speaker="interpreter")
    active.add(oid)
    dep.add(oid)
    return oid


def _parse_interp_ops(raw) -> list[dict]:
    if not raw:
        return []
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
               flags=re.DOTALL).strip()
    try:
        d = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return []
        try:
            d = json.loads(m.group(0))
        except Exception:
            return []
    ops = d.get("ops") if isinstance(d, dict) else None
    return ops if isinstance(ops, list) else []


def _gt_ops_for_edit(edit: dict, shadow) -> dict:
    engine, fol_to_id, active = shadow["engine"], shadow["fol_to_id"], \
        shadow["active"]
    delta = edit.get("edits_made", {}) or {}
    removed_texts, removed_missing = [], 0
    for coll in ("removed_facts", "removed_rules"):
        for item in delta.get(coll) or []:
            key = _reviseqa_fol_key(item["fol"])
            oid = fol_to_id.get(key)
            if oid and oid in engine.observations and oid in active:
                removed_texts.append(engine.observations[oid].content)
            else:
                removed_missing += 1
    added = ([("fact", f["nl"]) for f in delta.get("added_facts") or []]
             + [("rule", r["nl"]) for r in delta.get("added_rules") or []])
    return {"removed_texts": removed_texts,
            "removed_missing": removed_missing,
            "added": added}


def evaluate_scenario_interp(scenario, interp_key, qa_key, native_rec,
                             usage_tot):
    sid = scenario["id"]
    edits = scenario.get("edits", [])
    conclusion = scenario.get("conclusion", "")
    orig_ctx_text = "\n".join(scenario.get("original_context", []) or [])
    base_q = f"Does the context entail the conclusion '{conclusion}'?"
    demo_assistant = json.dumps({"reasoning": "",
                                 "answer": scenario.get("answer",
                                                        "Uncertain")})

    pipe_i, fol2id_i, active_i, dep_i, counter_i, init_diag = \
        _reviseqa_init_scenario_state(scenario)
    eng_i = pipe_i.engine
    pipe_n, fol2id_n, active_n, dep_n, counter_n, _ = \
        _reviseqa_init_scenario_state(scenario)
    shadow = {"engine": pipe_n.engine, "fol_to_id": fol2id_n,
              "active": active_n}

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

        gt_ops = _gt_ops_for_edit(edit, shadow)
        _reviseqa_apply_edit(shadow["engine"], fol2id_n, active_n,
                             dep_n, counter_n, edit)

        premises = "\n".join(
            f"{oid}: {eng_i.observations[oid].content}"
            for oid in sorted(active_i)
            if oid in eng_i.observations)
        interp_user = (f"Tracked premises:\n{premises}\n\n"
                       f"Edit:\n{delta_ctx}\n\n"
                       "Translate this edit into ops.")
        interp_raw, usage = chat_with_usage(
            [{"role": "system", "content": INTERPRETER_SYSTEM},
             {"role": "user", "content": interp_user}],
            interp_key, INTERP_URL, INTERP_MODEL)
        usage_tot["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        usage_tot["completion_tokens"] += \
            usage.get("completion_tokens", 0) or 0
        usage_tot["calls"] += 1
        ops = _parse_interp_ops(interp_raw)

        i_removed_texts, i_added = [], []
        n_bad_ops = 0
        for op in ops:
            if not isinstance(op, dict):
                n_bad_ops += 1
                continue
            if op.get("op") == "remove" and op.get("id"):
                txt = _interp_retract_by_id(eng_i, active_i, dep_i,
                                            str(op["id"]))
                if txt:
                    i_removed_texts.append(txt)
                else:
                    n_bad_ops += 1
            elif (op.get("op") == "add"
                  and op.get("kind") in ("fact", "rule")
                  and op.get("text")):
                _interp_add(eng_i, active_i, dep_i, counter_i,
                            op["kind"], op["text"])
                i_added.append((op["kind"], op["text"]))
            else:
                n_bad_ops += 1
        eng_i.dependencies["h1"] = {o for o in dep_i if o in active_i}

        gt_rem = {_norm(t) for t in gt_ops["removed_texts"]}
        in_rem = {_norm(t) for t in i_removed_texts}
        gt_add = {(k, _norm(t)) for k, t in gt_ops["added"]}
        in_add = {(k, _norm(t)) for k, t in i_added}
        exact = (gt_rem == in_rem and gt_add == in_add and n_bad_ops == 0)

        engine_block = _reviseqa_engine_block_dep_only(eng_i, "h1")
        user_msg = (engine_block + "\n" + REVISEQA_PROMPT_TEMPLATE.format(
            context=delta_ctx, question=step_q))
        messages.append({"role": "user", "content": user_msg})
        qa_raw = _reviseqa_chat_call(messages, qa_key, QA_URL, QA_MODEL)
        messages.append({"role": "assistant", "content": qa_raw or ""})
        pred = _reviseqa_parse_answer(qa_raw)
        correct = (pred == gt)
        trace.append(bool(correct))

        per_step.append({
            "step_idx": step_idx,
            "edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "gt": gt,
            "interp": {
                "raw": interp_raw,
                "ops": ops,
                "n_bad_ops": n_bad_ops,
                "removed_texts": i_removed_texts,
                "added": i_added,
                "gt_removed_texts": gt_ops["removed_texts"],
                "gt_removed_missing_in_native": gt_ops["removed_missing"],
                "gt_added": gt_ops["added"],
                "translation_exact": exact,
            },
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
        "hybrid_native_trace": (native_rec or {}).get(
            "hybrid_dep_only_trace"),
        "hybrid_native_lcata": (native_rec or {}).get(
            "hybrid_dep_only_lcata"),
        "n_edits_exact_translation":
            sum(1 for s in per_step if s["interp"]["translation_exact"]),
        "per_step": per_step,
    }


def _agg_lcata(records, key):
    out = {}
    for k in ("k2", "k4", "k7"):
        vals = [r[key][k] for r in records if r.get(key)]
        out[k] = (round(sum(vals) / len(vals), 4) if vals else None,
                  len(vals))
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4) if p is not None else None,
            "recall": round(r, 4) if r is not None else None}


def _translation_stats(records):
    tot = ex = 0
    rem = [0, 0, 0]
    add = [0, 0, 0]
    for r in records:
        for s in r["per_step"]:
            i = s["interp"]
            tot += 1
            ex += int(i["translation_exact"])
            gt_rem = {_norm(t) for t in i["gt_removed_texts"]}
            in_rem = {_norm(t) for t in i["removed_texts"]}
            gt_add = {(k, _norm(t)) for k, t in i["gt_added"]}
            in_add = {(k, _norm(t)) for k, t in i["added"]}
            rem[0] += len(gt_rem & in_rem)
            rem[1] += len(in_rem - gt_rem)
            rem[2] += len(gt_rem - in_rem)
            add[0] += len(gt_add & in_add)
            add[1] += len(in_add - gt_add)
            add[2] += len(gt_add - in_add)
    return {"n_edits": tot, "n_exact": ex,
            "exact_accuracy": round(ex / tot, 4) if tot else None,
            "removals": _prf(*rem), "additions": _prf(*add)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--force-restart", action="store_true")
    args = ap.parse_args()
    assert 0 <= args.shard_index < args.num_shards

    out_path = args.output or os.path.join(
        OUT_DIR, f"interp_shard{args.shard_index}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    interp_key = os.environ.get("OPENAI_API_KEY")
    if not interp_key:
        print("ERROR: OPENAI_API_KEY not set (source .env.local).")
        sys.exit(1)
    qa_key = "dummy"

    scenarios = load_reviseqa(DATA_DIR, max_scenarios=930)
    native_by_id = {}
    if os.path.isfile(REFERENCE_RUN):
        full = json.load(open(REFERENCE_RUN))
        ref_ids = full.get("completed_scenario_ids", [])
        if [s["id"] for s in scenarios] != ref_ids:
            print("ERROR: scenario set/order mismatch vs reference run.")
            sys.exit(2)
        # native two-arm records are copied alongside the interpreter trace
        native_by_id = {r["scenario_id"]: r for r in full["records"]}
    else:
        print(f"No reference two-arm run at {REFERENCE_RUN}; "
              "order check skipped, native traces left empty.")

    shard = scenarios[args.shard_index::args.num_shards]
    print(f"Shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard)} scenarios "
          f"({shard[0]['id']} ... {shard[-1]['id']})")

    checkpoint = None
    if not args.force_restart:
        try:
            checkpoint = _reviseqa_load_checkpoint(out_path)
        except json.JSONDecodeError as e:
            print(f"[warn] malformed checkpoint ({e}); fresh run.")
    records = (checkpoint or {}).get("records", []) or []
    usage_tot = (checkpoint or {}).get("interp_usage") or \
        {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    done_ids = {r["scenario_id"] for r in records}
    if done_ids:
        print(f"[resume] {len(done_ids)} scenarios already complete.")

    config = {
        "arm": "interpreter_in_the_loop",
        "interpreter_model": INTERP_MODEL,
        "interpreter_url": INTERP_URL,
        "qa_model": QA_MODEL,
        "qa_url": QA_URL,
        "task_setting": "explicit_no_correction_no_reasoning",
        "engine_block_mode": "dep_map_only_no_state_summary",
        "interpreter_input": ("tracked premises (id: text) + explicit NL "
                              "edit delta (_reviseqa_format_explicit_"
                              "context), identical to the 15-scenario "
                              "smoke"),
        "initialization": "benchmark-native; interpreter only for edits",
        "ground_truth": ("benchmark-native structured updates on an "
                         "identically-seeded native shadow state"),
        "sharding": {"num_shards": args.num_shards,
                     "shard_index": args.shard_index,
                     "rule": "scenarios[shard_index::num_shards] over the "
                             "same 930-scenario ordered list as the "
                             "two-arm full run"},
        "smoke_overlap": ("the 15 smoke scenarios are INCLUDED and re-run "
                          "fresh here (identical config to the smoke), so "
                          "this output is one uniform 930-scenario run"),
        "pricing_assumed_usd_per_1M": {"input": PRICE_IN_PER_M,
                                       "output": PRICE_OUT_PER_M},
    }

    def _cost(u):
        return round(u["prompt_tokens"] / 1e6 * PRICE_IN_PER_M
                     + u["completion_tokens"] / 1e6 * PRICE_OUT_PER_M, 4)

    def save(status):
        _reviseqa_atomic_save(out_path, {
            "status": status,
            "task_setting": "explicit_no_correction_no_reasoning",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "completed_scenario_ids": [r["scenario_id"] for r in records],
            "n_completed": len(records),
            "n_total_target": len(shard),
            "aggregate_interp_lcata": _agg_lcata(records, "interp_lcata"),
            "aggregate_hybrid_native_lcata_same_scenarios":
                _agg_lcata(records, "hybrid_native_lcata"),
            "edit_translation": _translation_stats(records),
            "interp_usage": usage_tot,
            "interp_cost_usd_estimate": _cost(usage_tot),
            "records": records,
            "config": config,
        })

    t0 = datetime.datetime.now()
    n_done_session = 0
    for scenario in shard:
        sid = scenario["id"]
        if sid in done_ids:
            continue
        idx = len(records) + 1
        print(f"[{idx}/{len(shard)}] {sid}", flush=True)
        rec = evaluate_scenario_interp(scenario, interp_key, qa_key,
                                       native_by_id.get(sid), usage_tot)
        records.append(rec)
        n_done_session += 1
        print(f"    interp_trace={rec['interp_trace']} "
              f"LCATA={rec['interp_lcata']} | "
              f"exact {rec['n_edits_exact_translation']}/7 | "
              f"cost_so_far=${_cost(usage_tot)}", flush=True)
        save("in_progress")

    save("completed")
    dt = datetime.datetime.now() - t0
    agg = _agg_lcata(records, "interp_lcata")
    ts = _translation_stats(records)
    print(f"\nSHARD {args.shard_index} COMPLETE: {len(records)} scenarios "
          f"in {dt} (this session: {n_done_session})")
    print(f"  LCATA: {agg}")
    print(f"  translation exact: {ts['n_exact']}/{ts['n_edits']} "
          f"({ts['exact_accuracy']})")
    print(f"  gpt-4o usage: {usage_tot} -> est ${_cost(usage_tot)}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
