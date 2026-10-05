#!/usr/bin/env python3
"""MemAB-FC through the engine, so selection is a dependency query.

The stream is ingested with EpistemicEngine.observe, one observation per fact.
Standing is the benchmark's larger-serial-wins rule: when a later fact carries
the same (template, subject) key, the earlier record is retracted through
EpistemicEngine.retract_assumption, so the engine's own state holds what is in
force.

A question is then treated as a candidate continuation. The engine opens a
hypothesis for the answer and records what it rests on: the active observations
reached from the entities named in the question by following recorded values to
the records that describe them. Dep(h_q) is that support set, and the QA model
reads exactly Dep(h_q).

  python experiments/scripts/run_cr_engine_native.py --variants mh_6k --check-only
"""
import argparse, importlib.util, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("cr", os.path.join(HERE, "run_memagentbench_cr.py"))
CR = importlib.util.module_from_spec(spec); spec.loader.exec_module(CR)
anch_src = open(os.path.join(HERE, "run_cr_anchored.py")).read().split("\ndef main()")[0]
import types
ANCH = types.ModuleType("anch"); ANCH.__file__ = os.path.join(HERE, "run_cr_anchored.py")
exec(compile(anch_src, "anch", "exec"), ANCH.__dict__)
from symbolic_engine import EpistemicEngine

OUT_DIR = os.path.join(ROOT, "experiments", "results", "memagentbench_cr")


def ingest(facts):
    """Serial-order ingestion into the engine. Later same-key facts retract the
    earlier record. Returns (engine, active_ids, by_subject_index)."""
    eng = EpistemicEngine()
    current = {}                      # (template, subject) -> obs id
    for serial, sent in facts:
        oid = f"o{serial}"
        eng.observe(oid, sent, turn=f"t{serial}", speaker="stream")
        k = CR.fact_key(sent)
        if k in current:
            eng.retract_assumption(current[k], transitive=True)
        current[k] = oid
    active_ids = set(current.values())
    by_subj = {}
    for oid in active_ids:
        sent = eng.observations[oid].content
        p = ANCH.parse_fact(sent)
        if p:
            by_subj.setdefault(ANCH.norm(p[1]), []).append((oid, sent, p[2]))
    return eng, active_ids, by_subj


def support_set(by_subj, question, depth, per_level_cap, total_cap):
    """The active observations a candidate answer to `question` rests on."""
    q = ANCH.norm(question)
    seeds = sorted([s for s in by_subj if len(s) > 2 and s in q], key=len, reverse=True)[:per_level_cap]
    picked, frontier, seen = {}, seeds, set(seeds)
    for _ in range(depth):
        nxt = []
        for subj in frontier:
            for oid, sent, obj in by_subj.get(subj, []):
                picked[oid] = sent
                o = ANCH.norm(obj)
                if o not in seen and o in by_subj:
                    seen.add(o); nxt.append(o)
        frontier = nxt[:per_level_cap]
        if not frontier:
            break
    items = sorted(picked.items(), key=lambda kv: int(kv[0][1:]))[:total_cap]
    return [(int(oid[1:]), sent) for oid, sent in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--variants", nargs="+", default=["sh_6k", "mh_6k", "sh_32k", "mh_32k"])
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--per-level-cap", type=int, default=8)
    ap.add_argument("--total-cap", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--check-only", action="store_true",
                    help="compare the engine's selection with the query-time one, no QA calls")
    ap.add_argument("--data", default=CR.DATA)
    args = ap.parse_args()
    args.key = os.environ.get(args.api_key_env, "dummy")

    import pyarrow.parquet as pq
    t = pq.read_table(args.data)
    rows = {}
    for i in range(t.num_rows):
        vid = t.column("metadata")[i].as_py()["qa_pair_ids"][0]
        rows["_".join(vid.split("_")[1:3])] = {
            "context": t.column("context")[i].as_py(),
            "questions": t.column("questions")[i].as_py(),
            "answers": t.column("answers")[i].as_py()}

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"cr_engine_{args.tag}.json")
    state = json.load(open(out_path)).get("variants", {}) if os.path.exists(out_path) else {}

    for var in args.variants:
        row = rows[var]
        facts = CR.parse_stream(row["context"])
        t0 = time.time()
        eng, active_ids, by_subj = ingest(facts)
        ingest_s = time.time() - t0
        ref_active, diag = CR.build_state(facts)          # dictionary pipeline, for the check
        same_state = {int(o[1:]) for o in active_ids} == {s for s, _ in ref_active}
        qs, golds = row["questions"][:args.max_questions], row["answers"][:args.max_questions]
        print(f"\n== {var}: ingested {len(facts)} facts in {ingest_s:.1f}s, "
              f"{len(active_ids)} active, state matches dictionary pipeline: {same_state}", flush=True)

        # selection check against the query-time implementation
        ref_by = ANCH.build_index(ref_active)
        mism = 0
        for qi, q in enumerate(qs):
            a = [s for s, _ in support_set(by_subj, q, args.depth, args.per_level_cap, args.total_cap)]
            b = [s for s, _ in ANCH.anchored_select(ref_by, q, args.depth, args.per_level_cap, args.total_cap)]
            if a != b:
                mism += 1
                if mism <= 3:
                    print(f"   selection differs on q{qi}: engine {a} vs query-time {b}")
        print(f"   selection identical on {len(qs)-mism}/{len(qs)} questions", flush=True)

        rec = state.get(var, {"diag": diag, "per_question": {}})
        rec["checks"] = {"state_matches_dictionary_pipeline": same_state,
                         "selection_matches_query_time": len(qs) - mism, "questions": len(qs),
                         "ingest_seconds": round(ingest_s, 1)}
        if args.check_only:
            state[var] = rec
            json.dump({"config": vars(args) | {"key": None}, "variants": state}, open(out_path, "w"), indent=1)
            continue

        pq_rec = rec["per_question"]

        def run_q(qi):
            q, gold = qs[qi], golds[qi]
            sel = support_set(by_subj, q, args.depth, args.per_level_cap, args.total_cap)
            hid = f"h_q{qi}"
            eng.hypothesize(hid, f"answer to: {q}", turn=f"q{qi}", speaker="qa")
            eng.set_dependencies(hid, [f"o{s}" for s, _ in sel])
            block = CR.fact_block(sel)
            prompt = (f"Here is a list of facts:\n{block}\n\n"
                      f"Question: {q}\nAnswer with the entity only.")
            resp = CR.chat(prompt, args)
            return qi, {"engine_dep": {"resp": resp, "correct": CR.correct(resp, gold),
                                       "n_facts": len(sel), "dep": [s for s, _ in sel]}}, gold

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(run_q, qi) for qi in range(len(qs))]):
                qi, out, gold = f.result()
                r = pq_rec.setdefault(str(qi), {"gold": gold}); r.update(out)
        vals = [r["engine_dep"]["correct"] for r in pq_rec.values() if "engine_dep" in r]
        rec["accuracy"] = {"engine_dep": round(sum(vals) / len(vals), 3)}
        state[var] = rec
        print(f"   {var}: {rec['accuracy']}", flush=True)
        json.dump({"config": vars(args) | {"key": None}, "variants": state}, open(out_path, "w"), indent=1)
    print("\nwritten to", out_path)


if __name__ == "__main__":
    main()
