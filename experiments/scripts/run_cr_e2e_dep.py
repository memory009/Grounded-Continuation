#!/usr/bin/env python3
"""Dependency selection on the Interpreter-maintained store (MemAB-FC).

The engine arm builds its state from the benchmark's own supersession key. This
script runs the same selection over the state the GPT-4o Interpreter maintained
end to end, replayed from the recorded ingestion decisions, so the two differ
only in where the state came from.

  python experiments/scripts/run_cr_e2e_dep.py --variants mh_6k --check-only
"""
import argparse, json, os, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments/scripts"))
import run_memagentbench_cr as CR
src = open(os.path.join(ROOT, "experiments/scripts/run_cr_engine_native.py")).read().split("\ndef main()")[0]
EN = types.ModuleType("en"); EN.__file__ = os.path.join(ROOT, "experiments/scripts/run_cr_engine_native.py")
exec(compile(src, "en", "exec"), EN.__dict__)
from symbolic_engine import EpistemicEngine

RES = os.path.join(ROOT, "experiments/results/memagentbench_cr")


def replay(facts, decisions):
    """The Interpreter-maintained active set, replayed from its decisions."""
    active = {}
    for serial, sent in facts:
        dec = decisions.get(str(serial), None)
        if dec is not None and int(dec) in active:
            del active[int(dec)]
        active[serial] = sent
    return sorted(active.items())


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
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--data", default=CR.DATA)
    args = ap.parse_args()
    args.key = os.environ.get(args.api_key_env, "dummy")

    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t = pq.read_table(args.data); rows = {}
    for i in range(t.num_rows):
        vid = t.column("metadata")[i].as_py()["qa_pair_ids"][0]
        rows["_".join(vid.split("_")[1:3])] = {
            "context": t.column("context")[i].as_py(),
            "questions": t.column("questions")[i].as_py(),
            "answers": t.column("answers")[i].as_py()}

    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"cr_e2e_dep_{args.tag}.json")
    state = json.load(open(out_path)).get("variants", {}) if os.path.exists(out_path) else {}

    for var in args.variants:
        length = var.split("_")[1]
        ing_path = os.path.join(ROOT, "experiments/cache/memagentbench_cr", f"e2e_ingest_{length}.json")
        if not os.path.exists(ing_path):
            print(f"== {var}: no end-to-end ingestion recorded, skipped"); continue
        ing = json.load(open(ing_path))
        row = rows[var]
        facts = CR.parse_stream(row["context"])
        active = replay(facts, ing["decisions"])
        assert len(active) == ing["n_active_llm"], (len(active), ing["n_active_llm"])
        tmpl, _ = CR.build_state(facts)
        jac = len({s for s, _ in active} & {s for s, _ in tmpl}) / len({s for s, _ in active} | {s for s, _ in tmpl})
        by_subj = {}
        for serial, sent in active:
            p = EN.ANCH.parse_fact(sent)
            if p:
                by_subj.setdefault(EN.ANCH.norm(p[1]), []).append((f"o{serial}", sent, p[2]))
        qs, golds = row["questions"][:args.max_questions], row["answers"][:args.max_questions]
        print(f"\n== {var}: replayed {len(active)} active records "
              f"(recorded {ing['n_active_llm']}), Jaccard vs template store {jac:.4f}", flush=True)

        rec = state.get(var, {"n_active": len(active), "jaccard_vs_template": round(jac, 4),
                              "per_question": {}})
        if args.check_only:
            state[var] = rec
            json.dump({"variants": state}, open(out_path, "w"), indent=1); continue

        pq_rec = rec["per_question"]

        def run_q(qi):
            q, gold = qs[qi], golds[qi]
            sel = EN.support_set(by_subj, q, args.depth, args.per_level_cap, args.total_cap)
            block = CR.fact_block(sel)
            prompt = (f"Here is a list of facts:\n{block}\n\n"
                      f"Question: {q}\nAnswer with the entity only.")
            resp = CR.chat(prompt, args)
            return qi, {"e2e_dep": {"resp": resp, "correct": CR.correct(resp, gold),
                                    "n_facts": len(sel)}}, gold

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(run_q, qi) for qi in range(len(qs))]):
                qi, o, gold = f.result()
                r = pq_rec.setdefault(str(qi), {"gold": gold}); r.update(o)
        vals = [r["e2e_dep"]["correct"] for r in pq_rec.values() if "e2e_dep" in r]
        rec["accuracy"] = {"e2e_dep": round(sum(vals) / len(vals), 3)}
        state[var] = rec
        print(f"   {var}: {rec['accuracy']}", flush=True)
        json.dump({"variants": state}, open(out_path, "w"), indent=1)
    print("\nwritten to", out_path)


if __name__ == "__main__":
    main()
