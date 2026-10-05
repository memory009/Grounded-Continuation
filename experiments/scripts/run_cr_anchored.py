#!/usr/bin/env python3
"""MemAB-FC: structural selection instead of similarity retrieval.

The `verifier` arm of run_memagentbench_cr.py selects what the QA model reads
by TF-IDF over the active set. This script adds an arm that selects by the
maintained structure alone:

  anchored   active records whose subject is named in the question, then the
             records reached by following their objects as subjects, to a
             fixed depth. No similarity ranking anywhere.

It is the MemAB-FC analogue of the ReviseQA verifier arm, where the dependency
map itself picks the premises. Reference arms `verifier` (top-k over the active
set) are recomputed in the same session for a paired comparison.

  python experiments/scripts/run_cr_anchored.py --variants mh_6k --max-questions 10
"""
import argparse, importlib.util, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("cr", os.path.join(HERE, "run_memagentbench_cr.py"))
CR = importlib.util.module_from_spec(spec); spec.loader.exec_module(CR)
OUT_DIR = os.path.join(ROOT, "experiments", "results", "memagentbench_cr")


def norm(x):
    return re.sub(r"\s+", " ", str(x).strip().lower().rstrip(".")).strip(' "\'')


def parse_fact(sent):
    s = sent.rstrip(".").strip()
    for i, c in enumerate(CR._COMP):
        m = c.match(s)
        if m:
            return (i, m.group("s").strip(), m.group("o").strip())
    if " is " in s:
        a, b = s.rsplit(" is ", 1)
        return ("is-fallback", a.strip(), b.strip())
    return None


def build_index(active):
    """subject -> [(serial, sentence, object)]"""
    by_subj = {}
    for serial, sent in active:
        p = parse_fact(sent)
        if p:
            by_subj.setdefault(norm(p[1]), []).append((serial, sent, p[2]))
    return by_subj


def anchored_select(by_subj, question, depth, per_level_cap, total_cap):
    """Records whose subject is named in the question, then their objects'
    records, to `depth` hops. Serial order, no similarity."""
    q = norm(question)
    seeds = [s for s in by_subj if len(s) > 2 and s in q]
    seeds.sort(key=len, reverse=True)          # prefer the most specific entity
    picked, frontier, seen = {}, seeds[:per_level_cap], set(seeds[:per_level_cap])
    for _ in range(depth):
        nxt = []
        for subj in frontier:
            for serial, sent, obj in by_subj.get(subj, []):
                picked[serial] = sent
                o = norm(obj)
                if o not in seen and o in by_subj:
                    seen.add(o); nxt.append(o)
        frontier = nxt[:per_level_cap]
        if not frontier:
            break
    items = sorted(picked.items())[:total_cap]
    return [(s, t) for s, t in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--variants", nargs="+", default=["sh_6k", "mh_6k", "sh_32k", "mh_32k"])
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--per-level-cap", type=int, default=8)
    ap.add_argument("--total-cap", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="qwen2.5-7b")
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
    out_path = os.path.join(OUT_DIR, f"cr_anchored_{args.tag}.json")
    state = json.load(open(out_path)).get("variants", {}) if os.path.exists(out_path) else {}

    for var in args.variants:
        row = rows[var]
        facts = CR.parse_stream(row["context"])
        active, diag = CR.build_state(facts)
        by_subj = build_index(active)
        qs, golds = row["questions"][:args.max_questions], row["answers"][:args.max_questions]
        print(f"\n== {var}: {diag['n_active']} active records, {len(by_subj)} distinct subjects, "
              f"{len(qs)} questions", flush=True)

        rec = state.get(var, {"diag": diag, "per_question": {}})
        pq_rec = rec["per_question"]

        def run_q(qi):
            q, gold = qs[qi], golds[qi]
            sel = anchored_select(by_subj, q, args.depth, args.per_level_cap, args.total_cap)
            out = {}
            for arm, block_items in (("anchored", sel),
                                     ("verifier", CR.retrieve(active, q, args.top_k))):
                block = CR.fact_block(block_items)
                prompt = (f"Here is a list of facts:\n{block}\n\n"
                          f"Question: {q}\nAnswer with the entity only.")
                resp = CR.chat(prompt, args)
                out[arm] = {"resp": resp, "correct": CR.correct(resp, gold),
                            "n_facts": len(block_items), "prompt_chars": len(prompt)}
            return qi, out, gold

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_q, qi) for qi in range(len(qs))]
            for n, f in enumerate(as_completed(futs), 1):
                qi, out, gold = f.result()
                r = pq_rec.setdefault(str(qi), {"gold": gold}); r.update(out)
                if n % 25 == 0:
                    print(f"   {n}/{len(qs)} ({time.time()-t0:.0f}s)", flush=True)

        acc, sizes = {}, {}
        for arm in ("anchored", "verifier"):
            vals = [r[arm]["correct"] for r in pq_rec.values() if arm in r]
            ns = [r[arm]["n_facts"] for r in pq_rec.values() if arm in r]
            acc[arm] = round(sum(vals) / len(vals), 3)
            sizes[arm] = {"mean_facts": round(sum(ns) / len(ns), 1), "max_facts": max(ns),
                          "empty": sum(1 for x in ns if x == 0)}
        rec["accuracy"], rec["selection"] = acc, sizes
        state[var] = rec
        print(f"   {var}: {acc}  selection={sizes}", flush=True)
        json.dump({"config": {"model": args.model, "top_k": args.top_k, "depth": args.depth,
                              "per_level_cap": args.per_level_cap, "total_cap": args.total_cap},
                   "variants": state}, open(out_path, "w"), indent=1)
    print("\nwritten to", out_path)


if __name__ == "__main__":
    main()
