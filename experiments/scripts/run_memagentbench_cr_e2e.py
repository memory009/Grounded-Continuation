#!/usr/bin/env python3
"""MemoryAgentBench-CR END-TO-END verifier arm.

Difference from the native verifier arm (run_memagentbench_cr.py): the
supersession decisions are made by an LLM Interpreter (GPT-4o), not by the
template-key + serial-order rule. For each incoming fact, the Interpreter
sees the top-5 most similar facts currently active and decides which one
(if any) the new fact supersedes. Decisions are cached to JSON (resumable),
so the QA phase and any later QA model reuse the same ingestion.

Phases:
  ingest   stream the facts of a context through the Interpreter,
           producing an LLM-maintained active set (+ agreement stats vs the
           template-key store)
  qa       answer the variant's questions with the QA model over TF-IDF
           top-k of the LLM-maintained active set (same k and retriever as
           the native/tr arms)

Usage:
  python ... --phase ingest --lengths 6k 32k
  python ... --phase qa --variants sh_6k mh_6k sh_32k mh_32k --qa-model Qwen/Qwen2.5-7B-Instruct \
      --qa-base-url http://localhost:8000/v1/chat/completions
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = os.environ.get(
    "CR_PROJECT_DIR", ".")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "experiments", "scripts"))
import run_memagentbench_cr as CR  # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "experiments/results/memagentbench_cr")
INGEST_PATH = os.path.join(OUT_DIR, "e2e_ingest_{length}.json")

INTERP_SYSTEM = (
    "You maintain a store of facts. Facts can be superseded: a NEW fact "
    "supersedes an OLD fact if both state a value for the SAME attribute "
    "of the SAME subject (e.g. both say who X is married to, or what the "
    "capital of Y is), so the new value replaces the old one. Given a NEW "
    "fact and numbered CANDIDATE facts from the store, decide which "
    "candidate (if any) the new fact supersedes. Reply ONLY with JSON: "
    '{"supersedes": <candidate serial number or null>}.'
)


def interp_call(new_fact, candidates, key, url="https://api.openai.com/v1/chat/completions",
                model="gpt-4o"):
    import requests
    cand = "\n".join(f"{s}. {t}" for s, t in candidates) or "(store is empty)"
    msg = [{"role": "system", "content": INTERP_SYSTEM},
           {"role": "user", "content": f"CANDIDATES:\n{cand}\n\nNEW FACT:\n{new_fact}"}]
    backoff = 2
    for attempt in range(6):
        try:
            r = requests.post(url, timeout=120,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "temperature": 0,
                                    "max_tokens": 20, "messages": msg,
                                    "response_format": {"type": "json_object"}})
            if r.status_code == 429:
                time.sleep(backoff); backoff = min(backoff*2, 32); continue
            d = r.json()
            if "error" in d:
                print("  [interp]", str(d["error"])[:150], flush=True)
            if "choices" in d and d["choices"]:
                u = d.get("usage", {}) or {}
                try:
                    out = json.loads(d["choices"][0]["message"]["content"])
                    sup = out.get("supersedes")
                    return (int(sup) if sup is not None else None), u
                except Exception:
                    return None, u
        except Exception as e:
            print(f"  [interp] err {e}", flush=True)
        time.sleep(backoff); backoff = min(backoff*2, 32)
    return None, {}


def phase_ingest(args):
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    t = pq.read_table(CR.DATA)
    key = os.environ.get("OPENAI_API_KEY")
    assert key, "OPENAI_API_KEY not set"
    row_by_len = {}
    for i in range(t.num_rows):
        vid = t.column("metadata")[i].as_py()["qa_pair_ids"][0]
        _, hop, length = vid.split("_")[:3]
        row_by_len.setdefault(length, t.column("context")[i].as_py())
    for length in args.lengths:
        out_path = INGEST_PATH.format(length=length)
        cache = {}
        if os.path.exists(out_path) and not args.force_restart:
            cache = json.load(open(out_path)).get("decisions", {})
            print(f"[{length}] resume: {len(cache)} cached decisions")
        facts = CR.parse_stream(row_by_len[length])
        active = {}   # serial -> sent
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        t0 = time.time()
        for n, (serial, sent) in enumerate(facts):
            dec = cache.get(str(serial), "MISS")
            if dec == "MISS":
                cands = []
                if active:
                    pool = sorted(active.items())
                    texts = [x[1] for x in pool]
                    try:
                        vec = TfidfVectorizer(lowercase=True)
                        mat = vec.fit_transform(texts + [sent])
                        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
                        order = sorted(range(len(texts)),
                                       key=lambda i: (-float(sims[i]), i))[:5]
                        cands = [pool[i] for i in order]
                    except ValueError:
                        cands = pool[:5]
                dec, u = interp_call(sent, cands, key)
                usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
                usage["calls"] += 1
                cache[str(serial)] = dec
                if usage["calls"] % 50 == 0:
                    json.dump({"decisions": cache, "usage": usage},
                              open(out_path + ".tmp", "w"))
                    os.replace(out_path + ".tmp", out_path)
                    print(f"[{length}] {n+1}/{len(facts)} "
                          f"({(n+1)/(time.time()-t0+1e-9):.1f}/s)", flush=True)
            if dec is not None and int(dec) in active:
                del active[int(dec)]
            active[serial] = sent
        # agreement vs template-key store
        tmpl_active, diag = CR.build_state(facts)
        tset = {s for s, _ in tmpl_active}
        aset = set(active)
        json.dump({"decisions": cache, "usage": usage,
                   "n_active_llm": len(aset), "n_active_template": len(tset),
                   "active_agreement_jaccard":
                       round(len(aset & tset) / len(aset | tset), 4),
                   "template_diag": diag},
                  open(out_path + ".tmp", "w"))
        os.replace(out_path + ".tmp", out_path)
        print(f"[{length}] DONE: llm_active={len(aset)} "
              f"template_active={len(tset)} "
              f"jaccard={len(aset & tset)/len(aset | tset):.4f} "
              f"usage={usage}", flush=True)


def phase_qa(args):
    import pyarrow.parquet as pq
    t = pq.read_table(CR.DATA)
    rows = {}
    for i in range(t.num_rows):
        vid = t.column("metadata")[i].as_py()["qa_pair_ids"][0]
        var = "_".join(vid.split("_")[1:3])
        rows[var] = {"context": t.column("context")[i].as_py(),
                     "questions": t.column("questions")[i].as_py(),
                     "answers": t.column("answers")[i].as_py()}
    class A: pass
    qa = A(); qa.base_url = args.qa_base_url; qa.model = args.qa_model
    qa.key = os.environ.get(args.api_key_env, "dummy")
    out_path = os.path.join(OUT_DIR, f"cr_{args.tag}_e2e.json")
    state = {}
    if os.path.exists(out_path) and not args.force_restart:
        state = json.load(open(out_path)).get("variants", {})
    for var in args.variants:
        length = var.split("_")[1]
        ing = json.load(open(INGEST_PATH.format(length=length)))
        facts = CR.parse_stream(rows[var]["context"])
        dec = ing["decisions"]
        active = {}
        for serial, sent in facts:
            d_ = dec.get(str(serial))
            if d_ is not None and int(d_) in active:
                del active[int(d_)]
            active[serial] = sent
        pool = sorted(active.items())
        qs = rows[var]["questions"][:args.max_questions]
        golds = rows[var]["answers"][:args.max_questions]
        rec = state.get(var, {"per_question": {}})
        rec["n_active_llm"] = len(pool)
        pqr = rec["per_question"]
        print(f"== {var}: llm-active {len(pool)} facts, {len(qs)} questions",
              flush=True)
        lock = threading.Lock()
        def run_q(qi):
            if str(qi) in pqr:
                return qi, pqr[str(qi)]
            q, gold = qs[qi], golds[qi]
            block = CR.fact_block(CR.retrieve(pool, q, args.top_k))
            prompt = (f"Here is a list of facts:\n{block}\n\n"
                      f"Question: {q}\nAnswer with the entity only.")
            resp = CR.chat(prompt, qa)
            return qi, {"verifier_e2e": {"resp": resp,
                                          "correct": CR.correct(resp, gold)},
                        "gold": gold}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_q, i) for i in range(len(qs))]
            for f in as_completed(futs):
                qi, entry = f.result()
                with lock:
                    pqr[str(qi)] = entry
        state[var] = rec
        json.dump({"config": {"qa_model": args.qa_model, "top_k": args.top_k,
                              "interpreter": "gpt-4o (cached ingestion)"},
                   "variants": state}, open(out_path + ".tmp", "w"))
        os.replace(out_path + ".tmp", out_path)
        acc = sum(1 for e in pqr.values()
                  if e["verifier_e2e"]["correct"]) / len(pqr)
        print(f"   verifier_e2e acc={acc:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["ingest", "qa"])
    ap.add_argument("--lengths", nargs="+", default=["6k", "32k"])
    ap.add_argument("--variants", nargs="+",
                    default=["sh_6k", "mh_6k", "sh_32k", "mh_32k"])
    ap.add_argument("--qa-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--qa-base-url",
                    default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--force-restart", action="store_true")
    ap.add_argument("--api-key-env", default="VLLM_API_KEY",
                    help="env var holding the QA-model API key")
    args = ap.parse_args()
    if args.phase == "ingest":
        phase_ingest(args)
    else:
        phase_qa(args)


if __name__ == "__main__":
    main()
