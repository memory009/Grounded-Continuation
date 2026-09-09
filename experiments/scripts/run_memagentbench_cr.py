#!/usr/bin/env python3
"""MemoryAgentBench Conflict-Resolution (FactConsolidation) three-arm runner.

Data: ai-hyz/MemoryAgentBench, Conflict_Resolution split (MIT). 8 rows =
{sh,mh} x {6k,32k,64k,262k}; each row is one numbered fact stream (later
serial numbers supersede earlier facts with the same (relation, subject)
key) plus 100 questions about the CURRENT value.

Arms (same QA model, same prompt shape, temperature 0):
  llm_full   the paper's long-context setting: the raw fact stream,
             tail-truncated to the QA context budget when it exceeds it
             (newest facts kept, which favours the baseline).
  tr         transcript-RAG: TF-IDF top-k facts over the FULL stream
             (stale and current mixed), serial numbers preserved.
  verifier   engine arm: facts are ingested in serial order into a
             supersession store keyed by (relation, subject) -- the
             benchmark-native structured update, no LLM extraction; the
             prompt gets TF-IDF top-k over ACTIVE (current) facts only,
             same retriever and same k as `tr`, so the ONLY difference
             is standing tracking.

Scoring: closed-form normalised containment against the gold answer list.
Metric: per-question accuracy per variant; paired exact McNemar
verifier-vs-tr (questions share one context per variant; reported as
question-level with that caveat).

Usage:
  python experiments/scripts/run_memagentbench_cr.py --variants sh_6k --max-questions 10   # smoke
  python experiments/scripts/run_memagentbench_cr.py                                       # all 8 variants, 100 q each
"""
import argparse, collections, datetime, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Defaults are the original server paths; both can be overridden with
# --project-dir / --data so the script also runs from a local checkout.
DEFAULT_PROJECT_DIR = os.environ.get(
    "CR_PROJECT_DIR", ".")
DEFAULT_DATA = os.environ.get(
    "CR_DATA", "datasets_cache/memoryagentbench/data/"
               "Conflict_Resolution-00000-of-00001.parquet")
# Backwards-compatible aliases: run_memagentbench_cr_e2e.py imports this
# module and reads CR.DATA / CR.PROJECT_DIR directly.
DATA = DEFAULT_DATA
PROJECT_DIR = DEFAULT_PROJECT_DIR

# ---------------------------------------------------------------- templates
# Relation templates observed in the data (specific first). Each maps a fact
# sentence to a supersession key (template_id, subject.lower()). A final
# last-" is " fallback covers rare title-style facts ("The Tánaiste is X").
TEMPLATES = [
    r"^The name of the current head of the (?P<s>.+?) government is (?P<o>.+)$",
    r"^The name of the current head of state in (?P<s>.+?) is (?P<o>.+)$",
    r"^The headquarters of (?P<s>.+?) is located in the city of (?P<o>.+)$",
    r"^The univeristy where (?P<s>.+?) was educated is (?P<o>.+)$",
    r"^The chief executive officer of (?P<s>.+?) is (?P<o>.+)$",
    r"^The origianl broadcaster of (?P<s>.+?) is (?P<o>.+)$",
    r"^The type of music that (?P<s>.+?) plays is (?P<o>.+)$",
    r"^The company that produced (?P<s>.+?) is (?P<o>.+)$",
    r"^The official language of (?P<s>.+?) is (?P<o>.+)$",
    r"^The head coach of (?P<s>.+?) is (?P<o>.+)$",
    r"^The chairperson of (?P<s>.+?) is (?P<o>.+)$",
    r"^The director of (?P<s>.+?) is (?P<o>.+)$",
    r"^The author of (?P<s>.+?) is (?P<o>.+)$",
    r"^The capital of (?P<s>.+?) is (?P<o>.+)$",
    r"^The Governor of (?P<s>.+?) is (?P<o>.+)$",
    r"^The Mayor of (?P<s>.+?) is (?P<o>.+)$",
    r"^(?P<s>.+?) is married to (?P<o>.+)$",
    r"^(?P<s>.+?) plays the position of (?P<o>.+)$",
    r"^(?P<s>.+?) is associated with the sport of (?P<o>.+)$",
    r"^(?P<s>.+?) was born in the city of (?P<o>.+)$",
    r"^(?P<s>.+?) died in the city of (?P<o>.+)$",
    r"^(?P<s>.+?) was founded in the city of (?P<o>.+)$",
    r"^(?P<s>.+?) was created in the country of (?P<o>.+)$",
    r"^(?P<s>.+?) was written in the language of (?P<o>.+)$",
    r"^(?P<s>.+?) was founded by (?P<o>.+)$",
    r"^(?P<s>.+?) was created by (?P<o>.+)$",
    r"^(?P<s>.+?) was developed by (?P<o>.+)$",
    r"^(?P<s>.+?) was performed by (?P<o>.+)$",
    r"^(?P<s>.+?) is located in the continent of (?P<o>.+)$",
    r"^(?P<s>.+?) is a citizen of (?P<o>.+)$",
    r"^(?P<s>.+?) is affiliated with the religion of (?P<o>.+)$",
    r"^(?P<s>.+?) is employed by (?P<o>.+)$",
    r"^(?P<s>.+?) is represented by the twitter account (?P<o>.+)$",
    r"^(?P<s>.+?) is famous for (?P<o>.+)$",
    r"^(?P<s>.+?) speaks the language of (?P<o>.+)$",
    r"^(?P<s>.+?) works in the field of (?P<o>.+)$",
    r"^(?P<s>.+?) worked in the city of (?P<o>.+)$",
]
_COMP = [re.compile(p) for p in TEMPLATES]


def fact_key(sent):
    s = sent.rstrip(".").strip()
    for i, c in enumerate(_COMP):
        m = c.match(s)
        if m:
            return (i, m.group("s").strip().lower())
    # fallback: split at the LAST " is " (title-style facts)
    if " is " in s:
        return ("is-fallback", s.rsplit(" is ", 1)[0].strip().lower())
    return ("opaque", s.lower())


def parse_stream(context):
    facts = []
    for line in context.split("\n"):
        m = re.match(r"^(\d+)\.\s+(.*?)\s*$", line.strip())
        if m:
            facts.append((int(m.group(1)), m.group(2).strip()))
    return facts


def build_state(facts):
    """Serial-order ingestion. Returns (active list [(serial, sent)], diag)."""
    current = {}
    n_supersede = 0
    for serial, sent in facts:
        k = fact_key(sent)
        if k in current:
            n_supersede += 1
        current[k] = (serial, sent)
    active = sorted(current.values())
    diag = {"n_facts": len(facts), "n_active": len(active),
            "n_superseded": len(facts) - len(active),
            "n_supersede_events": n_supersede,
            "n_fallback_keys": sum(1 for s_, t in facts
                                   if fact_key(t)[0] in ("is-fallback",
                                                         "opaque"))}
    return active, diag


# ---------------------------------------------------------------- chat call
USAGE = {"calls": 0, "failed": 0, "empty": 0,
         "prompt_tokens": 0, "completion_tokens": 0}
_ULOCK = threading.Lock()


def chat(prompt, args):
    import requests, time
    messages = [
        {"role": "system",
         "content": "Answer the question using only the provided facts. "
                    "Facts are indexed by serial numbers; when facts "
                    "conflict, the fact with the LARGER serial number is "
                    "current. Reply with the answer entity only, no "
                    "explanation."},
        {"role": "user", "content": prompt},
    ]
    backoff = 2
    for attempt in range(6):
        try:
            r = requests.post(args.base_url, timeout=600,
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {args.key}"},
                              json={"model": args.model, "temperature": 0,
                                    "max_tokens": 64, "messages": messages})
            if r.status_code == 429:
                time.sleep(backoff); backoff = min(backoff * 2, 32)
                continue
            d = r.json()
            if "error" in d:
                print(f"  [chat] API error (attempt {attempt+1}): "
                      f"{d['error'].get('message', d['error'])[:200]}",
                      flush=True)
            if "choices" in d and d["choices"]:
                u = d.get("usage", {}) or {}
                with _ULOCK:
                    USAGE["calls"] += 1
                    USAGE["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                    USAGE["completion_tokens"] += \
                        u.get("completion_tokens", 0) or 0
                c = d["choices"][0]["message"].get("content")
                if not c:
                    with _ULOCK:
                        USAGE["empty"] += 1
                return c
        except Exception as e:
            print(f"  [chat] error attempt {attempt+1}: {e}", flush=True)
        time.sleep(backoff); backoff = min(backoff * 2, 32)
    with _ULOCK:
        USAGE["failed"] += 1
    return None


def norm(x):
    x = (x or "").lower().strip().strip(".")
    x = re.sub(r"^(the|a|an)\s+", "", x)
    return re.sub(r"\s+", " ", x)


def correct(resp, golds):
    r = norm(resp)
    if not r:
        return False
    for g in golds:
        g = norm(g)
        if g and (g in r or (len(r) >= 3 and r in g)):
            return True
    return False


def fact_block(items):
    return "\n".join(f"{s}. {t}" for s, t in items)


def retrieve(pool, query, k):
    """Deterministic TF-IDF cosine; pool = [(serial, sent)]."""
    texts = [t for _, t in pool]
    try:
        vec = TfidfVectorizer(lowercase=True)
        mat = vec.fit_transform(texts + [query])
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    except ValueError:
        sims = np.zeros(len(texts))
    order = sorted(range(len(texts)), key=lambda i: (-float(sims[i]), i))
    return sorted(pool[i] for i in order[:k])   # serial order in the prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url",
                    default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--variants", nargs="+",
                    default=["sh_6k", "sh_32k", "sh_64k", "sh_262k",
                             "mh_6k", "mh_32k", "mh_64k", "mh_262k"])
    ap.add_argument("--arms", nargs="+",
                    default=["llm_full", "tr", "verifier"],
                    choices=["llm_full", "tr", "verifier"])
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=50,
                    help="facts retrieved for tr AND verifier (matched)")
    ap.add_argument("--ctx-char-budget", type=int, default=48000,
                    help="llm_full tail-truncation budget in characters "
                         "(~12K tokens, inside the 16K vLLM window)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--force-restart", action="store_true")
    ap.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR,
                    help="repo root; results are written under "
                         "<project-dir>/experiments/results/memagentbench_cr")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="path to Conflict_Resolution-*.parquet")
    args = ap.parse_args()
    args.key = os.environ.get(args.api_key_env, "dummy")
    sys.path.insert(0, args.project_dir)
    OUT_DIR = os.path.join(args.project_dir,
                           "experiments/results/memagentbench_cr")

    import pyarrow.parquet as pq
    t = pq.read_table(args.data)
    rows = {}
    for i in range(t.num_rows):
        meta = t.column("metadata")[i].as_py()
        vid = meta["qa_pair_ids"][0]              # factconsolidation_mh_6k_no0
        var = "_".join(vid.split("_")[1:3])       # mh_6k
        rows[var] = {
            "context": t.column("context")[i].as_py(),
            "questions": t.column("questions")[i].as_py(),
            "answers": t.column("answers")[i].as_py(),
        }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"cr_{args.tag}.json")
    state = {}
    if not args.force_restart and os.path.exists(out_path):
        try:
            state = json.load(open(out_path)).get("variants", {})
            print(f"[resume] {len(state)} variant records loaded")
        except json.JSONDecodeError:
            state = {}

    for var in args.variants:
        row = rows[var]
        facts = parse_stream(row["context"])
        active, diag = build_state(facts)
        qs = row["questions"][:args.max_questions]
        golds = row["answers"][:args.max_questions]
        print(f"\n== {var}: {diag['n_facts']} facts, {diag['n_active']} "
              f"active ({diag['n_superseded']} superseded, "
              f"{diag['n_fallback_keys']} fallback-keyed), "
              f"{len(qs)} questions", flush=True)

        rec = state.get(var, {"diag": diag, "per_question": {}})
        rec["diag"] = diag
        pq_rec = rec["per_question"]

        full_text = fact_block(facts)
        if len(full_text) > args.ctx_char_budget:
            # tail truncation keeps the newest facts (favours the baseline)
            cut = full_text[-args.ctx_char_budget:]
            full_text_trunc = cut[cut.index("\n") + 1:]
            trunc = True
        else:
            full_text_trunc, trunc = full_text, False

        def run_q(qi):
            q, gold = qs[qi], golds[qi]
            entry = pq_rec.get(str(qi), {})
            for arm in args.arms:
                if arm in entry:
                    continue
                if arm == "llm_full":
                    block = full_text_trunc
                elif arm == "tr":
                    block = fact_block(retrieve(facts, q, args.top_k))
                else:
                    block = fact_block(retrieve(active, q, args.top_k))
                prompt = (f"Here is a list of facts:\n{block}\n\n"
                          f"Question: {q}\nAnswer with the entity only.")
                resp = chat(prompt, args)
                entry[arm] = {"resp": resp, "correct": correct(resp, gold)}
            entry["gold"] = gold
            return qi, entry

        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_q, i) for i in range(len(qs))]
            done = 0
            for f in as_completed(futs):
                qi, entry = f.result()
                with lock:
                    pq_rec[str(qi)] = entry
                    done += 1
                    if done % 10 == 0:
                        state[var] = rec
                        _save(out_path, state, args, trunc_note=trunc)
        state[var] = rec
        _save(out_path, state, args, trunc_note=trunc)
        accs = {arm: sum(1 for e in pq_rec.values() if e.get(arm, {}).get("correct"))
                / max(len(pq_rec), 1) for arm in args.arms}
        print(f"   acc: " + "  ".join(f"{a}={v:.3f}" for a, v in accs.items())
              + f"   (llm_full truncated: {trunc})", flush=True)
    print("\nusage:", USAGE)


def _save(out_path, state, args, trunc_note=False):
    summary = {}
    for var, rec in state.items():
        pq_rec = rec.get("per_question", {})
        n = len(pq_rec)
        summary[var] = {"n": n, **{
            arm: round(sum(1 for e in pq_rec.values()
                           if e.get(arm, {}).get("correct")) / n, 4)
            for arm in ("llm_full", "tr", "verifier") if n and
            any(arm in e for e in pq_rec.values())}}
    tmp = out_path + ".tmp"
    json.dump({"config": {"model": args.model, "base_url": args.base_url,
                          "top_k": args.top_k,
                          "ctx_char_budget": args.ctx_char_budget,
                          "scoring": "normalised containment vs gold list",
                          "llm_full_truncation": "tail (newest kept)"},
               "summary": summary, "usage": USAGE, "variants": state},
              open(tmp, "w"), indent=None)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
