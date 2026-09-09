#!/usr/bin/env python3
"""Offline token-cost accounting for the MemAB-FC (Conflict_Resolution) arms.

No API calls. Reconstructs, byte-identically, the prompts that
run_memagentbench_cr.py sends for each arm (llm_full / tr / verifier),
counts tokens with tiktoken, and reports per-query prompt cost as a
function of context length. Also reads the logged e2e Interpreter
ingestion usage (write-side cost) and computes token-denominated
break-even query counts.

Outputs experiments/results/memagentbench_cr/token_cost.json and a
printed summary table.

Usage:
  python experiments/scripts/compute_cr_token_cost.py
"""
import json, os, statistics, sys, time

import tiktoken

PROJECT_DIR = os.environ.get(
    "CR_PROJECT_DIR", ".")
sys.path.insert(0, os.path.join(PROJECT_DIR, "experiments/scripts"))
import run_memagentbench_cr as CR

RES_DIR = os.path.join(PROJECT_DIR, "experiments/results/memagentbench_cr")
VARIANTS = ["sh_6k", "sh_32k", "sh_64k", "sh_262k",
            "mh_6k", "mh_32k", "mh_64k", "mh_262k"]
TOP_K = 50
BUDGETS = {"local48k": 48_000, "api480k": 480_000}   # as-run char budgets
MAX_Q = 100

SYSTEM_MSG = ("Answer the question using only the provided facts. "
              "Facts are indexed by serial numbers; when facts "
              "conflict, the fact with the LARGER serial number is "
              "current. Reply with the answer entity only, no "
              "explanation.")


def user_msg(block, q):
    return (f"Here is a list of facts:\n{block}\n\n"
            f"Question: {q}\nAnswer with the entity only.")


def truncate_tail(full_text, budget):
    if len(full_text) > budget:
        cut = full_text[-budget:]
        return cut[cut.index("\n") + 1:], True
    return full_text, False


def main():
    enc = tiktoken.get_encoding("o200k_base")
    ntok = lambda s: len(enc.encode(s))
    sys_tok = ntok(SYSTEM_MSG)

    import pyarrow.parquet as pq
    t = pq.read_table(CR.DATA)
    rows = {}
    for i in range(t.num_rows):
        meta = t.column("metadata")[i].as_py()
        vid = meta["qa_pair_ids"][0]
        var = "_".join(vid.split("_")[1:3])
        rows[var] = {"context": t.column("context")[i].as_py(),
                     "questions": t.column("questions")[i].as_py(),
                     "answers": t.column("answers")[i].as_py()}

    out = {"tokenizer": "o200k_base", "top_k": TOP_K,
           "system_tokens": sys_tok, "variants": {}}

    for var in VARIANTS:
        t0 = time.time()
        row = rows[var]
        facts = CR.parse_stream(row["context"])
        active, diag = CR.build_state(facts)
        qs = row["questions"][:MAX_Q]

        full_text = CR.fact_block(facts)
        stream_tok = ntok(full_text)
        llm_full = {}
        for name, budget in BUDGETS.items():
            txt, trunc = truncate_tail(full_text, budget)
            llm_full[name] = {"block_tokens": ntok(txt), "truncated": trunc}

        tr_toks, ver_toks = [], []
        for q in qs:
            tr_block = CR.fact_block(CR.retrieve(facts, q, TOP_K))
            ver_block = CR.fact_block(CR.retrieve(active, q, TOP_K))
            tr_toks.append(sys_tok + ntok(user_msg(tr_block, q)))
            ver_toks.append(sys_tok + ntok(user_msg(ver_block, q)))

        # per-query prompt tokens for llm_full: system + wrapper + block + q
        q_wrap = [sys_tok + ntok(user_msg("", q)) for q in qs]
        llm_full_perq = {name: {
            "mean": round(statistics.mean(q_wrap) + d["block_tokens"], 1),
            "truncated": d["truncated"]} for name, d in llm_full.items()}

        rec = {
            "diag": diag,
            "stream_tokens": stream_tok,
            "llm_full_block_tokens": {n: d["block_tokens"]
                                      for n, d in llm_full.items()},
            "per_query_prompt_tokens": {
                "llm_full": llm_full_perq,
                "tr": {"mean": round(statistics.mean(tr_toks), 1),
                       "sd": round(statistics.pstdev(tr_toks), 1),
                       "min": min(tr_toks), "max": max(tr_toks)},
                "verifier": {"mean": round(statistics.mean(ver_toks), 1),
                             "sd": round(statistics.pstdev(ver_toks), 1),
                             "min": min(ver_toks), "max": max(ver_toks)},
            },
        }
        out["variants"][var] = rec
        print(f"{var:8s} stream={stream_tok:>7,}  "
              f"llm_full(api480k)={llm_full_perq['api480k']['mean']:>9,.0f}"
              f"{'*' if llm_full_perq['api480k']['truncated'] else ' '} "
              f"tr={rec['per_query_prompt_tokens']['tr']['mean']:>7,.0f}  "
              f"verifier={rec['per_query_prompt_tokens']['verifier']['mean']:>7,.0f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------ write-side (e2e ingest)
    out["e2e_ingest"] = {}
    for tag in ("6k", "32k"):
        p = os.path.join(RES_DIR, f"e2e_ingest_{tag}.json")
        if os.path.exists(p):
            u = json.load(open(p))["usage"]
            out["e2e_ingest"][f"sh_{tag}"] = u

    # token-denominated break-even: one-time write cost divided by the
    # per-query saving of the verifier arm vs full-context querying
    out["break_even_queries"] = {}
    for var, u in out["e2e_ingest"].items():
        write_tok = u["prompt_tokens"] + u["completion_tokens"]
        v = out["variants"][var]["per_query_prompt_tokens"]
        saving = v["llm_full"]["api480k"]["mean"] - v["verifier"]["mean"]
        out["break_even_queries"][var] = {
            "write_tokens": write_tok,
            "per_query_saving_tokens": round(saving, 1),
            "n_queries": round(write_tok / saving, 1) if saving > 0 else None,
        }

    # ------------------------------------------------ sanity vs logged usage
    # Llama full run: 2400 calls, 9,630,970 prompt tokens, 48k-char budget.
    pred = 0.0
    for var in VARIANTS:
        v = out["variants"][var]["per_query_prompt_tokens"]
        pred += MAX_Q * (v["llm_full"]["local48k"]["mean"]
                         + v["tr"]["mean"] + v["verifier"]["mean"])
    out["sanity"] = {
        "predicted_llama_run_prompt_tokens_o200k": round(pred),
        "logged_llama_run_prompt_tokens": 9_630_970,
        "note": "Llama uses its own tokenizer; ballpark agreement expected",
    }
    print(f"\nsanity: predicted(o200k)={pred:,.0f} vs logged(llama)=9,630,970")
    for var, be in out["break_even_queries"].items():
        print(f"break-even {var}: write={be['write_tokens']:,} tok, "
              f"saving/query={be['per_query_saving_tokens']:,.0f} tok "
              f"-> N*={be['n_queries']}")

    dst = os.path.join(RES_DIR, "token_cost.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
