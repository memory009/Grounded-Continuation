#!/usr/bin/env python
"""RECON counterfactual, oracle setting, LLM arm (Tier A+).

The LLM receives exactly what the symbolic engine received -- the chain's links with their
requires/establishes attributes and the trigger -- and is asked which steps lose support.
Scored on the same 64 scenarios / per-step judgements as run_recon_oracle.py, then paired
(exact McNemar) against Affected*.
"""
import argparse, collections, json, os, re, sys, time
from math import comb
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import run_recon_oracle as RO  # noqa: E402

SYSTEM = ("You are given a provenance chain from an investigation log. Each step lists the attributes it "
          "REQUIRES (established by earlier steps) and the attributes it ESTABLISHES. A step is supported only "
          "if every attribute it requires is established by a supported earlier step. Answer with JSON only.")

def prompt_full(skel, trigger):
    """RECON's oracle setting: the whole provenance skeleton, and the same question."""
    tdesc = next((L["description"] for ch in RO.chains(skel) for L in ch["links"] if L["entry_id"] == trigger), trigger)
    body = json.dumps(skel, separators=(",", ":"), ensure_ascii=False)
    return (f"Below is the complete structured skeleton of case {skel['case_id']} as JSON (provenance chains with "
            f"requires/establishes attributes, findings, revisions, and log entries).\n\n{body}\n\n"
            f"Suppose log entry {trigger} (\"{tdesc}\") is retracted and no longer establishes anything. "
            f"Which OTHER chain steps lose support as a consequence (directly or transitively)? "
            f"Return JSON: {{\"affected\": [\"log_xxxx\", ...]}} listing entry_ids only; use [] if none.")

def prompt_for(chain, trigger):
    lines = []
    for L in chain["links"]:
        lines.append(f"- {L['entry_id']} (step {L['step']}): {L['description']}\n"
                     f"    requires: {L.get('requires') or []}\n    establishes: {L.get('establishes') or []}")
    tdesc = next(L["description"] for L in chain["links"] if L["entry_id"] == trigger)
    return (f"CHAIN {chain['chain_id']}:\n" + "\n".join(lines) +
            f"\n\nSuppose step {trigger} (\"{tdesc}\") is retracted and no longer establishes anything. "
            f"Which OTHER steps in this chain lose support as a consequence (directly or transitively)? "
            f"Return JSON: {{\"affected\": [\"log_xxxx\", ...]}} listing entry_ids only; use [] if none.")

def chat(args, msgs):
    for attempt in range(4):
        try:
            r = requests.post(args.base_url, timeout=120,
                              headers={"Authorization": f"Bearer {os.environ[args.api_key_env]}"},
                              json={"model": args.model, "temperature": 0, "max_tokens": 300,
                                    "messages": msgs})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], r.json().get("usage", {})
            print(f"  [chat] HTTP {r.status_code}: {r.text[:120]}", flush=True)
        except Exception as e:
            print(f"  [chat] error attempt {attempt+1}: {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    return "", {}

def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--base-url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--data", default=os.path.join(REPO, "data", "recon"))
    ap.add_argument("--mode", choices=["chain", "full"], default="chain",
                    help="chain: isolated chain (~500 tok); full: entire skeleton (RECON oracle setting)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(REPO, "experiments", "results", "recon",
                                f"llm_oracle_{args.model}{'_full' if args.mode == 'full' else ''}.json")
    assert os.environ.get(args.api_key_env), f"{args.api_key_env} not set"

    cm_llm, cm_sym = collections.Counter(), collections.Counter()
    exact_llm = exact_sym = n_scen = 0; b = c = 0   # b: sym right & llm wrong ; c: llm right & sym wrong
    usage = collections.Counter(); records = []
    import glob
    for skp in sorted(glob.glob(os.path.join(args.data, "*", "case_*", "skeleton.json"))):
        skel = json.load(open(skp))
        if not any("requires" in L for ch in RO.chains(skel) for L in ch["links"]): continue
        owner = RO.build_engines(skel)
        chain_by_id = {ch["chain_id"]: ch for ch in RO.chains(skel)}
        for cs in skel["counterfactual_scenarios"]:
            t = cs["trigger_entry_id"]
            if t not in owner: continue
            eng, cid = owner[t]; chain = chain_by_id[cid]
            gold_dep, gold_ind = set(cs["dependent_steps"]), set(cs["independent_steps"])
            sym = RO.affected_iterated(eng, t)
            user_msg = prompt_full(skel, t) if args.mode == "full" else prompt_for(chain, t)
            raw, u = chat(args, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}])
            for k, v in u.items():
                if isinstance(v, int): usage[k] += v
            m = re.search(r"\{.*\}", raw, re.S)
            try: llm = set(json.loads(m.group(0)).get("affected", [])) if m else set()
            except Exception: llm = set(re.findall(r"log_\d{4}", raw))
            llm.discard(t)
            n_scen += 1
            exact_sym += (sym == gold_dep and not (sym & gold_ind))
            exact_llm += (llm == gold_dep and not (llm & gold_ind))
            for step in gold_dep | gold_ind:
                g = step in gold_dep; ps, pl = step in sym, step in llm
                cm_sym[(ps, g)] += 1; cm_llm[(pl, g)] += 1
                if (ps == g) and (pl != g): b += 1
                elif (pl == g) and (ps != g): c += 1
            records.append({"case": skel["case_id"], "trigger": t, "gold_dependent": sorted(gold_dep),
                            "gold_independent": sorted(gold_ind), "sym": sorted(sym), "llm": sorted(llm), "raw": raw[:400]})
            print(f"  {skel['case_id'][:20]} {t}: sym={'✓' if sym==gold_dep else '✗'} llm={'✓' if llm==gold_dep else '✗'}  ({n_scen})", flush=True)
    def prf(cm):
        tp, fp, fn, tn = cm[(True, True)], cm[(True, False)], cm[(False, True)], cm[(False, False)]
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 4), recall=round(R, 4))
    summ = {"model": args.model, "mode": args.mode, "scenarios": n_scen, "exact_sym": exact_sym, "exact_llm": exact_llm,
            "sym": prf(cm_sym), "llm": prf(cm_llm), "mcnemar_sym_vs_llm": {"sym_right_llm_wrong": b, "llm_right_sym_wrong": c, "p": mcn(b, c)},
            "usage": dict(usage)}
    print("\n" + json.dumps(summ, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summ, "records": records}, open(args.out, "w"), indent=1)
    print(f"written -> {args.out}")

if __name__ == "__main__":
    main()
