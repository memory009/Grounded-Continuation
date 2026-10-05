#!/usr/bin/env python
"""RECON counterfactual task, oracle setting: the paper's symbolic engine is
handed the released provenance skeleton and asked Affected(trigger).

Two semantics are scored side by side:
  one-step   Affected(p)  = {a : p in Dep(a)}            (Prop. dep-sound, engine.get_affected as-is)
  iterated   Affected*(p) = least fixpoint of one-step   (what 'propagates through the graph' means)

Gold: skeleton.counterfactual_scenarios[].dependent_steps / independent_steps.
Only domains whose skeleton serialises requires/establishes are usable
(medical, finance); crime is skipped and logged.
"""
import argparse, collections, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from symbolic_engine import EpistemicEngine  # noqa: E402

def chains(s):
    return s.get("diagnostic_chains") or s.get("evidence_chains") or s.get("evaluation_chains") or []

def build_engines(skel):
    """One engine per chain (the generator keeps provenance per chain). Returns {entry_id: (engine, chain_id)}."""
    owner = {}
    for ch in chains(skel):
        eng = EpistemicEngine()
        est = collections.defaultdict(list)
        for L in ch["links"]:
            deps = sorted({src for a in (L.get("requires") or []) for src in est.get(a, [])})
            eng.hypothesize(L["entry_id"], L["description"], turn=f"step{L['step']}", speaker="log",
                            depends_on=deps)
            for a in (L.get("establishes") or []):
                est[a].append(L["entry_id"])
            owner[L["entry_id"]] = (eng, ch["chain_id"])
    return owner

def affected_one_step(eng, p):
    return set(eng.get_affected(p))

def affected_iterated(eng, p):
    seen, frontier = set(), [p]
    while frontier:
        q = frontier.pop()
        for a in eng.get_affected(q):
            if a not in seen:
                seen.add(a); frontier.append(a)
    return seen

def score_mcq(x, skel, aff, owner):
    """Map counterfactual MCQ/integer questions onto Affected*; returns (score, covered) or (None, False)."""
    fmt, qtext = x["format"], x["question"].lower()
    desc = {L["entry_id"]: L["description"] for ch in chains(skel) for L in ch["links"]}
    STOP = {"the","a","an","of","for","in","on","at","to","by","and","or","was","were","is","are","that",
            "this","with","from","which","would","have","had","been","not","note","noted","documented",
            "patient","patients","dr","obtained","performed","completed","during","provided","conducted"}
    def toks(t):
        return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) >= 3 and w not in STOP}
    dtok = {eid: toks(d) for eid, d in desc.items()}
    def step_of(text, thr=0.35):
        """Option texts paraphrase chain-link descriptions; match by content-word overlap coefficient
        (|A∩B| / min|A|,|B|), argmax over links, fixed a-priori threshold."""
        a = toks(text)
        if not a: return None
        best, bs = None, 0.0
        for eid, b in dtok.items():
            if not b: continue
            sc = len(a & b) / min(len(a), len(b))
            if sc > bs: best, bs = eid, sc
        return best if bs >= thr else None
    negq = bool(re.search(r"\bnot\b|no longer|prevent", qtext))
    if fmt == "mcq_multiple" and negq:
        pred = [o["label"] for o in x["options"] if step_of(o["text"]) in aff]
        gold = set(x["answer"]); k = len(gold); n = len(x["options"])
        if not gold or k == n: return None, False
        c = len(set(pred) & gold); w = len(set(pred) - gold)
        return max(0.0, c / k - w / (n - k)), True
    if fmt == "integer" and negq:
        try: g = int(x["answer"])
        except Exception: return None, False
        return float(len(aff) == g), True
    if fmt == "mcq_single":
        tgt = x["required_entries"][1] if len(x["required_entries"]) > 1 else None
        if tgt is None: return None, False
        dep = tgt in aff
        pick = None
        for o in x["options"]:
            t = o["text"].lower()
            if dep and re.search(r"would not have occurred|not have (occurred|taken place)|prevented", t) and re.search(r"depend|prerequisite|relied", t):
                pick = o["label"]; break
            if (not dep) and re.search(r"would have occurred|would still|still have", t) and re.search(r"separate|independent|does not depend|unrelated|different (path|line)", t):
                pick = o["label"]; break
        if pick is None: return None, False
        return float(pick == x["answer"]), True
    return None, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "recon"))
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "results", "recon", "oracle_results.json"))
    a = ap.parse_args()

    per_dom = collections.defaultdict(lambda: {"scenarios": 0, "exact1": 0, "exactS": 0,
                                               "cm1": collections.Counter(), "cmS": collections.Counter(),
                                               "mcq": collections.defaultdict(lambda: [0.0, 0, 0])})
    records, skipped = [], []
    for skp in sorted(glob.glob(os.path.join(a.data, "*", "case_*", "skeleton.json"))):
        skel = json.load(open(skp)); dom = skel["domain"]; case = skel["case_id"]
        if not any("requires" in L for ch in chains(skel) for L in ch["links"]):
            skipped.append(case); continue
        owner = build_engines(skel)
        qs = json.load(open(skp.replace("skeleton.json", "questions.json")))
        qs = qs if isinstance(qs, list) else next(v for v in qs.values() if isinstance(v, list))
        D = per_dom[dom]
        for cs in skel["counterfactual_scenarios"]:
            t = cs["trigger_entry_id"]
            if t not in owner: continue
            eng, cid = owner[t]
            gold_dep, gold_ind = set(cs["dependent_steps"]), set(cs["independent_steps"])
            A1, AS = affected_one_step(eng, t), affected_iterated(eng, t)
            D["scenarios"] += 1
            D["exact1"] += (A1 == gold_dep and not (A1 & gold_ind))
            D["exactS"] += (AS == gold_dep and not (AS & gold_ind))
            for step in gold_dep | gold_ind:
                g = step in gold_dep
                D["cm1"][(step in A1, g)] += 1
                D["cmS"][(step in AS, g)] += 1
            rec = {"case": case, "domain": dom, "chain": cid, "trigger": t,
                   "gold_dependent": sorted(gold_dep), "gold_independent": sorted(gold_ind),
                   "affected_one_step": sorted(A1), "affected_iterated": sorted(AS), "mcq": []}
            for x in qs:
                if x["type"] != "counterfactual" or t not in x.get("required_entries", []): continue
                sc, cov = score_mcq(x, skel, AS, owner)
                key = x["format"]
                D["mcq"][key][2] += 1
                if cov:
                    D["mcq"][key][0] += sc; D["mcq"][key][1] += 1
                rec["mcq"].append({"id": x["id"], "format": key, "score": sc, "covered": cov})
            records.append(rec)

    def prf(cm):
        tp, fp, fn, tn = cm[(True, True)], cm[(True, False)], cm[(False, True)], cm[(False, False)]
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 4), recall=round(R, 4),
                    f1=round(2 * P * R / (P + R), 4) if P + R else 0.0)

    summary = {}
    print(f"{'domain':<9}{'scen':>5}{'exact 1-step':>14}{'exact iter':>12}   "
          f"{'1-step P/R':>13}   {'iter P/R':>11}   per-step judgements")
    for dom, D in sorted(per_dom.items()):
        s1, sS = prf(D["cm1"]), prf(D["cmS"]); n = D["scenarios"]
        nj = sum(D["cmS"].values())
        summary[dom] = {"scenarios": n, "exact_one_step": D["exact1"], "exact_iterated": D["exactS"],
                        "one_step": s1, "iterated": sS, "n_step_judgements": nj,
                        "mcq": {k: {"mean_score": round(v[0] / v[1], 4) if v[1] else None,
                                    "covered": v[1], "total": v[2]} for k, v in D["mcq"].items()}}
        print(f"{dom:<9}{n:>5}{D['exact1']:>8}/{n:<5}{D['exactS']:>6}/{n:<5}   "
              f"{s1['precision']:.3f}/{s1['recall']:.3f}   {sS['precision']:.3f}/{sS['recall']:.3f}   n={nj}")
    print("\nMCQ/integer (Affected*, RECON scoring):")
    for dom, S in summary.items():
        for k, v in sorted(S["mcq"].items()):
            print(f"  {dom:<9}{k:<13} mean={v['mean_score']}  covered {v['covered']}/{v['total']}")
    if skipped: print(f"\nskipped (no requires/establishes in skeleton): {len(skipped)} cases: {skipped[0]} ...")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "skipped": skipped, "records": records}, open(a.out, "w"), indent=1)
    print(f"\nwritten -> {a.out}")

if __name__ == "__main__":
    main()
