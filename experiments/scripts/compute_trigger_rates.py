#!/usr/bin/env python3
"""Supersession trigger rates across the four external benchmarks, one definition.

DEFINITION (item-level, applied to the unit that is actually scored)
    An evaluation item is *supersession-triggering* iff its context contains an
    assertion p and a later assertion p' that supersedes p (same subject/key,
    incompatible value), and producing the gold answer requires p' rather than p.

Measured two ways, because they answer different questions:

    T1  benchmark-native rate  -- determined from the benchmark's own
        annotations or construction. Independent of our pipeline. This is the
        property that licenses "a gain is possible on this benchmark".

    T2  engine-observed rate   -- fraction of scored items on which our
        pipeline actually recorded >= 1 supersession / retraction event. This is
        what the verifier can act on.

A gain requires BOTH to be high. T1 high with T2 low is an extraction failure,
not a benchmark property -- reporting only T2 would mislabel it as the latter.

Usage:
    python compute_trigger_rates.py --results experiments/results \
"""
import argparse, glob, json, os, sys
from collections import Counter


def _load(p):
    with open(p) as f:
        return json.load(f)


def _load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------
def reviseqa(R):
    """T1: an edit step triggers iff the benchmark's declared edit removes at
    least one premise that is present in the maintained dependency set
    (apply_diag.n_removed_found > 0). The removed-premise list is the
    benchmark's own annotation, so T1 == T2 for this benchmark: the 'native
    updates' arm applies those annotations directly."""
    src = os.path.join(R, 'reviseqa_multimodel/gpt-4o/two_arm.json')
    recs = _load(src)['records']
    steps = trig = 0
    sc_trig = 0
    for r in recs:
        hit = False
        for s in r['per_step']:
            ad = s['hybrid_dep_only'].get('apply_diag') or {}
            steps += 1
            if ad.get('n_removed_found', 0) > 0:
                trig += 1
                hit = True
        sc_trig += hit
    # pure-native T1: the benchmark declares >=1 removed premise at all
    decl = 0
    for r in recs:
        for s in r['per_step']:
            ad = s['hybrid_dep_only'].get('apply_diag') or {}
            decl += (ad.get('n_removed_found', 0) + ad.get('n_removed_missing', 0)) > 0

    # T2: the e2e arm replaces benchmark annotations with a GPT-4o Interpreter.
    # Recall = of the steps that natively trigger, how many does the Interpreter
    # actually emit a remove op for.
    e2e = _load(os.path.join(R, 'reviseqa_multimodel/gpt-4o/e2e_replay.json'))
    byid = {r['scenario_id']: r for r in recs}
    tp = fn = fp = tn = 0
    for r in e2e['records']:
        nr = byid.get(r['scenario_id'])
        if nr is None:
            continue
        for se, sn in zip(r['per_step'], nr['per_step']):
            ad = sn['hybrid_dep_only'].get('apply_diag') or {}
            nat_t = ad.get('n_removed_found', 0) > 0
            llm_t = any(o.get('op') == 'remove' for o in se['interp']['ops'])
            if nat_t and llm_t:   tp += 1
            elif nat_t:           fn += 1
            elif llm_t:           fp += 1
            else:                 tn += 1
    recall = tp / (tp + fn) if tp + fn else None
    return {
        'benchmark': 'ReviseQA',
        'unit': 'edit step',
        'n_items': steps,
        'T1_native': decl / steps,
        'T1_source': 'benchmark edit annotations: edit declares >=1 removed premise',
        'T2_engine': recall,
        'T2_source': 'GPT-4o Interpreter recall on natively-triggering steps '
                     f'({tp}/{tp+fn}); e2e arm emits a remove op',
        'aux': {'scenario_level_T1': sc_trig / len(recs), 'n_scenarios': len(recs),
                'n_triggering_steps_in_depset': trig,
                'n_steps_declaring_removal': decl,
                'extraction_confusion': {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn},
                'over_removal_rate_on_nontriggering': fp / (fp + tn) if fp + tn else None},
        'src': src + ' + reviseqa_multimodel/gpt-4o/e2e_replay.json',
    }


def memab_cr(R):
    """T1: a fact triggers iff a later same-key fact supersedes it under the
    benchmark's stated larger-serial-wins rule (template_diag.n_superseded).
    T2: the LLM Interpreter's per-fact supersede decisions, compared against
    that template ingestion (active_agreement_jaccard)."""
    nat = _load(os.path.join(R, 'memagentbench_cr/cr_qwen2.5-7b.json'))['variants']
    per_len = {}
    for L in ['6k', '32k', '64k', '262k']:
        d = nat[f'sh_{L}']['diag']
        per_len[L] = {'n_facts': d['n_facts'], 'n_superseded': d['n_superseded'],
                      'rate': d['n_superseded'] / d['n_facts']}
    rates = [v['rate'] for v in per_len.values()]

    t2 = {}
    for L in ['6k', '32k']:
        p = os.path.join(R, f'memagentbench_cr/e2e_ingest_{L}.json')
        if os.path.exists(p):
            g = _load(p)
            t2[L] = {'jaccard': g['active_agreement_jaccard'],
                     'n_active_llm': g['n_active_llm'],
                     'n_active_template': g['n_active_template']}
    return {
        'benchmark': 'MemAB-CR',
        'unit': 'fact in stream',
        'n_items': sum(v['n_facts'] for v in per_len.values()),
        'T1_native': (min(rates), max(rates)),
        'T1_source': "benchmark's larger-serial-wins rule (template_diag)",
        'T2_engine': min(v['jaccard'] for v in t2.values()) if t2 else None,
        'T2_source': 'LLM Interpreter decides supersession fact-by-fact and '
                     'reproduces the template store; agreement (Jaccard) '
                     + '/'.join(f"{v['jaccard']:.3f}" for v in t2.values())
                     + ' on the 6K/32K subset',
        'aux': {'per_length': per_len, 'e2e_ingest': t2},
        'src': 'memagentbench_cr/cr_qwen2.5-7b.json + e2e_ingest_*.json',
    }


def fmt(x):
    if x is None:
        return 'n/a'
    if isinstance(x, tuple):
        return f'{x[0]:.1%}--{x[1]:.1%}'
    return f'{x:.1%}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='experiments/results')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows = [reviseqa(a.results), memab_cr(a.results)]

    w = 24
    print(f"{'benchmark':<{w}}{'unit':<16}{'n':>7}{'T1 native':>14}{'T2 engine':>14}")
    print('-' * (w + 16 + 7 + 28))
    for r in rows:
        print(f"{r['benchmark']:<{w}}{r['unit']:<16}{r['n_items']:>7}"
              f"{fmt(r['T1_native']):>14}{fmt(r['T2_engine']):>14}")
    print()
    for r in rows:
        print(f"* {r['benchmark']}")
        print(f"    T1 <- {r['T1_source']}")
        print(f"    T2 <- {r['T2_source']}")
        print(f"    aux: {json.dumps(r['aux'])[:400]}")
        print(f"    src: {r['src']}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rows, f, indent=1, default=str)
        print(f'\nwritten -> {a.out}')


if __name__ == '__main__':
    sys.exit(main())
