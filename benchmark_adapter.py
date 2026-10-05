#!/usr/bin/env python3
"""
Benchmark adapter for ReviseQA.

Loads the verified ReviseQA scenarios, maintains the engine state under the
benchmark's structured edits, renders the dependency-map block read by the QA
model, and scores answers in closed form (True / False / Uncertain) with the
benchmark's accuracy@k.
"""

import os, sys, json, re, time
from typing import Optional
from pipeline import EpistemicPipeline

# ============================================================
# ReviseQA (explicit_no_correction_no_reasoning)
# ------------------------------------------------------------
# Dataset source: https://github.com/ChadiHelwe/reviseqa
# Sample schema: per reviseqa_data/nl/verified/*.json :
#   {original_context: list[str], original_context_fol: list[str],
#    conclusion: str, conclusion_fol: str,
#    answer: 'True'|'False'|'Uncertain',
#    reasoning_chain: [{conclusion, facts, rules}, ...],
#    edits: [{edit_number, modification_type, edited_context_fol,
#             edited_natural_language_context, edits_made, answer,
#             conclusion, conclusion_fol, prover9_input}, ...]
#   }  # edits is always length 7.
#
# Scope of this adapter:
#  - Only the `explicit_no_correction_no_reasoning` task setting.
#  - Closed-form auto-scoring against GT True/False/Uncertain (no judge).
#  - The verifier arm injects a dependency-map-only block.
#  - Engine state is maintained benchmark-natively: each fact/rule is
#    registered as an engine observation and edits directly mutate the
#    engine state. The end-to-end arm (run_reviseqa_interp_full.py) replaces
#    the structured edits with an LLM Interpreter.
# ============================================================


def _reviseqa_fol_key(fol_str: str) -> str:
    """Canonical key for a FOL string. Normalizes whitespace + preserves
    unicode logic operators; meant as a deterministic primary key for
    mapping between (fact/rule in edit.edits_made) and (engine obs id)."""
    return re.sub(r"\s+", " ", fol_str).strip()


def load_reviseqa(data_dir: str,
                  skip_truncated: bool = True,
                  max_scenarios: Optional[int] = None
                  ) -> list[dict]:
    """Load verified ReviseQA scenarios.

    data_dir: e.g. 'reviseqa_data/nl/verified'.  Reads ex_*.json (skipping
    *_truncated.json by default), returns the raw scenario schema verbatim
    (does NOT flatten into our speaker/text turns+queries format — ReviseQA
    is an edit-sequence benchmark, not a dialogue one). Each scenario gets
    id = filename stem (e.g., "ex_712").
    """
    out: list[dict] = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"ReviseQA data_dir not found: {data_dir}")
    fnames = sorted(os.listdir(data_dir))
    for fname in fnames:
        if not fname.endswith(".json"):
            continue
        if skip_truncated and fname.endswith("_truncated.json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            d = json.load(f)
        # Required keys guard (fail-fast if schema drifts).
        for k in ("original_context", "conclusion", "answer", "edits"):
            if k not in d:
                print(f"[warn] skip {fname}: missing key {k}")
                break
        else:
            d["id"] = fname[:-5]  # strip .json
            out.append(d)
            if max_scenarios is not None and len(out) >= max_scenarios:
                break
    return out


# ---- "benchmark-native structured update" state ----
# A lightweight side-state we maintain in parallel with EpistemicEngine:
#   fol_to_id: FOL canonical key -> observation id (f_<n> for fact, r_<n>
#              for rule); stable per scenario.
#   active:    set of currently-registered observation ids (shrinks on
#              removed_facts/rules, grows on added_facts/rules).
#   dep_h1:    set of observation ids declared as Dep(h1).
# We do NOT store in engine dictionaries beyond what symbolic_engine.py
# already supports.


def _reviseqa_register_premise(engine, fol_to_id: dict, active: set,
                               counter: dict, fol_str: str, nl_text: str,
                               kind: str) -> str:
    """Register one premise as an engine observation, unless already present.
    kind is 'fact' (prefix 'f') or 'rule' (prefix 'r').
    Returns the observation id used."""
    key = _reviseqa_fol_key(fol_str)
    if key in fol_to_id:
        # Duplicate FOL string (can happen if the same fact re-appears across
        # edits); re-activate without double-registering.
        oid = fol_to_id[key]
        active.add(oid)
        return oid
    prefix = "f" if kind == "fact" else "r"
    counter[prefix] = counter.get(prefix, 0) + 1
    oid = f"{prefix}_{counter[prefix]}"
    engine.observe(oid, nl_text, turn="reviseqa", speaker="dataset")
    fol_to_id[key] = oid
    active.add(oid)
    return oid


def _reviseqa_retract_premise(engine, fol_to_id: dict, active: set,
                              dep_h1: set, fol_str: str) -> tuple[bool, str]:
    """Actually remove a premise from engine state. symbolic_engine's
    retract_assumption() is non-destructive (only reports Affected), so we
    mutate the engine dicts directly here (benchmark-local behaviour; does
    NOT alter symbolic_engine.py).  Returns (found, oid)."""
    key = _reviseqa_fol_key(fol_str)
    if key not in fol_to_id:
        return False, ""
    oid = fol_to_id[key]
    # Drop from engine observation table if present
    if oid in getattr(engine, "observations", {}):
        del engine.observations[oid]
    # Drop from awareness
    if oid in getattr(engine, "awareness", set()):
        engine.awareness.discard(oid)
    # Drop from every dependency set
    for hyp_id, deps in list(getattr(engine, "dependencies", {}).items()):
        if oid in deps:
            deps.discard(oid)
    active.discard(oid)
    dep_h1.discard(oid)
    return True, oid


def _reviseqa_build_initial_dep(reasoning_chain: list, fol_to_id: dict) -> set:
    """Build Dep(h1) from the dataset's reasoning_chain. Walks every step,
    unions all {facts, rules} FOL strings used, looks them up in the
    already-populated fol_to_id, and returns the set of matched oids.
    Unmatched FOL strings (rare — reasoning_chain usually references the
    same strings as original_context) are reported separately."""
    wanted_fols: set = set()
    for step in reasoning_chain or []:
        for f in step.get("facts") or []:
            if isinstance(f, dict) and "fol" in f:
                wanted_fols.add(_reviseqa_fol_key(f["fol"]))
        for r in step.get("rules") or []:
            if isinstance(r, dict) and "fol" in r:
                wanted_fols.add(_reviseqa_fol_key(r["fol"]))
    dep_ids = {fol_to_id[k] for k in wanted_fols if k in fol_to_id}
    return dep_ids


def _reviseqa_engine_block_dep_only(engine, h1_id: str) -> str:
    """Dep-map-only block (per Phase 1 spec, no state summary).  Resolves
    each oid in Dep(h1) to its NL text from engine.observations when
    available; otherwise falls back to the bare id."""
    dep_ids = sorted(engine.dependencies.get(h1_id, set()))
    h1 = engine.hypotheses.get(h1_id)
    h1_content = h1.content if h1 else "<hypothesis>"
    lines = [
        "## Dependency map (ECM)",
        f"Hypothesis h1: \"{h1_content}\"",
        "Currently-tracked supporting premises (Dep(h1)):",
    ]
    if not dep_ids:
        lines.append("  (none)")
    else:
        for oid in dep_ids:
            nl = engine.observations[oid].content if oid in engine.observations else f"<retracted {oid}>"
            lines.append(f"  - {oid}: {nl}")
    return "\n".join(lines) + "\n"


def _reviseqa_apply_edit(engine, fol_to_id: dict, active: set, dep_h1: set,
                        counter: dict, edit: dict) -> dict:
    """Apply one edit's add/remove to engine state.  Updates Dep(h1) so
    that it tracks the currently-available subset of originally-depended
    premises PLUS any newly-added premises (conservative: new additions
    may participate in the conclusion's support)."""
    delta = edit.get("edits_made", {}) or {}
    diag = {"edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "n_removed_found": 0, "n_removed_missing": 0,
            "n_added_facts": 0, "n_added_rules": 0}

    # Removals first (so re-added FOLs, if any, can re-register fresh)
    for f in delta.get("removed_facts") or []:
        found, _ = _reviseqa_retract_premise(engine, fol_to_id, active, dep_h1, f["fol"])
        diag["n_removed_found"] += int(found)
        diag["n_removed_missing"] += int(not found)
    for r in delta.get("removed_rules") or []:
        found, _ = _reviseqa_retract_premise(engine, fol_to_id, active, dep_h1, r["fol"])
        diag["n_removed_found"] += int(found)
        diag["n_removed_missing"] += int(not found)

    # Additions
    for f in delta.get("added_facts") or []:
        oid = _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                          f["fol"], f["nl"], "fact")
        dep_h1.add(oid)
        diag["n_added_facts"] += 1
    for r in delta.get("added_rules") or []:
        oid = _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                          r["fol"], r["nl"], "rule")
        dep_h1.add(oid)
        diag["n_added_rules"] += 1

    # Re-write engine.dependencies[h1] so the dep-map-only block sees the
    # current active Dep(h1). We keep only ids still in `active`.
    current = {oid for oid in dep_h1 if oid in active}
    engine.dependencies["h1"] = current

    diag["dep_h1_size_after"] = len(current)
    diag["n_active_after"] = len(active)
    return diag


def _reviseqa_dep_map_stats_from_engine(engine, h1_id: str) -> dict:
    """Diagnostic snapshot of engine dep map + counts (LoCoMo/MT-Bench-101
    parity — h_auto/h_to_h/degenerate flags don't really apply here since
    we're seeding the engine ourselves, but we report them for consistency).
    """
    dep = {k: list(v) for k, v in engine.dependencies.items()}
    total = len(dep)
    h_auto = sum(1 for k in dep if isinstance(k, str) and k.startswith("h_auto_"))
    h_to_h = sum(1 for deps in dep.values()
                 if any(isinstance(d, str) and d.startswith("h") for d in deps))
    degenerate = sum(1 for k, deps in dep.items()
                     if isinstance(k, str) and k.startswith("h_auto_")
                     and len(deps) == 1
                     and isinstance(deps[0], str) and deps[0].startswith("o"))
    return {"n_entries": total, "h_auto": h_auto, "h_to_h": h_to_h,
            "degenerate": degenerate,
            "dep_h1_size": len(dep.get(h1_id, []))}


def _reviseqa_format_explicit_context(edit: dict) -> str:
    """Mirror ReviseQA evaluation.py explicit-context formatting (edit
    delta, not full context).  Used for both baseline and hybrid prompts."""
    delta = edit.get("edits_made", {}) or {}
    parts: list[str] = []
    if delta.get("removed_facts"):
        parts.append("Removed facts:\n"
                     + "\n".join(f"- {f['nl']}" for f in delta["removed_facts"]))
    if delta.get("removed_rules"):
        parts.append("Removed rules:\n"
                     + "\n".join(f"- {r['nl']}" for r in delta["removed_rules"]))
    if delta.get("added_rules"):
        parts.append("Added rules:\n"
                     + "\n".join(f"- {r['nl']}" for r in delta["added_rules"]))
    if delta.get("added_facts"):
        parts.append("Added facts:\n"
                     + "\n".join(f"- {f['nl']}" for f in delta["added_facts"]))
    if not parts:
        parts = ["(no explicit edits)"]
    return "\n\n".join(parts)


REVISEQA_SYSTEM = (
    "When you reply, output *only* a JSON object with exactly "
    "two fields:\n"
    "  - reasoning  (a string; may be empty)\n"
    "  - answer     (one of 'True','False','Uncertain')\n"
    "Do not wrap it in markdown, do not say anything else."
)

REVISEQA_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Options:\nA) True\nB) False\nC) Uncertain\n\n"
)


def _reviseqa_parse_answer(raw: Optional[str]) -> Optional[str]:
    """Extract 'True'/'False'/'Uncertain' from the model's response.
    Tries JSON parse first, then regex fallback."""
    if raw is None:
        return None
    s = raw.strip()
    # Option letters map to the prompt's fixed option list (A) True B) False
    # C) Uncertain). Reasoning-tuned models (DeepSeek-R1 distills) answer
    # with the letter; the models in the paper never did, so
    # this is a pure superset of the original parser.
    LETTER = {"A": "True", "B": "False", "C": "Uncertain"}

    def _norm_ans(a):
        a = a.strip()
        if a.upper() in LETTER:
            return LETTER[a.upper()]
        m = re.match(r"^([ABC])\)\s*(True|False|Uncertain)$", a, flags=re.I)
        if m:
            return m.group(2).capitalize()
        return a.capitalize()

    # JSON parse first (strips possible ```json fences)
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
    try:
        d = json.loads(s2)
        a = d.get("answer")
        if isinstance(a, str) and _norm_ans(a) in {"True", "False", "Uncertain"}:
            return _norm_ans(a)
    except Exception:
        pass
    # Regex fallback for unstructured output (also finds JSON embedded in
    # prose or inside a fenced block that is not at the string start)
    m = re.search(r'"answer"\s*:\s*"([^"]{1,24})"', s)
    if m:
        cand = _norm_ans(m.group(1))
        if cand in {"True", "False", "Uncertain"}:
            return cand
    # Last-ditch: bare token match
    for tok in ["Uncertain", "True", "False"]:
        if re.search(rf'\b{tok}\b', s):
            return tok
    return None


def _reviseqa_chat_call(messages: list, api_key: str,
                        url: str,
                        model: str,
                        temperature: float = 0,
                        max_tokens: Optional[int] = None,
                        max_retries: int = 3) -> Optional[str]:
    """Multi-turn chat completion for ReviseQA cumulative conversation.
    Separate from pipeline.call_llm (which is 2-message only) because the
    ReviseQA protocol requires a growing dialogue history per chain.

    max_tokens defaults to 800 (the value used in all reported runs);
    REVISEQA_MAX_TOKENS overrides it for QA models that spend tokens on
    hidden reasoning before the JSON answer (e.g. DeepSeek-R1 distills
    served with vLLM's reasoning parser)."""
    if max_tokens is None:
        max_tokens = int(os.environ.get("REVISEQA_MAX_TOKENS", "800"))
    try:
        import requests
    except ImportError:
        return None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json={"model": model, "temperature": temperature,
                      "max_tokens": max_tokens, "messages": messages},
                timeout=120)
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            print(f"  [reviseqa] API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', data)}")
        except Exception as e:
            print(f"  [reviseqa] Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


def _reviseqa_init_scenario_state(scenario: dict) -> tuple:
    """Fresh EpistemicPipeline; seed engine with original_context facts and
    any rules mined from reasoning_chain; register h1 as the conclusion
    hypothesis with Dep(h1) from reasoning_chain. No LLM calls.

    Returns (pipe, fol_to_id, active, dep_h1, counter, init_diag).
    """
    pipe = EpistemicPipeline(api_key="unused-for-structured-only",
                             verbose=False, max_history_turns=0)
    engine = pipe.engine
    fol_to_id: dict = {}
    active: set = set()
    dep_h1: set = set()
    counter: dict = {"f": 0, "r": 0}

    # 1. original_context facts: original_context_fol lines (positional
    #    match with original_context NL lines)
    ctx_nl = scenario.get("original_context", []) or []
    ctx_fol = scenario.get("original_context_fol", []) or []
    if len(ctx_nl) != len(ctx_fol):
        # partial match: use min
        pass
    n_ctx = min(len(ctx_nl), len(ctx_fol))
    for i in range(n_ctx):
        _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                    ctx_fol[i], ctx_nl[i], "fact")

    # 2. rules found in reasoning_chain that are NOT already in fol_to_id
    #    (original_context often contains facts only; rules are mined
    #    from reasoning_chain)
    for step in scenario.get("reasoning_chain") or []:
        for r in step.get("rules") or []:
            if isinstance(r, dict) and "fol" in r and "text" in r:
                _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                            r["fol"], r["text"], "rule")
        for f in step.get("facts") or []:
            if isinstance(f, dict) and "fol" in f and "text" in f:
                if _reviseqa_fol_key(f["fol"]) not in fol_to_id:
                    _reviseqa_register_premise(engine, fol_to_id, active, counter,
                                                f["fol"], f["text"], "fact")

    # 3. hypothesize h1 with dep from reasoning_chain
    init_dep = _reviseqa_build_initial_dep(
        scenario.get("reasoning_chain") or [], fol_to_id)
    dep_h1 = set(init_dep)
    engine.hypothesize(
        hyp_id="h1",
        content=scenario.get("conclusion", ""),
        turn="reviseqa",
        speaker="dataset",
        explains=[],
        depends_on=list(dep_h1),
    )

    init_diag = {
        "n_ctx_facts_registered": n_ctx,
        "n_total_registered": counter["f"] + counter["r"],
        "n_facts_registered": counter["f"],
        "n_rules_registered": counter["r"],
        "dep_h1_size_initial": len(dep_h1),
        "dep_h1_coverage": (
            round(len(dep_h1) / len(active), 3) if active else 0.0
        ),
    }
    return pipe, fol_to_id, active, dep_h1, counter, init_diag


def evaluate_reviseqa_scenario(
        scenario: dict,
        tested_api_key: str,
        base_url: str,
        model: str,
        include_reasoning: bool,
        include_correction: bool,
        run_baseline: bool,
        run_hybrid: bool,
        ) -> dict:
    """Run one scenario chain (1 demo + 7 edits) under the
    explicit_no_correction_no_reasoning setting.

    - `include_reasoning`: controls whether the 0-shot demo's assistant
      message includes a `reasoning` text (set to False for the no_reasoning
      variant).
    - `include_correction`: if True, a CORRECTION line is appended after
      a wrong answer (set to False for no_correction).
    """
    scenario_id = scenario["id"]
    edits = scenario.get("edits", [])
    orig_ctx_text = "\n".join(scenario.get("original_context", []) or [])
    conclusion = scenario.get("conclusion", "")
    orig_answer = scenario.get("answer", "Uncertain")
    base_question = f"Does the context entail the conclusion '{conclusion}'?"

    demo_reasoning = ""  # no_reasoning: demo carries an empty reasoning
    demo_assistant = json.dumps(
        {"reasoning": demo_reasoning if include_reasoning else "",
         "answer": orig_answer})

    baseline_messages = [
        {"role": "system", "content": REVISEQA_SYSTEM},
        {"role": "user", "content": REVISEQA_PROMPT_TEMPLATE.format(
            context=orig_ctx_text, question=base_question)},
        {"role": "assistant", "content": demo_assistant},
    ]
    hybrid_messages = [
        {"role": "system", "content": REVISEQA_SYSTEM},
        {"role": "user", "content": REVISEQA_PROMPT_TEMPLATE.format(
            context=orig_ctx_text, question=base_question)},
        {"role": "assistant", "content": demo_assistant},
    ]

    # Engine state for hybrid.
    if run_hybrid:
        pipe, fol_to_id, active, dep_h1, counter, init_diag = \
            _reviseqa_init_scenario_state(scenario)
        engine = pipe.engine
    else:
        pipe = engine = fol_to_id = active = dep_h1 = counter = None
        init_diag = {}

    per_step: list[dict] = []
    baseline_trace: list[bool] = []
    hybrid_trace: list[bool] = []

    for step_idx, edit in enumerate(edits, start=1):
        gt = edit.get("answer", "")
        step_conclusion = edit.get("conclusion", conclusion)
        step_question = f"Does the context entail the conclusion '{step_conclusion}'?"
        delta_ctx = _reviseqa_format_explicit_context(edit)
        user_msg_plain = REVISEQA_PROMPT_TEMPLATE.format(
            context=delta_ctx, question=step_question)

        # ---- baseline ----
        base_pred = None
        base_raw = None
        if run_baseline:
            baseline_messages.append({"role": "user", "content": user_msg_plain})
            base_raw = _reviseqa_chat_call(baseline_messages, tested_api_key,
                                           base_url, model)
            baseline_messages.append({"role": "assistant",
                                       "content": base_raw or ""})
            base_pred = _reviseqa_parse_answer(base_raw)
            base_correct = (base_pred == gt)
            baseline_trace.append(bool(base_correct))
            if include_correction and not base_correct:
                baseline_messages.append({
                    "role": "user",
                    "content": f"You made a mistake, the correct answer was: "
                               f"{gt}. Now answer the next problem."})

        # ---- hybrid (apply edit into engine BEFORE generating prompt) ----
        hyb_pred = None
        hyb_raw = None
        apply_diag = None
        dep_stats = None
        if run_hybrid:
            apply_diag = _reviseqa_apply_edit(
                engine, fol_to_id, active, dep_h1, counter, edit)
            dep_stats = _reviseqa_dep_map_stats_from_engine(engine, "h1")
            engine_block = _reviseqa_engine_block_dep_only(engine, "h1")
            user_msg_hybrid = (
                engine_block + "\n" + REVISEQA_PROMPT_TEMPLATE.format(
                    context=delta_ctx, question=step_question))
            hybrid_messages.append({"role": "user", "content": user_msg_hybrid})
            hyb_raw = _reviseqa_chat_call(hybrid_messages, tested_api_key,
                                          base_url, model)
            hybrid_messages.append({"role": "assistant",
                                     "content": hyb_raw or ""})
            hyb_pred = _reviseqa_parse_answer(hyb_raw)
            hyb_correct = (hyb_pred == gt)
            hybrid_trace.append(bool(hyb_correct))
            if include_correction and not hyb_correct:
                hybrid_messages.append({
                    "role": "user",
                    "content": f"You made a mistake, the correct answer was: "
                               f"{gt}. Now answer the next problem."})

        per_step.append({
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "edit_number": edit.get("edit_number"),
            "modification_type": edit.get("modification_type"),
            "gt": gt,
            "baseline": None if not run_baseline else {
                "raw": base_raw, "parsed": base_pred,
                "correct": base_pred == gt,
            },
            "hybrid_dep_only": None if not run_hybrid else {
                "raw": hyb_raw, "parsed": hyb_pred,
                "correct": hyb_pred == gt,
                "apply_diag": apply_diag,
                "dep_stats": dep_stats,
            },
        })

    # LCATA traces are per-step bool lists (length = n edits = 7).
    def _lcata_ks(trace):
        return {
            "k2": int(all(trace[:2])) if len(trace) >= 2 else 0,
            "k4": int(all(trace[:4])) if len(trace) >= 4 else 0,
            "k7": int(all(trace[:7])) if len(trace) >= 7 else 0,
        }

    final_stats_hyb = _reviseqa_dep_map_stats_from_engine(engine, "h1") if engine else {}
    return {
        "scenario_id": scenario_id,
        "task_setting": "explicit_no_correction_no_reasoning",
        "n_edits": len(edits),
        "modification_types": [e.get("modification_type") for e in edits],
        "init_diag": init_diag,
        "baseline_trace": baseline_trace,
        "hybrid_dep_only_trace": hybrid_trace,
        "baseline_lcata": _lcata_ks(baseline_trace) if run_baseline else {},
        "hybrid_dep_only_lcata": _lcata_ks(hybrid_trace) if run_hybrid else {},
        "final_dep_stats_hybrid": final_stats_hyb,
        "per_step": per_step,
    }


def _reviseqa_aggregate_lcata(records: list[dict], key: str) -> dict:
    """LCATA@k = fraction of chains with all-correct in first k edits."""
    totals = {"k2": [], "k4": [], "k7": []}
    for r in records:
        lc = r.get(key, {}) or {}
        for k in totals:
            if k in lc:
                totals[k].append(lc[k])
    return {k: (round(sum(v) / len(v), 4) if v else None,
               len(v)) for k, v in totals.items()}


def _reviseqa_preflight_check(records: list[dict]) -> tuple[bool, list[str]]:
    """Gate preflight to full smoke. Conditions:
      - Every scenario has dep_h1_size_initial >= 2
      - Across all edits, the removal-miss rate is < 25%
      - Final Dep(h1) stays non-empty on at least half the scenarios
    Returns (passed, list_of_issues)."""
    issues = []
    small_init = [r["scenario_id"] for r in records
                  if r.get("init_diag", {}).get("dep_h1_size_initial", 0) < 2]
    if small_init:
        issues.append(f"Dep(h1) initial size < 2 on: {small_init}")

    tot_found = tot_missing = 0
    for r in records:
        for step in r["per_step"]:
            d = (step.get("hybrid_dep_only") or {}).get("apply_diag") or {}
            tot_found += d.get("n_removed_found", 0)
            tot_missing += d.get("n_removed_missing", 0)
    miss_rate = (tot_missing / (tot_found + tot_missing)
                 if (tot_found + tot_missing) else 0.0)
    if miss_rate >= 0.25:
        issues.append(f"Removal miss rate {miss_rate:.2%} >= 25% "
                      f"(found={tot_found} missing={tot_missing})")

    empty_final = sum(
        1 for r in records
        if r.get("final_dep_stats_hybrid", {}).get("dep_h1_size", 0) == 0)
    if empty_final > len(records) / 2:
        issues.append(f"Final Dep(h1) empty on {empty_final}/{len(records)} "
                      "scenarios (engine became semantically vacuous).")

    return (len(issues) == 0, issues)


def _reviseqa_atomic_save(out_path: str, data: dict) -> None:
    """Write JSON via temp+rename for crash-safety during long runs.
    If the process dies mid-write, the original file is untouched.
    """
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, out_path)


def _reviseqa_load_checkpoint(out_path: str) -> Optional[dict]:
    """Load an in-progress / completed ReviseQA output for resume.
    Returns None if file does not exist. Raises on malformed JSON."""
    if not os.path.isfile(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)
