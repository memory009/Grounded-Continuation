#!/usr/bin/env python3
"""
End-to-End Epistemic Pipeline
Grounded Continuation runtime verifier (ICLR 2027 submission)

This module provides the full LLM-to-Engine pipeline:
  1. Feed any conversation turn-by-turn
  2. LLM classifies each turn into operations + extracts propositions
  3. Symbolic engine updates automatically from LLM output
  4. Query the engine state at any point (including counterfactuals)

This bridges the gap between the classification experiments
(run_experiments.py) and a deployable system. It is the
"LLM Interpreter" component from Figure 1.

Usage:
    # As a library
    from pipeline import EpistemicPipeline
    pipe = EpistemicPipeline(api_key="sk-ant-...")
    for turn in conversation:
        result = pipe.process_turn(turn["speaker"], turn["text"])
        print(result)
    print(pipe.engine.get_state_summary())
    print(pipe.engine.retract_assumption("o9"))

    # As a script (runs Phase 2 end-to-end)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python pipeline.py
    python pipeline.py --conversation my_conversation.json
    python pipeline.py --query "What depends on h2?"
"""

import os, sys, json, time, re, argparse
from typing import Optional
from symbolic_engine import EpistemicEngine

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
BACKEND = "anthropic"

# Per-model token usage counter. Populated by _call_llm_openai / _call_llm_anthropic
# whenever the API returns usage info. Callers can read this between phases to
# attribute cost across e.g. cheap-classifier + expensive-QA setups.
# Format: {model_name: {"prompt_tokens": int, "completion_tokens": int, "calls": int}}
USAGE_LOG: dict = {}


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int):
    rec = USAGE_LOG.setdefault(
        model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    )
    rec["prompt_tokens"] += prompt_tokens
    rec["completion_tokens"] += completion_tokens
    rec["calls"] += 1


# ============================================================
# Single-Turn Classification Prompt
# ============================================================

CLASSIFY_SYSTEM = """You are an epistemic analyst classifying a single utterance in a multi-agent conversation.

## Operations (assign one or more)

- **Observe**: Report factual data without explanation.
- **Hypothesize**: Propose a NEW causal/explanatory chain not previously stated.
- **Support**: Provide evidence STRENGTHENING an existing hypothesis/position.
- **Undermine**: Provide evidence WEAKENING an existing hypothesis/position.
- **Revise**: ABANDON a hypothesis/position or REVERSE a causal understanding.
- **Expand-Awareness**: Introduce a concept PREVIOUSLY UNCONCEIVED by anyone.
- **Resolve**: Elevate a hypothesis to accepted conclusion / make a decision.
- **Question**: Ask for information or flag something anomalous.

## Key distinctions
- Undermine WEAKENS; Revise ABANDONS or REVERSES.
- Expand-Awareness adds a NEW DIMENSION nobody was thinking about, not just new data.
- A turn can have MULTIPLE operations (e.g., Observe + Question, Hypothesize + Revise).

## Your task
Given the conversation so far, the current model state, and the new utterance,
output a JSON object with the classification and extracted propositions.

You MUST respond with ONLY a JSON object (no markdown, no explanation):
{
  "operations": ["Op1", "Op2"],
  "observations": [
    {"id": "o_auto_N", "content": "description of observed fact"}
  ],
  "hypotheses": [
    {"id": "h_auto_N", "content": "causal chain description",
     "explains": ["o1", "o2"], "depends_on": ["o3"]}
  ],
  "supports": [{"target": "h1", "evidence": "why it's supported"}],
  "undermines": [{"target": "h2", "evidence": "why it's weakened"}],
  "revisions": [{"target": "h2", "reason": "why it's abandoned"}],
  "awareness_expansions": [{"id": "prop_name", "content": "new concept"}],
  "resolutions": [{"target": "h4"}],
  "questions": [{"content": "the question being asked"}]
}

Rules:
- observation/hypothesis IDs: use o1, o2... for observations, h1, h2... for hypotheses.
  If the model state already has oN or hN, use the NEXT available number.
- "explains": which existing observations does this hypothesis explain?
- "depends_on": which observations/hypotheses does this depend on?
  This is crucial for dependency tracking.
- "target" in supports/undermines/revisions: the ID of the hypothesis being affected.
- Include ONLY the arrays that are relevant. Empty arrays can be omitted.
"""


# Ablated classification prompt (ablation, --ablate-layers
# del_awareness): the DEL plausibility layer (Support = soft plausibility
# upgrade) and the awareness layer (Expand-Awareness) are removed from the
# extraction interface. What remains is the argument/dependency skeleton:
# hypothesis nodes with a status lifecycle (Hypothesize / Undermine /
# Revise / Resolve), observations, dependency links (depends_on/explains),
# and attack edges (Undermine). Question is retained (it touches neither
# layer). The JSON schema drops the "supports" and "awareness_expansions"
# arrays accordingly.
CLASSIFY_SYSTEM_ABLATED = """You are an epistemic analyst classifying a single utterance in a multi-agent conversation.

## Operations (assign one or more)

- **Observe**: Report factual data without explanation.
- **Hypothesize**: Propose a NEW causal/explanatory chain not previously stated.
- **Undermine**: Provide evidence WEAKENING an existing hypothesis/position.
- **Revise**: ABANDON a hypothesis/position or REVERSE a causal understanding.
- **Resolve**: Elevate a hypothesis to accepted conclusion / make a decision.
- **Question**: Ask for information or flag something anomalous.

## Key distinctions
- Undermine WEAKENS; Revise ABANDONS or REVERSES.
- A turn can have MULTIPLE operations (e.g., Observe + Question, Hypothesize + Revise).

## Your task
Given the conversation so far, the current model state, and the new utterance,
output a JSON object with the classification and extracted propositions.

You MUST respond with ONLY a JSON object (no markdown, no explanation):
{
  "operations": ["Op1", "Op2"],
  "observations": [
    {"id": "o_auto_N", "content": "description of observed fact"}
  ],
  "hypotheses": [
    {"id": "h_auto_N", "content": "causal chain description",
     "explains": ["o1", "o2"], "depends_on": ["o3"]}
  ],
  "undermines": [{"target": "h2", "evidence": "why it's weakened"}],
  "revisions": [{"target": "h2", "reason": "why it's abandoned"}],
  "resolutions": [{"target": "h4"}],
  "questions": [{"content": "the question being asked"}]
}

Rules:
- observation/hypothesis IDs: use o1, o2... for observations, h1, h2... for hypotheses.
  If the model state already has oN or hN, use the NEXT available number.
- "explains": which existing observations does this hypothesis explain?
- "depends_on": which observations/hypotheses does this depend on?
  This is crucial for dependency tracking.
- "target" in undermines/revisions: the ID of the hypothesis being affected.
- Include ONLY the arrays that are relevant. Empty arrays can be omitted.
"""


def build_classify_prompt(conversation_so_far: list[dict],
                          engine: EpistemicEngine,
                          new_speaker: str,
                          new_text: str) -> str:
    """Build the user prompt for single-turn classification."""
    # Conversation context
    conv_lines = []
    for turn in conversation_so_far:
        conv_lines.append(f"{turn['speaker']}: {turn['text']}")
    conv_text = "\n\n".join(conv_lines) if conv_lines else "(conversation start)"

    # Engine state
    state = engine.get_state_summary()

    # New utterance
    prompt = f"""## Conversation so far
{conv_text}

## Current model state
{state}

## New utterance to classify
**{new_speaker}**: {new_text}

Classify this utterance. Respond with ONLY the JSON object."""

    return prompt


# ============================================================
# LLM API Caller
# ============================================================

def call_llm(system: str, user: str, api_key: str,
             temperature: float = 0, max_retries: int = 3,
             seed: Optional[int] = None) -> Optional[str]:
    """Dispatch to Anthropic or OpenAI-compatible endpoint based on BACKEND.

    `seed` (OpenAI-compatible endpoints only) is passed through to the API's
    sampling-seed parameter when not None; Anthropic backend ignores it.
    """
    if BACKEND == "openai":
        return _call_llm_openai(system, user, api_key, temperature, max_retries,
                                seed=seed)
    return _call_llm_anthropic(system, user, api_key, temperature, max_retries)


def _call_llm_anthropic(system: str, user: str, api_key: str,
                        temperature: float, max_retries: int) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                json={"model": MODEL, "max_tokens": 1500,
                      "temperature": temperature,
                      "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=60)
            data = resp.json()
            if "content" in data and data["content"]:
                return data["content"][0]["text"]
            print(f"  API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', 'unknown')}")
        except Exception as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


def _call_llm_openai(system: str, user: str, api_key: str,
                     temperature: float, max_retries: int,
                     seed: Optional[int] = None) -> Optional[str]:
    """OpenAI-compatible chat/completions (vLLM, Ollama, OpenAI)."""
    payload = {"model": MODEL, "max_tokens": 1500,
               "temperature": temperature,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if seed is not None:
        payload["seed"] = seed
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=120)
            data = resp.json()
            if "choices" in data and data["choices"]:
                # Record usage by current global MODEL (callers can swap MODEL
                # between phases to attribute per-phase cost).
                usage = data.get("usage") or {}
                _record_usage(
                    MODEL,
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                )
                return data["choices"][0]["message"]["content"]
            print(f"  API error (attempt {attempt+1}): "
                  f"{data.get('error', {}).get('message', data)}")
        except Exception as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


# ============================================================
# Response Parser
# ============================================================

def parse_classification(text: str) -> Optional[dict]:
    """Parse the LLM's JSON classification response."""
    if not text:
        return None
    # Strip markdown fences
    s = text.strip()
    s = re.sub(r'^```json\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    s = s.strip()

    # Find JSON object
    m = re.search(r'\{[\s\S]+\}', s)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        return obj
    except json.JSONDecodeError:
        return None


# ============================================================
# Engine Updater — applies parsed classification to engine
# ============================================================

def apply_classification(engine: EpistemicEngine, classification: dict,
                         turn_id: str, speaker: str,
                         ablate_del_awareness: bool = False) -> list[str]:
    """
    Apply a parsed classification to the symbolic engine.
    Returns a log of operations applied.

    When ablate_del_awareness is True, "supports" (DEL plausibility layer)
    and "awareness_expansions" (awareness layer) are ignored even if the
    LLM emits them; the engine-level flag makes them no-ops anyway
    (belt-and-suspenders).
    """
    log = []
    ops = classification.get("operations", [])

    # Observations
    for obs in classification.get("observations", []):
        obs_id = obs.get("id", f"o_auto_{len(engine.observations)+1}")
        content = obs.get("content", "")
        if content:
            engine.observe(obs_id, content, turn_id, speaker)
            log.append(f"  Observe: {obs_id} = {content}")

    # Hypotheses
    for hyp in classification.get("hypotheses", []):
        hyp_id = hyp.get("id", f"h_auto_{len(engine.hypotheses)+1}")
        content = hyp.get("content", "")
        explains = hyp.get("explains", [])
        depends_on = hyp.get("depends_on", [])
        if content:
            engine.hypothesize(hyp_id, content, turn_id, speaker,
                               explains=explains, depends_on=depends_on)
            log.append(f"  Hypothesize: {hyp_id} = {content}")
            if depends_on:
                log.append(f"    Dep({hyp_id}) = {{{', '.join(depends_on)}}}")

    # Supports (DEL plausibility layer — skipped under ablation)
    for sup in ([] if ablate_del_awareness else classification.get("supports", [])):
        target = sup.get("target", "")
        evidence = sup.get("evidence", "")
        if target:
            engine.support(target, evidence, turn_id, speaker)
            log.append(f"  Support: {target} ({evidence[:60]})")

    # Undermines
    for und in classification.get("undermines", []):
        target = und.get("target", "")
        evidence = und.get("evidence", "")
        if target:
            engine.undermine(target, evidence, turn_id, speaker)
            log.append(f"  Undermine: {target} ({evidence[:60]})")

    # Revisions
    for rev in classification.get("revisions", []):
        target = rev.get("target", "")
        reason = rev.get("reason", "")
        if target:
            engine.revise(target, reason, turn_id, speaker)
            log.append(f"  Revise: {target} ({reason[:60]})")

    # Awareness expansions (awareness layer — skipped under ablation)
    for aw in ([] if ablate_del_awareness else classification.get("awareness_expansions", [])):
        aw_id = aw.get("id", f"aw_auto_{len(engine.awareness)+1}")
        content = aw.get("content", "")
        if content:
            engine.expand_awareness(aw_id, content, turn_id, speaker)
            log.append(f"  Expand-Awareness: {aw_id} = {content}")

    # Resolutions
    for res in classification.get("resolutions", []):
        target = res.get("target", "")
        if target:
            engine.resolve(target, turn_id, speaker)
            log.append(f"  Resolve: {target}")

    # Questions
    for q in classification.get("questions", []):
        content = q.get("content", "")
        if content:
            engine.question(content, turn_id, speaker)
            log.append(f"  Question: {content[:60]}")

    return log


# ============================================================
# Pipeline — ties everything together
# ============================================================

class EpistemicPipeline:
    """
    End-to-end pipeline: conversation → LLM classification →
    symbolic engine → queryable epistemic model.
    """

    def __init__(self, api_key: str, verbose: bool = True,
                 max_history_turns: Optional[int] = None,
                 ablate_del_awareness: bool = False,
                 temperature: float = 0,
                 llm_seed: Optional[int] = None):
        self.api_key = api_key
        self.temperature = temperature
        self.llm_seed = llm_seed
        self.ablate_del_awareness = ablate_del_awareness
        self.engine = EpistemicEngine(ablate_del_awareness=ablate_del_awareness)
        self.classify_system = (CLASSIFY_SYSTEM_ABLATED if ablate_del_awareness
                                else CLASSIFY_SYSTEM)
        self.conversation_history: list[dict] = []
        self.turn_count = 0
        self.verbose = verbose
        self.classifications: list[dict] = []
        self.max_history_turns = max_history_turns

    def process_turn(self, speaker: str, text: str) -> dict:
        """
        Process one conversation turn end-to-end:
        1. Send to LLM for classification
        2. Parse the response
        3. Apply to symbolic engine
        4. Return the classification + engine state
        """
        self.turn_count += 1
        turn_id = f"T{self.turn_count}"

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"  {turn_id} — {speaker}: {text[:80]}{'...' if len(text)>80 else ''}")

        # Build prompt (truncate history for long-horizon conversations; engine state carries long-term memory)
        history_slice = (self.conversation_history[-self.max_history_turns:]
                         if self.max_history_turns is not None
                         else self.conversation_history)
        user_prompt = build_classify_prompt(
            history_slice, self.engine, speaker, text)

        # Call LLM
        raw_response = call_llm(self.classify_system, user_prompt, self.api_key,
                                temperature=self.temperature, seed=self.llm_seed)

        if not raw_response:
            if self.verbose:
                print("  [LLM CALL FAILED]")
            return {"turn": turn_id, "error": "LLM call failed"}

        # Parse
        classification = parse_classification(raw_response)

        if not classification:
            if self.verbose:
                print(f"  [PARSE FAILED]: {raw_response[:100]}")
            # Retry once
            raw_response = call_llm(self.classify_system, user_prompt, self.api_key,
                                    temperature=self.temperature, seed=self.llm_seed)
            classification = parse_classification(raw_response) if raw_response else None
            if not classification:
                return {"turn": turn_id, "error": "Parse failed",
                        "raw": raw_response[:200] if raw_response else ""}

        # Apply to engine
        log = apply_classification(self.engine, classification, turn_id, speaker,
                                   ablate_del_awareness=self.ablate_del_awareness)

        if self.verbose:
            ops = classification.get("operations", [])
            print(f"  Operations: {', '.join(ops)}")
            for line in log:
                print(line)

            # Show consistency warnings
            issues = self.engine.check_consistency()
            for issue in issues:
                print(f"  ⚠ ENGINE: {issue['message']}")

        # Update history
        self.conversation_history.append({"speaker": speaker, "text": text})
        self.classifications.append({
            "turn": turn_id, "speaker": speaker,
            "classification": classification, "log": log
        })

        return {
            "turn": turn_id,
            "operations": classification.get("operations", []),
            "classification": classification,
            "log": log,
        }

    def query_affected(self, assumption_id: str) -> dict:
        """Run a counterfactual query: what if this assumption were retracted?"""
        result = self.engine.retract_assumption(assumption_id)
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Counterfactual: Retract {assumption_id}")
            print(f"  Affected: {[h['id'] for h in result['affected']]}")
            print(f"  Unaffected: {[h['id'] for h in result['unaffected']]}")
        return result

    def get_state(self) -> str:
        """Get the current engine state summary."""
        return self.engine.get_state_summary()

    def get_dependency_map(self) -> dict:
        """Get the current dependency map."""
        return {k: list(v) for k, v in self.engine.dependencies.items()}


# ============================================================
# Built-in Conversations (for testing)
# ============================================================

PHASE2_CONVERSATION = [
    {"speaker": "Carol", "text": "P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation. Customer complaints started around 2:15am."},
    {"speaker": "Alice", "text": "I'm seeing 401 Unauthorized spikes in auth logs starting around 2am. Tokens being rejected as expired. But I also see our auth request volume is way up -- about 3x normal. That's strange. More users shouldn't be logging in at 2am."},
    {"speaker": "Bob", "text": "Payment side -- I see Stripe returning 429s. We're sending them way more requests than normal. And I can confirm I see the database alert too -- Redis connection timeouts from Payment Service."},
    {"speaker": "Carol", "text": "OK, so the database alert is about Redis, not the primary DB. Still concerning. Bob, are you seeing Redis issues from Payment's side?"},
    {"speaker": "Bob", "text": "Yes. Connection pool exhausted. We can't get connections to Redis. That's why our rate-limit checks are failing -- we store rate-limit counters in Redis. When we can't check the counter, our code falls through and sends the request to Stripe anyway. That would explain the 429s -- we're hitting Stripe without rate limiting."},
    {"speaker": "Carol", "text": "So the chain might be: Redis is sick -> Payment loses rate limiting -> Stripe gets hammered. And separately, Redis being sick -> Auth has problems too? Alice, does Auth use Redis?"},
    {"speaker": "Alice", "text": "Yes, Auth uses the same Redis cluster for session caching. If Redis is down, token validation would fail because we can't look up the session. That would explain the 401s... but wait, the 401s I'm seeing are specifically 'token expired,' not 'session lookup failed.' Those are different error codes. The tokens are being rejected because their expiry timestamps are in the past, not because Redis is unavailable."},
    {"speaker": "Carol", "text": "Hmm. So the auth failures might not be caused by Redis after all?"},
    {"speaker": "Alice", "text": "I don't think so. The error type is wrong. If Redis were down, Auth would return a 503 Service Unavailable, not a 401 with 'token expired.' I'm seeing 401s. So the token expiry issue is a different problem from the Redis problem."},
    {"speaker": "Bob", "text": "But then why is auth traffic 3x normal? If the token expiry issue is independent of Redis, what's generating all that auth traffic?"},
    {"speaker": "Alice", "text": "Could be the frontend retrying. When a user gets a 'token expired' error, the frontend automatically tries to refresh the token. If the new token is also expired, it retries again. That's a retry loop. So the token bug generates its own amplified traffic."},
    {"speaker": "Carol", "text": "Wait -- so the retry storm from the token bug could be what's exhausting Redis? Not Redis causing the auth problem, but the auth problem causing the Redis overload?"},
    {"speaker": "Alice", "text": "That's... actually plausible. If auth traffic is 3x, and every auth request hits Redis for session lookup, the Redis connection pool could be overwhelmed. So the chain would be: token bug -> retry storm -> Redis exhaustion. And then Redis exhaustion -> Payment rate-limit bypass -> Stripe 429s. The whole thing cascades from the token bug."},
]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the epistemic pipeline end-to-end")
    parser.add_argument("--conversation", type=str,
        help="Path to JSON file with conversation "
             "(list of {speaker, text} objects). "
             "Default: built-in Phase 2 scenario.")
    parser.add_argument("--query", type=str,
        help="After processing, run Affected(p) query for this assumption ID")
    parser.add_argument("--all-queries", action="store_true",
        help="Run Affected() for all tracked dependencies")
    parser.add_argument("--quiet", action="store_true",
        help="Suppress per-turn output")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic",
        help="API backend. 'openai' targets any OpenAI-compatible server (vLLM, Ollama, OpenAI).")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--base-url", type=str, default=None,
        help="Override API URL. Defaults: anthropic -> Messages API; openai -> http://localhost:8000/v1/chat/completions")
    args = parser.parse_args()

    global BACKEND, MODEL, API_URL
    BACKEND = args.backend
    if args.model:
        MODEL = args.model
    if args.base_url:
        API_URL = args.base_url
    elif BACKEND == "openai":
        API_URL = "http://localhost:8000/v1/chat/completions"

    if BACKEND == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    # Load conversation
    if args.conversation:
        with open(args.conversation) as f:
            conversation = json.load(f)
        print(f"Loaded {len(conversation)} turns from {args.conversation}")
    else:
        conversation = PHASE2_CONVERSATION
        print(f"Using built-in Phase 2 scenario ({len(conversation)} turns)")

    print(f"Model: {MODEL}")
    print()

    # Run pipeline
    pipe = EpistemicPipeline(api_key, verbose=not args.quiet)

    for turn in conversation:
        pipe.process_turn(turn["speaker"], turn["text"])
        time.sleep(0.5)  # Rate limit courtesy

    # Print final state
    print(f"\n{'='*60}")
    print(pipe.get_state())

    # Print dependency map
    deps = pipe.get_dependency_map()
    if deps:
        print(f"\n{'='*60}")
        print("DEPENDENCY MAP")
        for hyp_id, dep_list in deps.items():
            print(f"  Dep({hyp_id}) = {{{', '.join(sorted(dep_list))}}}")

    # Run queries
    if args.query:
        pipe.query_affected(args.query)
    elif args.all_queries:
        print(f"\n{'='*60}")
        print("COUNTERFACTUAL ANALYSIS")
        seen = set()
        for dep_set in pipe.engine.dependencies.values():
            for assumption in dep_set:
                if assumption not in seen and assumption in pipe.engine.observations:
                    seen.add(assumption)
                    pipe.query_affected(assumption)

    # Save classifications
    output_file = "pipeline_output.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL,
            "turns": pipe.classifications,
            "dependency_map": deps,
            "hypotheses": {
                hid: {"content": h.content, "status": h.status.value}
                for hid, h in pipe.engine.hypotheses.items()
            },
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
