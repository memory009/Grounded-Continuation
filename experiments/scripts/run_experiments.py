#!/usr/bin/env python3
"""
Reproducibility Script — Grounded Continuation classification experiment
ICLR 2027 submission

Runs all classification experiments and reports results.

Requirements:
    pip install requests

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --phase2-only      # Phase 2 only (3 conditions)
    python run_experiments.py --phase3-only      # Phase 3 only
    python run_experiments.py --runs 3           # Override number of runs (default 5)
    python run_experiments.py --save results.json  # Save results to JSON

Expected results (Claude Sonnet, 5 runs):
    Phase 2 Minimal:       F1=0.66, Key=0.50
    Phase 2 Definitions:   F1=0.85, Key=0.74
    Phase 2 State-aug:     F1=0.91, Key=0.83
    Phase 3 Definitions:   F1=0.85, Key=0.92
"""

import os, sys, json, time, argparse
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: pip install requests")
    sys.exit(1)

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
BACKEND = "anthropic"

# ============================================================
# Conversations
# ============================================================

PHASE2_CONVERSATION = """T1 -- Carol: P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation. Customer complaints started around 2:15am.

T2 -- Alice: I'm seeing 401 Unauthorized spikes in auth logs starting around 2am. Tokens being rejected as expired. But I also see our auth request volume is way up -- about 3x normal. That's strange. More users shouldn't be logging in at 2am.

T3 -- Bob: Payment side -- I see Stripe returning 429s. We're sending them way more requests than normal. And I can confirm I see the database alert too -- Redis connection timeouts from Payment Service.

T4 -- Carol: OK, so the database alert is about Redis, not the primary DB. Still concerning. Bob, are you seeing Redis issues from Payment's side?

T5 -- Bob: Yes. Connection pool exhausted. We can't get connections to Redis. That's why our rate-limit checks are failing -- we store rate-limit counters in Redis. When we can't check the counter, our code falls through and sends the request to Stripe anyway. That would explain the 429s -- we're hitting Stripe without rate limiting.

T6 -- Carol: So the chain might be: Redis is sick -> Payment loses rate limiting -> Stripe gets hammered. And separately, Redis being sick -> Auth has problems too? Alice, does Auth use Redis?

T7 -- Alice: Yes, Auth uses the same Redis cluster for session caching. If Redis is down, token validation would fail because we can't look up the session. That would explain the 401s... but wait, the 401s I'm seeing are specifically 'token expired,' not 'session lookup failed.' Those are different error codes. The tokens are being rejected because their expiry timestamps are in the past, not because Redis is unavailable.

T8 -- Carol: Hmm. So the auth failures might not be caused by Redis after all?

T9 -- Alice: I don't think so. The error type is wrong. If Redis were down, Auth would return a 503 Service Unavailable, not a 401 with 'token expired.' I'm seeing 401s. So the token expiry issue is a different problem from the Redis problem.

T10 -- Bob: But then why is auth traffic 3x normal? If the token expiry issue is independent of Redis, what's generating all that auth traffic?

T11 -- Alice: Could be the frontend retrying. When a user gets a 'token expired' error, the frontend automatically tries to refresh the token. If the new token is also expired, it retries again. That's a retry loop. So the token bug generates its own amplified traffic.

T12 -- Carol: Wait -- so the retry storm from the token bug could be what's exhausting Redis? Not Redis causing the auth problem, but the auth problem causing the Redis overload?

T13 -- Alice: That's... actually plausible. If auth traffic is 3x, and every auth request hits Redis for session lookup, the Redis connection pool could be overwhelmed. So the chain would be: token bug -> retry storm -> Redis exhaustion. And then Redis exhaustion -> Payment rate-limit bypass -> Stripe 429s. The whole thing cascades from the token bug."""

PHASE3_CONVERSATION = """T1 -- Alice: We need real-time collaboration. Users should see each other's edits live, like Google Docs. Ship in six weeks. How do we build it?

T2 -- Bob: Two established approaches: Operational Transformation -- that's what Google Docs uses -- and CRDTs, which is what Figma and newer tools use. OT needs a central server for coordination. CRDTs are peer-to-peer capable but more complex to implement.

T3 -- Alice: What about the six-week timeline? Can we ship either one?

T4 -- Bob: Implementing either from scratch in six weeks is risky. OT's transformation functions are subtle and buggy. CRDTs have complex data structures.

T5 -- Carol: What about using a library? Yjs is a mature CRDT library. ShareDB for OT.

T6 -- Carol: I've prototyped with Yjs before on a side project. The yjs-prosemirror binding -- which is our editor -- is well documented. I don't know if ShareDB has the same ProseMirror integration.

T7 -- Bob: Both work with our Node.js backend. But with Yjs we could go serverless or use a central server. More architectural flexibility.

T8 -- Bob: CRDTs have a known problem with document size. The CRDT metadata grows over time and can get large for long-lived documents. Yjs has some GC mechanisms but they're not trivial.

T9 -- Alice: Is that a problem for our initial launch? Our documents are typically 5-10 pages.

T10 -- Bob: Probably not for launch. It's a long-term concern. But I want to flag it because switching from CRDT to OT later would be a rewrite, not a refactor.

T11 -- Carol: If we go with Yjs and WebRTC, we could support offline editing natively. User research showed spotty connectivity is a pain point.

T12 -- Bob: Hmm, but if edits are peer-to-peer, access control is hard. We need role-based permissions.

T13 -- Carol: Can we use Yjs but with a central server as the sync point? We'd get the CRDT benefits -- conflict resolution, offline merge -- but the server can enforce access control.

T14 -- Bob: Yes, that's actually the recommended production setup for Yjs. You run a Yjs WebSocket server as the sync point. And we already run WebSocket servers for notifications.

T15 -- Bob: I want to come back to the document size issue. If we go CRDT, every edit operation is stored permanently in the CRDT state. For a 10-page document edited for months, the CRDT metadata could be 10-50x larger than the content. Yjs has compaction but it's not trivial. And switching from CRDT to OT later would be a six-month rewrite.

T16 -- Alice: How confident are you that the problem will actually manifest? Our documents are short and have burst editing -- a few days of activity, then they become read-only.

T17 -- Bob: For the current use case, probably 80% chance it's fine. But the Q2 roadmap includes long-running project documents. Those would be edited continuously for months.

T18 -- Alice: Q2 isn't confirmed. I don't want to make an architectural decision now based on a feature that might not happen. Here's what I propose: we go with Yjs for launch. Bob, write up the risk with specific thresholds -- when should we start worrying. If Q2 confirms long-running documents, we evaluate then.

T19 -- Bob: I'll write it up. But I want it on the record that I think this is short-sighted. If we'd gone with ShareDB, we wouldn't be carrying this risk at all."""

# ============================================================
# Ground Truth
# ============================================================

PHASE2_GT = {
    "T1": ["Observe", "Question"], "T2": ["Observe", "Question"],
    "T3": ["Observe"], "T4": ["Expand-Awareness", "Question"],
    "T5": ["Observe", "Hypothesize"], "T6": ["Support", "Hypothesize", "Question"],
    "T7": ["Observe", "Undermine"], "T8": ["Question"],
    "T9": ["Revise"], "T10": ["Question"],
    "T11": ["Hypothesize"], "T12": ["Hypothesize", "Revise"],
    "T13": ["Support", "Resolve"],
}
PHASE2_KEY_SHIFTS = ["T4", "T7", "T9", "T12"]

PHASE3_GT = {
    "T1": ["Question", "Observe"], "T2": ["Observe"],
    "T3": ["Question"], "T4": ["Undermine"],
    "T5": ["Expand-Awareness", "Hypothesize"], "T6": ["Support", "Undermine"],
    "T7": ["Support"], "T8": ["Undermine"],
    "T9": ["Question"], "T10": ["Support", "Undermine"],
    "T11": ["Support"], "T12": ["Undermine"],
    "T13": ["Hypothesize"], "T14": ["Support"],
    "T15": ["Undermine"], "T16": ["Question"],
    "T17": ["Support", "Undermine"], "T18": ["Resolve"],
    "T19": ["Observe"],
}
PHASE3_KEY_SHIFTS = ["T5", "T12", "T13", "T18"]

# ============================================================
# Prompts
# ============================================================

PROMPT_MINIMAL = """You are classifying each utterance in a debugging conversation into epistemic operations.

The valid operations are:
- Observe: reporting factual observations
- Hypothesize: proposing a new explanatory hypothesis
- Support: evidence increasing confidence in existing hypothesis
- Undermine: evidence decreasing confidence in existing hypothesis
- Revise: explicitly abandoning or restructuring a belief
- Expand-Awareness: introducing entirely new concept not previously considered
- Resolve: elevating hypothesis to accepted conclusion
- Question: asking for information or clarification

Each utterance can have one or more operations.

Here is the conversation:

{conversation}

For EACH turn, classify into operations. Respond with ONLY a JSON object:
{{"T1": ["Op1", "Op2"], "T2": ["Op1"], ...}}"""

PROMPT_DEFINITIONS_PHASE2 = """You are classifying utterances in a multi-agent conversation into epistemic operations. Each utterance may have one or more operations.

## Operation Definitions

**Observe**: Report factual data without explanation. New measurements, error codes, metrics, timestamps.
  Generic example: "The server logs show 502 errors starting at 3pm."

**Hypothesize**: Propose a NEW explanatory causal chain that did not previously exist in the conversation.
  Generic example: "Maybe the memory leak is causing the timeouts."

**Support**: Provide evidence or reasoning that STRENGTHENS an existing hypothesis already proposed by someone.
  Generic example: "That's consistent with what I'm seeing in the logs too."

**Undermine**: Provide evidence or reasoning that WEAKENS an existing hypothesis.
  Generic example: "But the timing doesn't match -- the errors started before the deployment."

**Revise**: ABANDON a hypothesis entirely, or FUNDAMENTALLY RESTRUCTURE the group's causal understanding.
  Generic example: "So the network issue isn't related to the database at all -- they're independent problems."
  Generic example: "Wait, it's the opposite -- the client is overwhelming the server, not the server failing on its own."

**Expand-Awareness**: Introduce a COMPLETELY NEW dimension of reasoning that was previously UNCONCEIVED.
  Generic example: "Actually, the 'disk full' alert is being triggered by the logging system, not actual disk usage -- the monitoring is misconfigured."

**Resolve**: Elevate a tentative hypothesis to an accepted group conclusion.
  Generic example: "Yes, that's it -- the root cause is the configuration change from yesterday."

**Question**: Ask for information or open a new line of inquiry. Includes IMPLICIT questions.
  Generic example: "Something is off -- we shouldn't be seeing this much traffic at night."

## Key Distinctions
- **Undermine vs Revise**: Undermine weakens; Revise abandons or restructures.
- **Expand-Awareness vs Observe**: Observe adds data within existing reasoning. Expand-Awareness adds a new dimension nobody was thinking about.
- **Hypothesize+Revise**: Proposing a new causal chain that reverses a previous understanding = both.

## Conversation to classify

{conversation}

For each turn, provide operations. JSON only:
{{"T1": ["Op1", "Op2"], "T2": ["Op1"], ...}}"""

PROMPT_STATE_AUGMENTED = """You are an expert epistemic analyst classifying utterances in a multi-agent debugging conversation.

## Operations (classify each turn into ONE OR MORE)

- **Observe**: Reporting new factual data. NOT explaining or hypothesising.
- **Hypothesize**: Proposing a NEW causal explanation that didn't exist before.
- **Support**: Providing evidence FOR an existing hypothesis.
- **Undermine**: Providing evidence AGAINST an existing hypothesis, weakening but not abandoning it.
- **Revise**: ABANDONING a hypothesis entirely or RESTRUCTURING the causal model. Key signal: explicit statement that old belief is wrong. Also applies when causal direction is reversed.
- **Expand-Awareness**: Introducing a concept that was COMPLETELY OUTSIDE anyone's reasoning. Example: realising a "database" alert is actually about Redis.
- **Resolve**: Elevating a tentative hypothesis to an accepted conclusion.
- **Question**: Asking for information. Includes IMPLICIT questions.

## Critical distinctions
1. **Undermine vs Revise**: Undermine = "this evidence weakens hypothesis h." Revise = "hypothesis h is WRONG, we're abandoning it."
2. **Expand-Awareness vs Observe/Revise**: Expand-Awareness = a proposition enters reasoning that was previously UNCONCEIVED.
3. **Implicit Questions**: Opening a discussion or flagging something anomalous = Question.
4. **Hypothesize+Revise**: New causal direction that REVERSES previous understanding = BOTH.

## Conversation

{conversation}

## Epistemic model state at each turn
Before T1: No observations or hypotheses.
Before T4: Observations o1-o8. No hypotheses yet. Nobody has considered monitoring miscategorisation.
Before T5: Awareness expanded: "database" alert is really Redis.
Before T7: h1 (Redis->Stripe 429s) ACTIVE. h2 (Redis->Auth failures) ACTIVE. Leading picture: Redis is single root cause.
Before T9: h2 WEAKENED by T7's error code evidence.
Before T12: h2 ABANDONED. h3 (token bug->retry->traffic) ACTIVE. Two independent causal chains.

## Task
For EACH turn, classify into operations. JSON only:
{{"T1": ["Op1", "Op2"], "T2": ["Op1"], ...}}"""

PROMPT_DEFINITIONS_PHASE3 = """You are classifying utterances in a multi-agent DELIBERATION conversation (a team making a technical decision) into epistemic operations. Each utterance may have one or more operations.

## Operation Definitions

**Observe**: Report factual data, domain knowledge, or constraints without proposing a solution.
  Generic example: "The server runs Node.js and we have a 6-week deadline."

**Hypothesize**: Propose a NEW position, approach, or solution that did not previously exist.
  Generic example: "What if we use a message queue instead of direct API calls?"

**Support**: Provide evidence or reasoning that STRENGTHENS an existing position.
  Generic example: "That approach also has the advantage of offline support."

**Undermine**: Provide evidence or reasoning that WEAKENS an existing position.
  Generic example: "But that library doesn't support our database version."

**Revise**: ABANDON a position entirely, or FUNDAMENTALLY RESTRUCTURE the decision framing.
  Generic example: "So building from scratch is off the table -- we need to use existing tools."

**Expand-Awareness**: Introduce a COMPLETELY NEW dimension of reasoning previously UNCONCEIVED.
  Generic example: "Instead of choosing between building A or B, what if we use an existing library?"

**Resolve**: Make a DECISION -- commit the group to a specific position.
  Generic example: "OK, let's go with approach B. Ship it by Friday."

**Question**: Ask for information or challenge an assumption.
  Generic example: "Is that really a problem for our use case?"

## Key Distinctions
- **Undermine vs Revise**: Undermine weakens; Revise abandons or reframes.
- **Expand-Awareness vs Hypothesize**: Hypothesize proposes within current framing. Expand-Awareness introduces new framing.
- **Support+Undermine in same turn**: Conceding a point while raising a different concern.
- **Resolve with dissent**: Decision-maker = Resolve. Dissenter's objection = Observe.

## Conversation to classify

{conversation}

For each turn T1-T19, provide operations. JSON only:
{{"T1": ["Op1", "Op2"], "T2": ["Op1"], ...}}"""

# ============================================================
# API + Parsing
# ============================================================

VALID_OPS = {"Observe", "Hypothesize", "Support", "Undermine", "Revise",
             "Expand-Awareness", "Resolve", "Question"}

NORMALIZE_MAP = {
    "observe": "Observe", "hypothesize": "Hypothesize", "hypothesis": "Hypothesize",
    "support": "Support", "undermine": "Undermine", "revise": "Revise",
    "expandawareness": "Expand-Awareness", "awareness": "Expand-Awareness",
    "resolve": "Resolve", "question": "Question",
}


def normalize_op(s: str) -> Optional[str]:
    if not s:
        return None
    key = s.lower().replace("-", "").replace("_", "").replace(" ", "")
    return NORMALIZE_MAP.get(key)


def parse_response(text: str, turn_ids: list[str]) -> Optional[dict]:
    """Extract a {turn_id: [operations]} dict from the LLM response."""
    if not text:
        return None
    s = text.strip().replace("```json", "").replace("```", "").strip()
    import re
    m = re.search(r'\{[\s\S]+\}', s)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    result = {}
    for tid in turn_ids:
        val = obj.get(tid) or obj.get(tid.lower()) or []
        if isinstance(val, str):
            val = [val]
        if isinstance(val, list):
            result[tid] = [o for o in (normalize_op(x) for x in val) if o]
        else:
            result[tid] = []
    return result


def call_api(prompt: str, temperature: float, api_key: str, max_retries: int = 3) -> Optional[str]:
    """Dispatch to Anthropic or OpenAI-compatible endpoint based on BACKEND."""
    if BACKEND == "openai":
        return _call_openai(prompt, temperature, api_key, max_retries)
    return _call_anthropic(prompt, temperature, api_key, max_retries)


def _call_anthropic(prompt: str, temperature: float, api_key: str, max_retries: int) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                json={"model": MODEL, "max_tokens": 1200, "temperature": temperature,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60)
            data = resp.json()
            if "content" in data and data["content"]:
                return data["content"][0]["text"]
            print(f"    API error (attempt {attempt+1}): {data.get('error', {}).get('message', 'unknown')}")
        except Exception as e:
            print(f"    Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


def _call_openai(prompt: str, temperature: float, api_key: str, max_retries: int) -> Optional[str]:
    """OpenAI-compatible chat/completions (vLLM, Ollama, OpenAI, etc.)."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                json={"model": MODEL, "max_tokens": 1200, "temperature": temperature,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=120)
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            print(f"    API error (attempt {attempt+1}): {data.get('error', {}).get('message', data)}")
        except Exception as e:
            print(f"    Request error (attempt {attempt+1}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return None


# ============================================================
# Metrics
# ============================================================

def compute_f1(predicted: list[str], ground_truth: list[str]) -> float:
    """Set-based multi-label F1."""
    p_set, g_set = set(predicted), set(ground_truth)
    if not p_set and not g_set:
        return 1.0
    tp = len(p_set & g_set)
    prec = tp / len(p_set) if p_set else 0
    rec = tp / len(g_set) if g_set else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def exact_match(predicted: list[str], ground_truth: list[str]) -> bool:
    return set(predicted) == set(ground_truth)


# ============================================================
# Run Experiment
# ============================================================

def run_condition(name: str, prompt_template: str, conversation: str,
                  ground_truth: dict, key_shifts: list[str],
                  num_runs: int, api_key: str) -> dict:
    """Run one experimental condition and return aggregated results."""
    turn_ids = sorted(ground_truth.keys(), key=lambda t: int(t[1:]))
    prompt = prompt_template.format(conversation=conversation)

    all_runs = []
    for run_id in range(num_runs):
        temp = 0.0 if run_id == 0 else 1.0
        print(f"  Run {run_id+1}/{num_runs} (temp={temp})...", end=" ", flush=True)

        raw_text = call_api(prompt, temp, api_key)
        if not raw_text:
            print("FAILED")
            continue

        parsed = parse_response(raw_text, turn_ids)
        if not parsed:
            print(f"PARSE FAILED (response: {raw_text[:80]}...)")
            continue

        run_results = {}
        for tid in turn_ids:
            pred = parsed.get(tid, [])
            gt = ground_truth[tid]
            run_results[tid] = {
                "predicted": pred, "ground_truth": gt,
                "f1": compute_f1(pred, gt),
                "exact": exact_match(pred, gt),
                "is_key": tid in key_shifts,
            }
        all_runs.append(run_results)
        avg_f1 = sum(r["f1"] for r in run_results.values()) / len(run_results)
        print(f"F1={avg_f1:.2f}")

    if not all_runs:
        return {"name": name, "error": "No successful runs", "num_runs": 0}

    # Aggregate
    n = len(all_runs)
    per_turn = {}
    for tid in turn_ids:
        f1s = [run[tid]["f1"] for run in all_runs]
        exacts = [1 if run[tid]["exact"] else 0 for run in all_runs]
        per_turn[tid] = {
            "ground_truth": ground_truth[tid],
            "is_key": tid in key_shifts,
            "avg_f1": sum(f1s) / n,
            "exact_pct": sum(exacts) / n * 100,
            "predictions": [run[tid]["predicted"] for run in all_runs],
        }

    overall_f1 = sum(t["avg_f1"] for t in per_turn.values()) / len(per_turn)
    overall_exact = sum(t["exact_pct"] for t in per_turn.values()) / len(per_turn)
    key_turns = {tid: t for tid, t in per_turn.items() if t["is_key"]}
    key_f1 = sum(t["avg_f1"] for t in key_turns.values()) / len(key_turns) if key_turns else 0
    key_exact = sum(t["exact_pct"] for t in key_turns.values()) / len(key_turns) if key_turns else 0

    return {
        "name": name, "num_runs": n,
        "overall_f1": round(overall_f1, 3),
        "overall_exact": round(overall_exact, 1),
        "key_shift_f1": round(key_f1, 3),
        "key_shift_exact": round(key_exact, 1),
        "per_turn": per_turn,
    }


def print_results(result: dict):
    """Pretty-print results for one condition."""
    if "error" in result:
        print(f"\n  {result['name']}: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  {result['name']} ({result['num_runs']} runs)")
    print(f"{'='*60}")
    print(f"  {'Turn':<6} {'Ground Truth':<30} {'F1':>6} {'Exact':>6} {'Key':>4}")
    print(f"  {'-'*56}")

    for tid in sorted(result["per_turn"].keys(), key=lambda t: int(t[1:])):
        t = result["per_turn"][tid]
        gt_str = ", ".join(t["ground_truth"])
        key_str = "★" if t["is_key"] else ""
        print(f"  {tid:<6} {gt_str:<30} {t['avg_f1']:>6.2f} {t['exact_pct']:>5.0f}% {key_str:>4}")

    print(f"  {'-'*56}")
    print(f"  {'OVERALL':<37} {result['overall_f1']:>6.2f} {result['overall_exact']:>5.0f}%")
    print(f"  {'KEY SHIFTS':<37} {result['key_shift_f1']:>6.2f} {result['key_shift_exact']:>5.0f}%  ★")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run epistemic classification experiments")
    parser.add_argument("--phase2-only", action="store_true", help="Run only Phase 2 experiments")
    parser.add_argument("--phase3-only", action="store_true", help="Run only Phase 3 experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per condition (default: 5)")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
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
            print("  export ANTHROPIC_API_KEY='sk-ant-...'")
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    print(f"Backend: {BACKEND}")
    print(f"API URL: {API_URL}")
    print(f"Model: {MODEL}")
    print(f"Runs per condition: {args.runs}")
    print(f"API key: ...{api_key[-8:] if len(api_key) >= 8 else api_key}")
    print()

    all_results = []

    # Phase 2
    if not args.phase3_only:
        print("=" * 60)
        print("PHASE 2: System Debugging (13 turns)")
        print("=" * 60)

        print("\nCondition 1: Minimal prompt")
        r1 = run_condition("Phase 2 — Minimal", PROMPT_MINIMAL,
                           PHASE2_CONVERSATION, PHASE2_GT, PHASE2_KEY_SHIFTS,
                           args.runs, api_key)
        print_results(r1)
        all_results.append(r1)

        print("\nCondition 2: Definitions prompt")
        r2 = run_condition("Phase 2 — Definitions", PROMPT_DEFINITIONS_PHASE2,
                           PHASE2_CONVERSATION, PHASE2_GT, PHASE2_KEY_SHIFTS,
                           args.runs, api_key)
        print_results(r2)
        all_results.append(r2)

        print("\nCondition 3: State-augmented prompt")
        r3 = run_condition("Phase 2 — State-augmented", PROMPT_STATE_AUGMENTED,
                           PHASE2_CONVERSATION, PHASE2_GT, PHASE2_KEY_SHIFTS,
                           args.runs, api_key)
        print_results(r3)
        all_results.append(r3)

    # Phase 3
    if not args.phase2_only:
        print("\n" + "=" * 60)
        print("PHASE 3: Architecture Deliberation (19 turns)")
        print("=" * 60)

        print("\nCondition 2: Definitions prompt")
        r4 = run_condition("Phase 3 — Definitions", PROMPT_DEFINITIONS_PHASE3,
                           PHASE3_CONVERSATION, PHASE3_GT, PHASE3_KEY_SHIFTS,
                           args.runs, api_key)
        print_results(r4)
        all_results.append(r4)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Condition':<30} {'F1':>6} {'Exact':>6} {'Key F1':>7} {'Key Ex':>7}")
    print(f"  {'-'*58}")
    for r in all_results:
        if "error" not in r:
            print(f"  {r['name']:<30} {r['overall_f1']:>6.2f} {r['overall_exact']:>5.0f}% {r['key_shift_f1']:>7.2f} {r['key_shift_exact']:>6.0f}%")

    # Save
    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
