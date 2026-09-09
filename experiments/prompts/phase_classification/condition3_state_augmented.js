// Experiment Condition 3: State-Augmented Prompt
// Grounded Continuation — Phase 2 Classification
// Model: claude-sonnet-4-20250514 | 5 runs (1×temp=0, 4×temp=1)
//
// This script sends the full 13-turn debugging conversation to Claude Sonnet
// with all definitions from Condition 2, PLUS epistemic model state at key turns
// and conversation-specific examples in the operation definitions. This simulates
// what the symbolic engine would provide in the hybrid architecture.
//
// Result: F1=0.91, Exact=63%, Key Shift F1=0.83, Key Shift Exact=30%

const MODEL = "claude-sonnet-4-20250514";

const CONVERSATION = `T1 -- Carol: P1 incident. Three alerts firing: auth failure rate up, payment failure rate up, database health degradation. Customer complaints started around 2:15am.

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

T13 -- Alice: That's... actually plausible. If auth traffic is 3x, and every auth request hits Redis for session lookup, the Redis connection pool could be overwhelmed. So the chain would be: token bug -> retry storm -> Redis exhaustion. And then Redis exhaustion -> Payment rate-limit bypass -> Stripe 429s. The whole thing cascades from the token bug.`;

const PROMPT = `You are an expert epistemic analyst classifying utterances in a multi-agent debugging conversation.

## Operations (classify each turn into ONE OR MORE)

- **Observe**: Reporting new factual data (error codes, metrics, timestamps). NOT explaining or hypothesising.
- **Hypothesize**: Proposing a NEW causal explanation that didn't exist before. Must be a novel explanatory chain, not restating someone else's hypothesis.
- **Support**: Providing evidence FOR an existing hypothesis (not proposing a new one).
- **Undermine**: Providing evidence AGAINST an existing hypothesis, weakening but not abandoning it. The hypothesis remains possible but less plausible.
- **Revise**: ABANDONING a hypothesis entirely or RESTRUCTURING the causal model. Key signal: the speaker explicitly states the old belief is wrong, not just weakened. Also applies when the causal direction between two things is reversed.
- **Expand-Awareness**: Introducing a concept/proposition that was COMPLETELY OUTSIDE anyone's reasoning until now. NOT revising an existing belief — discovering a new dimension. Example: realising a "database" alert is actually about Redis (the monitoring miscategorisation was unconceived, not merely unknown).
- **Resolve**: Elevating a tentative hypothesis to an accepted conclusion. The group converges on an answer.
- **Question**: Asking for information. Includes IMPLICIT questions where a statement opens a new line of inquiry (e.g., "We need to figure out X" implicitly asks "how?").

## Critical distinctions (these are where errors happen most)

1. **Undermine vs Revise**: Undermine = "this evidence weakens hypothesis h." Revise = "hypothesis h is WRONG, we're abandoning it." If someone says "so X is a different problem from Y," that's Revise (separation), not Undermine (weakening).

2. **Expand-Awareness vs Observe/Revise**: Expand-Awareness = a proposition enters reasoning that was previously UNCONCEIVED (not just unknown). If someone realises "the database alert is actually about Redis," this introduces the concept of monitoring miscategorisation — something nobody was thinking about. This is NOT Revise (no prior belief is being changed) and NOT just Observe (it's not reporting a measurement).

3. **Implicit Questions**: When someone opens a discussion ("P1 incident, three alerts firing") or flags something anomalous ("That's strange, 3x traffic at 2am"), they are implicitly asking "what's causing this?" — classify as Question in addition to Observe.

4. **Hypothesize+Revise**: When someone proposes a new causal direction that REVERSES a previous understanding (e.g., "not A causes B, but B causes A"), this is BOTH Hypothesize (new causal link) AND Revise (restructuring the causal model).

## Conversation

${CONVERSATION}

## Epistemic model state at each turn (for context)

Before T1: No observations or hypotheses.
Before T4: Observations o1-o8. No hypotheses yet. Nobody has considered monitoring miscategorisation.
Before T5: Awareness expanded: "database" alert is really Redis.
Before T7: h1 (Redis->Stripe 429s) ACTIVE. h2 (Redis->Auth failures) ACTIVE. Leading picture: Redis is single root cause.
Before T9: h2 WEAKENED by T7's error code evidence.
Before T12: h2 ABANDONED. h3 (token bug->retry->traffic) ACTIVE. Two independent causal chains.

## Task

For EACH turn T1-T13, classify into operations. Respond with ONLY this JSON:
{
  "T1": ["Op1", "Op2"],
  "T2": ["Op1"],
  ...
}`;

const GROUND_TRUTH = {
  T1: ["Observe", "Question"], T2: ["Observe", "Question"],
  T3: ["Observe"], T4: ["Expand-Awareness", "Question"],
  T5: ["Observe", "Hypothesize"], T6: ["Support", "Hypothesize", "Question"],
  T7: ["Observe", "Undermine"], T8: ["Question"],
  T9: ["Revise"], T10: ["Question"],
  T11: ["Hypothesize"], T12: ["Hypothesize", "Revise"],
  T13: ["Support", "Resolve"],
};

// To run: send PROMPT to the API with model=claude-sonnet-4-20250514,
// max_tokens=1000, temperature=0 (run 1) or 1 (runs 2-5).
// Parse the JSON response and compute set-based F1 per turn against GROUND_TRUTH.
