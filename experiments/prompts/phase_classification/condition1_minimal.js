// Experiment Condition 1: Minimal Prompt
// Grounded Continuation — Phase 2 Classification
// Model: claude-sonnet-4-20250514 | 5 runs (1×temp=0, 4×temp=1)
//
// This script sends the full 13-turn debugging conversation to Claude Sonnet
// with a MINIMAL prompt: short operation definitions, no examples, no
// distinction guidance, no epistemic model state.
//
// Result: F1=0.66, Exact=11%, Key Shift F1=0.50, Key Shift Exact=5%

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

const PROMPT = `You are classifying each utterance in a debugging conversation into epistemic operations.

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

${CONVERSATION}

For EACH turn (T1 through T13), classify it into one or more operations.

Respond with ONLY a JSON object mapping turn IDs to arrays of operations:
{
  "T1": ["Observe", "Question"],
  "T2": ["Observe"],
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
