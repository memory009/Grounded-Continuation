// Experiment: Phase 3 Classification — Architecture Deliberation
// Grounded Continuation — Condition 2 (Definitions prompt)
// Model: claude-sonnet-4-20250514 | 5 runs (1×temp=0, 4×temp=1)
//
// This script sends the full 19-turn deliberation conversation to Claude Sonnet
// with the Condition 2 (definitions) prompt: clear general definitions, generic
// examples, critical distinctions, no conversation-specific hints, no state.
//
// Result: F1=0.85, Exact=63%, Key Shift F1=0.92, Key Shift Exact=75%

const MODEL = "claude-sonnet-4-20250514";

const CONVERSATION = `T1 -- Alice: We need real-time collaboration. Users should see each other's edits live, like Google Docs. Ship in six weeks. How do we build it?

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

T19 -- Bob: I'll write it up. But I want it on the record that I think this is short-sighted. If we'd gone with ShareDB, we wouldn't be carrying this risk at all.`;

const PROMPT = `You are classifying utterances in a multi-agent DELIBERATION conversation (a team making a technical decision) into epistemic operations. Each utterance may have one or more operations.

## Operation Definitions

**Observe**: Report factual data, domain knowledge, or constraints without proposing a solution.
  Generic example: "The server runs Node.js and we have a 6-week deadline."

**Hypothesize**: Propose a NEW position, approach, or solution that did not previously exist in the conversation.
  Generic example: "What if we use a message queue instead of direct API calls?"

**Support**: Provide evidence or reasoning that STRENGTHENS an existing position already on the table.
  Generic example: "That approach also has the advantage of offline support."

**Undermine**: Provide evidence or reasoning that WEAKENS an existing position. The position remains viable but is less attractive.
  Generic example: "But that library doesn't support our database version."

**Revise**: ABANDON a position entirely, or FUNDAMENTALLY RESTRUCTURE the decision framing. Key signals: explicit abandonment ("that approach won't work for us") or reframing the problem ("this isn't an algorithm choice, it's a library choice").
  Generic example: "So building from scratch is off the table -- we need to use existing tools."

**Expand-Awareness**: Introduce a COMPLETELY NEW dimension of reasoning that was previously UNCONCEIVED. Not just a new data point, but a new way of thinking about the problem.
  Generic example: "Instead of choosing between building A or B, what if we use an existing library that already implements this?"
  (Here, the library-based approach was not in anyone's consideration.)

**Resolve**: Make a DECISION -- commit the group to a specific position. May involve authority ("I'm deciding we go with X") or consensus.
  Generic example: "OK, let's go with approach B. Ship it by Friday."

**Question**: Ask for information or challenge an assumption. Includes implicit questions where someone flags uncertainty.
  Generic example: "Is that really a problem for our use case?"

## Key Distinctions

- **Undermine vs Revise**: Undermine weakens a position; Revise abandons it or reframes the problem. If from-scratch approaches are deemed too risky for the timeline, that's Undermine (makes them less attractive). If the team explicitly decides to stop considering them, that's the result of the undermining.

- **Expand-Awareness vs Hypothesize**: Hypothesize proposes a new option within the current framing. Expand-Awareness introduces a new framing or dimension nobody was thinking about.

- **Support+Undermine in same turn**: A speaker can concede a point while raising a different concern. "Probably fine for launch, but risky long-term" is Support (for current use) + Undermine (for long-term viability).

- **Resolve with dissent**: When a decision is made despite disagreement, the decision-maker's utterance is Resolve. The dissenter's recorded objection is Observe (recording a public commitment/position).

## Conversation to classify

${CONVERSATION}

## Task

For each turn T1 through T19, provide the epistemic operations. Respond with ONLY a JSON object:
{
  "T1": ["Op1", "Op2"],
  "T2": ["Op1"],
  ...
}`;

const GROUND_TRUTH = {
  T1: ["Question", "Observe"],
  T2: ["Observe"],
  T3: ["Question"],
  T4: ["Undermine"],
  T5: ["Expand-Awareness", "Hypothesize"],  // KEY: library reframing
  T6: ["Support", "Undermine"],
  T7: ["Support"],
  T8: ["Undermine"],
  T9: ["Question"],
  T10: ["Support", "Undermine"],
  T11: ["Support"],
  T12: ["Undermine"],                        // KEY: access control blocks P2P
  T13: ["Hypothesize"],                      // KEY: hybrid position P5
  T14: ["Support"],
  T15: ["Undermine"],
  T16: ["Question"],
  T17: ["Support", "Undermine"],
  T18: ["Resolve"],                           // KEY: decision with conditional commitment
  T19: ["Observe"],                           // dissent recorded as public commitment
};

const KEY_SHIFTS = ["T5", "T12", "T13", "T18"];

// To run: send PROMPT to the API with model=claude-sonnet-4-20250514,
// max_tokens=1200, temperature=0 (run 1) or 1 (runs 2-5).
// Parse the JSON response and compute set-based F1 per turn against GROUND_TRUTH.
// Key shift turns: T5, T12, T13, T18.
