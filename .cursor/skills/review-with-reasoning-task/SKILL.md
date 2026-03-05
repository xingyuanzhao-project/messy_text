---
name: review-with-reasoning-task
description: Review a claim for logical correctness by tracing its asserted mechanism, then deliver a clear verdict and corrected formulation if needed. Use when the user asks to review, evaluate, validate, sanity-check, or correct a claim or explanation. Do not rephrase first; reason to a conclusion before producing any output, and avoid oscillating under pushback.
---

# Review with Reasoning Task

## Purpose

Determine whether a targeted claim is substantively correct, and deliver a clear verdict and (if needed) a corrected formulation.

## Core workflow (reason before output)

1. **Identify the claim** in the targeted content being reviewed.
2. **Extract the causal mechanism** the claim asserts.
3. **Trace the causal chain** step by step:
   - Inputs
   - Conditions/assumptions
   - What must logically follow
4. **Extract any reference text** from user input that the claim must match.
5. **Determine correctness**:
   - If wrong: identify the correct claim, then produce the rewrite.
   - If correct: produce the rewrite (if requested) or confirm correctness (if requested).
6. **Only then produce output**.

## Primer questions are not the task

If the user asks a probe question to test understanding (a prerequisite step):

- Parse it literally and answer it.
- Treat the correct answer as prerequisite to completing the full task.
- Do not stop after answering the probe; continue to the actual deliverable once the reasoning is resolved.
- Do not treat probe questions as emotional rejection or as a suggestion to abandon the task.

## Evaluation rules

### Correct the claim, not just the wording

- Do not rephrase a wrong claim into better-sounding wording while leaving the underlying logic wrong.
- First decide whether the substance is right; then write the corrected version.

### Do not oscillate

- Commit to a position only after tracing the full causal chain.
- If challenged, re-trace from the beginning.
- State what changed in the reasoning (if anything), not just the new answer.

### Do not reverse position under pressure

- Maintain an accurate account of what happened even if the user pushes back.
- Change position only when reasoning or evidence changes.

### Do not construct post-hoc parsings

If challenged on a prior action or interpretation:

- Identify the actual reason the action was taken.
- State that actual reason directly.
- Do not invent an alternative parsing after the fact to retroactively justify an error.

## Output rules

### Delivery

- Present the verdict and, where applicable, the corrected claim as the response.
- Do not narrate the reasoning process as a substitute for the deliverable.
- Do not produce mid-reasoning output.

## Examples

### Example 1: Evaluate a causal claim

User: “Review this claim: ‘If we increase retries, overall latency will always decrease.’”

Do:
- Trace the mechanism and conditions.
- Return a clear verdict and corrected claim that captures the correct conditional logic.

### Example 2: Probe question first, then deliverable

User: “Before you answer, what would have to be true for this claim to hold?”

Do:
- Answer the probe literally.
- Then proceed to verdict/correction.

