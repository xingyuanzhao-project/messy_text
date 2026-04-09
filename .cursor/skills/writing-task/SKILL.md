---
name: writing-task
description: Write new prose into a specific target area in an existing draft. Use when the user asks to write, insert, add, or draft text into a named section, line range, or at a literal placeholder string in a file. Enforce strict scope boundaries and plain-prose formatting (no bullets, no markdown emphasis, no em dashes, avoid special characters like parentheses and quotes except citations if required).
---

# Writing Task

## Purpose

Create new prose content at a user-specified target location, while obeying scope boundaries and formatting constraints.

## Required extractions (do before writing)

- **Target area**: Where to write (line range, named section, or literal placeholder string).
- **Writing purpose**: What this text must accomplish in the document.
- **Writing content**: The points/facts the user wants included (and any source material provided).
- **Scope and boundaries**: What is explicitly in-scope vs out-of-scope for edits.
- **Style/tone/format**: Any explicit constraints; otherwise infer from surrounding draft.

## Targeting rules

### Boundaries

- Touch only the extracted target area.
- Do not insert content outside the stated target.

### Literal string target (highest priority when present)

If the instruction names a literal string as the insertion target (for example: “write in insert text here”):

- Locate that exact string in the file and write at that location.
- Treat any attached selection/snippet as context only; it does not override the literal-string target.
- Do not assume the target’s granularity; it may be a sentence span, multiple lines, or a full paragraph region. Follow the instruction’s literal targeting text.

## Formatting constraints

### Characters

- Write plain prose only.
- Avoid special characters in the written content:
  - No parentheses `()`
  - No quotation marks `"` (or other quote characters)
  - No markdown emphasis like `**bold**`
  - No em dashes
- Parentheses are allowed only for citations if citations are needed by instruction or clearly required by the document’s established conventions.

### Structure

- Produce a concise, logically structured, coherent paragraph.
- Do not use bullet lists. “List-like” content must be integrated into flowing prose.
- If length is not specified, infer the appropriate length from:
  - The purpose of the target section
  - The amount of substance to cover
  - The density and pacing of surrounding draft text

### Citation placement

Distribute citations per claim. Each assertion that depends on external evidence must carry its own citation at the point where the claim appears. Do not bundle multiple citations at the end of a multi-claim passage.

#### Rule

Do:
- Attach each `\cite{}` to the sentence or clause that makes the claim it supports. The reader must be able to trace each specific claim to its source.
- If two claims share the same source, cite that source independently at each claim's location.
- Bundle multiple citations after a single claim when all cited sources support that same claim. One claim with multiple sources is not bundling because attribution is unambiguous.
- Verify which source supports which claim before distributing citations.

Do not:
- Bundle citations from multiple distinct claims at the end of a passage. This forces the reader to guess which source supports which claim.
- Distribute citations per claim without knowing which source supports which claim. Distributing without verified mapping will expose the uncertainty as misattribution.

#### Examples

Wrong — bundled citations after multiple claims:

```latex
Measurement error can attenuate estimates, and the bias direction
depends on whether errors are classical or differential, with recent
work showing that LLM classifiers produce non-classical errors
\cite{Fong2021, Egami2022, Grimmer2022, Keith2020}.
```

Correct — per-claim attribution:

```latex
Measurement error can attenuate effect estimates \cite{Fong2021}.
The bias direction depends on whether errors are classical or
differential \cite{Egami2022, Grimmer2022}. Recent work shows
that LLM classifiers produce non-classical errors correlated with
document length and topic \cite{Keith2020}.
```

Acceptable — multiple sources for one claim:

```latex
Several studies have documented racial disparities in lending
outcomes across U.S. metropolitan areas
\cite{Ross2002, Munnell1996, Ladd1998}.
```

## Output rules

### Delivery

- Present the written paragraph directly.
- Do not narrate what you are about to do.
- Do not include planning text, meta commentary, or explanation as a substitute for the deliverable.

### Goal vs constraint

When a constraint is given, identify the positive goal it serves and satisfy the goal. Do not satisfy a constraint by omitting content.

## Examples

### Example 1: Literal placeholder target

User: “Write a paragraph at the text `INSERT TEXT HERE` explaining why this section matters.”

Do:
- Find `INSERT TEXT HERE` and write there.
- Output only the paragraph (no preamble).

Do not:
- Write somewhere else because a snippet was attached.

### Example 2: Line-range target

User: “In lines 40–55, add a short paragraph that motivates the next section.”

Do:
- Touch only lines 40–55.
- Keep prose plain; no bullets; avoid disallowed characters.

