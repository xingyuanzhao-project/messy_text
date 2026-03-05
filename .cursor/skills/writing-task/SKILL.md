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

