---
name: reformat-task
description: Reformat existing text within a specified target area while preserving substance. Use when the user asks to reformat, convert bullets to prose, remove markdown/special characters, or otherwise change formatting without rewriting content outside the target. Enforce strict boundaries: modify only the target region and do not delete information to satisfy formatting constraints.
---

# Reformat Task

## Purpose

Transform content into the requested format while preserving the original substance, within a strictly bounded target region.

## Required extractions (do before editing)

- **Target area**: Line range or named section to reformat.
- **Requested format change**: What formatting must change (for example: bullets to paragraph, remove special characters).
- **Boundaries**: Confirm nothing outside the target is in scope.

## Scope and boundaries

- Touch only the stated target area.
- Do not reorganize or rewrite content outside the target.

## Definition of modification

- Preserve the substance of the original text while applying the format change.
- Do not delete content as a way to remove unwanted formatting.
- If the user pushes back, infer precisely what you got wrong about where the formatting problem is, then correct only that.

## Formatting constraints

### Characters

- Scan the target text for disallowed characters (such as `()`, `"`, `**`, and similar markdown formatting).
- Remove each disallowed character while keeping the surrounding prose intact and natural.
- Do not introduce new disallowed characters while rewriting.

### Structure

If the target contains bullets:

- Extract the substance from each bullet point.
- Construct a single flowing paragraph that integrates all extracted substance.
- Replace the bullet list with the paragraph.
- Do not drop any bullet substance to eliminate bullet formatting.

## Goal vs constraint

When a constraint is given, identify the positive goal it serves and satisfy the goal. A constraint against bullets must be satisfied by prose conversion, not deletion of bullet content.

## Examples

### Example 1: Bullets → paragraph (no substance loss)

User: “Reformat lines 12–25 to remove bullets and make it a paragraph.”

Do:
- Touch only lines 12–25.
- Convert each bullet’s meaning into a single coherent paragraph.
- Keep all points.

Do not:
- Delete bullets to “solve” the bullet constraint.

### Example 2: Remove disallowed characters

User: “In the next paragraph, remove parentheses and markdown bold, but keep the meaning.”

Do:
- Remove `()` and `**` while keeping the prose intact.

