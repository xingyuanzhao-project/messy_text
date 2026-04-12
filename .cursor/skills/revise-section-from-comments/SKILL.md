---
name: revise-based-on-comments
description: Revise an existing draft section in response to reviewer, PI, editor, or inline comments. Use when the user asks to modify, consolidate, expand, support, or rewrite a specific section based on comments, feedback, marginal notes, or embedded comment blocks while preserving scope, section function, and the requested transformation level.
---

# Revise Based on Comments

## Purpose

Revise an existing section of a draft from comments without drifting from the target area, the section's function, or the requested transformation level.

## Required extractions

Before revising, extract and record all of the following:

- **Target area**: The exact lines, span, section, or literal placeholder that may change.
- **Comment directives**: Every explicit instruction in the comment, transcript, or inline note.
- **Section function**: What this section does in the document's argument.
- **Underlying goal**: The problem the comment is trying to solve.
- **Component profile**: The selected comment purpose, output unit, evidence object, drafting action, freedom level, and binding anchors from Step 3 (leave blank until then).
- **Transformation level**: Reorder existing prose, add support, fill fields, or fully rewrite.
- **Binding references**: Any file, note, transcript, figure, table, source pool, or dataset the comment names.
- **Output unit**: The smallest unit the final result must make visible (determined in Step 3; leave blank until then).
- **Scope and non-goals**: What the comment is not asking for.
- **Footprint budget**: The length and density allowed by neighboring paragraphs or field structure.
- **Established vocabulary**: Domain terms, recurring local wording, and concept labels already used in the target and neighboring text. Reuse them by default. Replace only if equivalence is directly verified.
- **Formatting constraints**: Local or project-level prose rules such as plain prose, no bullets, or banned characters and constructions. These constrain form, not substance.

If any binding reference has not been read, stop and read it before revising.

## Phase boundary

Interpretation requests are not edit permission.

- If the current turn asks to analyze, rank, map, compare, or explain the comment, do that first and stop.
- Revise only when the user asks to modify, rewrite, insert, add, consolidate, expand, or similar.

## Core rule

Turn analysis results into hard constraints before editing.

Do not let later drafting overwrite earlier conclusions about:

- section function
- hierarchy
- output unit
- evidence source
- scope boundaries

A correct analysis followed by unconstrained drafting is still failure.

## Step 1: Read the local context

- Read the target area.
- Read the paragraphs immediately before and after it.
- Determine what the target section does in the argument.
- Treat neighboring content as exclusions. Do not re-cover what they already establish.
- Infer the footprint budget from the neighboring paragraphs or field structure.
- If the target includes a literal placeholder, comment block, or draft slot, treat it as a binding anchor.

## Step 2: Parse the comment

Extract four things separately:

1. **Target**: what span must change
2. **Problem**: what is wrong with the current text
3. **Goal**: what the revised text must accomplish
4. **Non-goals**: what should not change

Do not stop at surface wording. A comment about a heading may really be about structure. A comment about broadness may really be about claim level, not more coverage.

## Step 3: Classify the revision components

Do not classify from keywords alone. First classify the task at the component level. Then turn the classified components into a drafting profile.

Record the following dimensions before drafting:

1. **Comment purpose**: what problem the revision must solve. One or more may apply, such as repairing hierarchy, narrowing a claim, adding support, adding examples, redistributing citations, filling a missing slot, grounding a field, describing a result, interpreting a result, or aligning terminology.
2. **Output unit**: the smallest visible object the revision must produce. Examples include a sentence role, clause revision, paragraph span, example-problem-citation pair, field entry, placeholder fill, result-description sentence, interpretation sentence, or bridge sentence.
3. **Evidence object**: the authoritative evidence source for that output unit. Examples include existing local prose, neighboring paragraphs, the comment block, a named source pool, a direct source object, a figure or table, result output, a transcript, or a field schema.
4. **Drafting action**: the operation required to complete the task. Examples include reorder, compress, preserve hierarchy, reuse wording, retrieve, select, integrate, classify, describe, interpret, fill a slot, relabel, or redistribute citations.
5. **Freedom level**: how much new phrasing or restructuring is allowed. Default to low. Preserve established wording, domain vocabulary, sentence roles, and local syntax unless the comment clearly allows broader rewriting.
6. **Binding anchors**: any literal slot, placeholder fragment, field name, named resource, user template, domain vocabulary, explicit non-goal, or footprint budget that must constrain the draft.

## Step 4: Freeze the hard constraints from the component profile

Turn the classified components into non-negotiable drafting constraints before writing.

Freeze all of the following:

- the final output unit or units
- the evidence object for each output unit
- the drafting action for each output unit
- the freedom level
- the binding anchors
- the scope boundaries and non-goals

Do not let later drafting overwrite these decisions. A correct component analysis followed by unconstrained drafting is still failure.

### Constrain by output unit

If the output unit is a sentence role, clause revision, or paragraph span:

- Treat the task as transformation, not free generation.
- Build the sentence roles or slots first. Then map material into them.
- By default, keep established domain vocabulary and recurring local wording.
- Reuse existing sentences, sub-sentences, and sentence roles unless the comment clearly asks for broader rewriting.
- Verify exact equivalence before replacing a term. Do not swap in a general English synonym for convenience or smoothness.
- If the user says `not paraphrase`, `use old wording`, or equivalent, keep freedom low and do not smooth by synonym substitution.
- Preserve hierarchy explicitly. Do not promote support points into headline claims.
- Interpret broadness as level of claim unless the comment clearly asks for more coverage.

Do not:

- treat `refine` as permission for a fresh rewrite
- replace structure with smoother wording
- expand one sentence by listing every supporting condition

If the output unit is a support sentence, example-problem-citation pair, or inserted citation-bearing clause:

- Define the retrieval object before searching or drafting.
- Separate `document/source type`, `domain`, `messiness feature`, and `exclusion`.
- Use domain as metadata unless the comment explicitly asks to organize by domain.
- Exclude methods or tool papers unless the task explicitly asks for them.
- Keep the support visible at the unit level.
- If the user gives a model pair or comparison pair, reuse that structure.

Do not:

- search for `published research` when the real target is source or document situations
- group results by domain when the task is about source type
- collapse examples into a taxonomy sentence that hides the pairs

If the output unit is a field entry, label, or short source-grounded description:

- Distinguish the paper, the cited source inside the paper, and the source object itself.
- Access the named source object directly before classifying.
- If direct access fails, mark the point as unverified instead of guessing.
- Treat category lists in the prompt as examples unless the user makes them hard constraints.
- Write short descriptions of what the object is, not how the paper analyzed it.

Do not:

- classify an unseen object
- infer a field from a paper's discussion alone
- use a secondary paper as a substitute for the source object

If the output unit is a placeholder fill, result-description sentence, interpretation sentence, or bridge sentence:

- Use the surrounding fragment or local paragraph role as a binding anchor.
- Continue the existing sentence or local paragraph movement rather than restarting it in fresh language.
- Preserve local syntax and rhetorical fit.
- Keep the concept labels already established around the slot unless equivalence is directly verified.
- Separate description from interpretation unless the comment explicitly combines them.
- Base result claims on the actual figure, table, or result output.

### Constrain by evidence object

If the evidence object is existing local prose or neighboring paragraphs:

- Treat neighboring content as exclusions.
- Do not re-cover what adjacent paragraphs already establish.

If the evidence object is a comment block or transcript:

- Treat it as the authoritative record of what was asked, rejected, or required.
- Do not infer past instructions from the current file state.

If the evidence object is a named source pool:

- Freeze the inclusion rule before searching.
- Select evidence for the local rhetorical job, not for generic completeness.

If the evidence object is a direct source object:

- Let the source object override secondary discussion.
- Ground claims in the object itself before using any paper-level interpretation.

If the evidence object is a figure, table, or result output:

- Describe what is shown before interpreting what it means.
- Do not infer values or comparisons that are not visible or directly reported.

### Constrain by drafting action

If the drafting action includes `reorder`, `compress`, `preserve hierarchy`, or `reuse wording`:

- Make the smallest revision that satisfies the new structure.
- Preserve the distinction between headline claims and support.

If the drafting action includes `retrieve`, `select`, or `integrate`:

- Choose only the strongest fits for the local rhetorical job.
- Keep each selected support unit visible in the prose.

If the drafting action includes `classify`, `describe`, or `relabel`:

- Write only what direct evidence supports.
- Prefer concrete short descriptions over abstract category language.

If the drafting action includes `fill a slot`:

- Continue the existing sentence or paragraph naturally.
- Preserve existing syntax, rhetorical fit, and established concept labels unless equivalence is directly verified.
- Do not restart the section in a new shape unless the comment explicitly asks for restructuring.

If the drafting action includes `redistribute citations`:

- Place each citation at the claim it supports.
- Do not bundle citations after multiple distinct claims.

If the drafting action includes `align terminology`:

- By default keep the established local term.
- Replace a term only after reading the local context and verifying exact equivalence.
- Maintain lexical continuity across neighboring sentences and paragraphs. Do not rotate among near-synonyms for the same phenomenon.
- Do not replace a domain term with a general synonym just to smooth the prose.

### Constrain by formatting continuity

- Default to plain prose and follow the local or project writing guidance already in force.
- If the local guidance bans bullets, parentheses, quotation marks, markdown emphasis, em dashes, or similar constructions, revise into compliant prose without deleting substance.
- If local guidance additionally bans semicolons, colons as logical connectors, ampersands, or similar punctuation patterns, rewrite the sentence so the relationship is carried by words.
- Do not use format cleanup as permission to drop support, collapse distinctions, or remove citations.

### Resolve conflicts

When components pull in different directions, use this priority order:

1. Binding anchors and explicit non-goals
2. Evidence constraints
3. Output unit requirements
4. Freedom level
5. Stylistic smoothing

If multiple output units are required, preserve each unit's visibility instead of flattening them into generic summary prose.

## Step 5: Read the binding references

When a comment names a resource, that is a command.

Read:

- the target file and neighboring context
- embedded comment blocks
- named notes or source pools
- cited transcripts if the task is about a prior exchange
- source objects such as dataset pages, files, figures, tables, or code when the revision depends on them

Use the conversation or transcript as the authoritative evidence source when the task is about what was instructed, rejected, or previously written. Do not infer past instructions from the current file state.

## Step 6: Draft the revision from the component profile

After reading the binding references, draft from the frozen component profile rather than from habit.

1. Restate the drafting plan in working form:
   - what output unit or units must appear
   - what evidence object supports each unit
   - what drafting action applies to each unit
   - what anchors, non-goals, and footprint limits must constrain the draft

2. Draft the smallest required unit first, not the whole paragraph by instinct.

3. If the task includes `reorder`, `compress`, `preserve hierarchy`, or `reuse wording`:
   - write the sentence roles or slots first
   - assign existing material to each role
   - remove scaffolding that repeats the same move
   - write the smallest revision that satisfies the new hierarchy

4. If the task includes `retrieve`, `select`, or `integrate`:
   - build or read the source pool
   - select only the strongest fits for the local rhetorical job
   - write each support unit so the reader can see the subject, the problem, and the citation
   - use 1 to 3 sentences if that improves visibility
   - do not force one long sentence when separate units are clearer

5. If the task includes `classify`, `describe`, or `relabel`:
   - fill each field or description from directly supported evidence only
   - keep descriptions concrete and short
   - use a new label when the provided examples do not fit
   - leave out claims you cannot verify

6. If the task includes `fill a slot`, `describe a result`, `interpret a result`, or `build a bridge sentence`:
   - use the surrounding fragment or local paragraph role as the frame
   - preserve local syntax, pacing, and rhetorical fit
   - if both description and interpretation are needed, make the descriptive unit visible before the interpretive one
   - do not infer unseen values, trends, or comparisons

7. When multiple actions apply, compose them in the order that protects the output unit:
   - structure before smoothing
   - evidence selection before integration
   - direct grounding before classification
   - description before interpretation

8. Keep the revision inside the footprint budget. Multiple comment items do not imply multiple paragraphs. Fit the result to the local slot unless the comment explicitly asks for expansion.

9. Prefer local continuity over freshness:
   - reuse established domain vocabulary
   - follow local formatting conventions
   - preserve the section function
   - do not broaden beyond the comment's actual goal

10. Stop when the requested unit is complete. Do not add adjacent improvements, extra examples, or broader rewrites that were not asked for.

## Correction handling

Treat corrections as conceptual, not token-level.

If the user rejects a formulation:

- identify what concept or role was rejected
- remove that concept, not just that exact wording
- state the real cause of the failure if asked
- do not invent a new parsing to defend the old move

Examples:

- If the user rejects broadness as a list of conditions, do not replace it with a different long list.
- If the user rejects category-level prose, do not switch to a near-synonym that still hides the example-problem pairs.
- If the user says the task is not paraphrase, do not produce a smoother paraphrase with some old words copied over.

## Validation loop

Before finishing, check:

1. Did the revision solve the comment's actual problem?
2. Did it stay inside the target area?
3. Does it match the section function?
4. Does the output unit match the task?
5. If the task required hierarchy-sensitive revision, did hierarchy survive?
6. If the task required support units or example pairs, are they visible and is domain kept separate from source type?
7. If the task required direct source grounding, is every grounded field or claim tied to a directly read source object?
8. Did any examples list or schema turn into free-form mush during drafting?
9. Did you preserve established vocabulary, lexical continuity, and local syntax where the task required transformation or slot filling?
10. Did you preserve substance while obeying local formatting conventions and any banned-character rules?
11. If the user gave an explicit slot, template, or example, did you follow it literally?

If any check fails, revise before presenting.

## Output rules

- Deliver the scoped edit or the requested revised text.
- Do not substitute analysis for revision once the task is actionable.
- If a required source object is unreadable or unavailable, report that gap plainly instead of filling with inference.

## Examples

### Example 1: Consolidation comment

Comment:

`Sentence 1 should make point 1 the umbrella contribution. Sentence 2 should keep the better-use-of-LLMs claim. Task is not paraphrase.`

Do:

- preserve the sentence roles
- reuse existing sentence material
- keep support material under the umbrella claim

Do not:

- write a fresh summary paragraph
- turn support points into separate headline contributions

### Example 2: Reviewer asks for other messy cases

Comment:

`Give a list of messy and related other cases and citations here.`

Do:

- search for source or document situations with the same structural problem
- group by source type if needed
- write visible example-problem-citation pairs

Do not:

- search for methods papers by default
- mix domain names and document types in one category list
- compress everything into one taxonomy sentence that hides the pairs

### Example 3: Fill data fields from a paper

Task:

`Read the source and fill data description, data type, and data problem.`

Do:

- access the dataset page or files before classifying
- describe what the data are
- use a new label when the provided examples do not fit

Do not:

- classify from a secondary paper alone
- describe the paper's analysis instead of the data object
- force the object into an example category without evidence
