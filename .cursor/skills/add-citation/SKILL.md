---
name: add-citation
description: Add one or more citations to refs.bib and optionally insert \cite{key} into draft_3.tex. Use when the user provides a citation (full or partial), a paper name, or a reference identifier and asks to add it, insert it, or return its key.
---

# Add Citation

## What this skill does

Given a citation (full metadata, paper name, or partial info), this skill:
1. Resolves the full bibliographic record if needed
2. Adds the entry to `docs/refs.bib`
3. Inserts `\cite{key}` into `docs/draft_3.tex` if a target location is given
4. Returns the key as `\cite{key}`

---

## Step 1: Parse the input

Extract from the user's message:

- **Citation data**: full text, paper name/shorthand, arXiv ID, DOI, or URL
- **Target location** (optional): file + line range where `\cite{key}` should be inserted
- **Number of citations**: handle each independently if multiple

---

## Step 2: Resolve the citation (if incomplete)

When the user provides a paper name, shorthand, or incomplete metadata, follow this lookup order **strictly — do not skip steps**:

1. **Search `docs/literatures.md`** — the project's curated literature notes
2. **Check the references folder** — read the untracked bib file directly even if not visible in the file tree (path pattern: `../references/literature_review/extracted_all_citations.bib` or similar paths noted in `literatures.md` entries)
3. **Web search** — use arXiv, DOI resolver, or publisher page to verify and retrieve full metadata
4. **If all three fail** — report that the citation cannot be resolved and ask the user to provide the full reference

**Do not**: search the draft `.tex` file for context clues. The draft is not a citation source.

---

## Step 3: Construct the BibTeX entry

**Key naming convention:**

| Case | Pattern | Example |
|---|---|---|
| Multi-author | `Lastname_etal{YYYY}` | `Bohannon_etal2007` |
| Single author | `Lastname{YYYY}` | `Rahm2011` |
| Software / tool | `ToolName{YYYY}` | `OpenRefine2025` |

**Entry type selection:**

- Conference paper → `@inproceedings`
- Journal article → `@article`
- arXiv preprint → `@misc` with `howpublished = {arXiv:XXXX.XXXXX}`
- Software / website → `@misc` with `howpublished = {\url{...}}` and `note = {Accessed: YYYY-MM-DD}`
- Book chapter → `@incollection`

Include: `author`, `title`, `year`, and the venue fields appropriate to the type. Do not invent field values.

---

## Step 4: Write to `docs/refs.bib`

Append the new entry to `docs/refs.bib`. Do not modify any existing entries.

---

## Step 5: Insert into `docs/draft_3.tex` (only if target location given)

- Insert `\cite{key}` at the specified location.
- Place it **before** the sentence-ending period if inline.
- Touch only the stated line range — no other edits.

If no target location is given, skip this step entirely.

---

## Step 6: Return the key

Always return the key for every entry added:

```
\cite{KeyName}
```

For multiple entries, return a table:

| Name | Key |
|---|---|
| AutoLabel | `\cite{Ming_etal2024}` |
| LLMAnno | `\cite{Haq_etal2025}` |

---

## Decision tree

```
User provides citation
│
├── Full metadata provided?
│   └── YES → go to Step 3
│   └── NO  → run Step 2 lookup (literatures.md → refs folder → web search)
│
├── Target .tex location given?
│   └── YES → Steps 3 → 4 → 5 → 6
│   └── NO  → Steps 3 → 4 → 6
```
