What is done:
1. stacking prompts: send DOC BY DOC, and UPDATE the SUMMARY. (done)
2. modify prompt: FOCUSED EXTRACTIVE, then SUMMARIZATION. (done)
    ie. is there info ABOUT X? RETURN the EXACT TEXT about X. give summary ITEM BY ITEM.
2.1 improve prompts (todo)
3. collapse categories.(done)
3.1 Post-processing pipeline (opr_7_post_processing.ipynb)

### Original Problems

1. **Monophyletic violations**: Parent and child categories at same level (e.g., "State agent" vs "Municipal police")
2. **Catch-all categories**: Residual categories like "Other", "No information" that absorb ambiguous cases
3. **Parenthetical suffixes**: LLM outputs "Municipal police" but human annotates "Municipal police (they depend on...)" → mismatch
4. **Separated Labels**: perp_tipo1/perp_tipo2, proced_contacto1/proced_contacto2 split across columns
5. **Semantic overlap**: Same concept with different wording (e.g., "Kidnapping", "Plagio", "Levantón")

### Solution: 4-Stage Post-Processing Pipeline

**STAGE 1: Normalize Values**
- NaN → ""
- Strip parenthetical suffixes: "Foo (bar explanation)" → "Foo"
- Handles LLM outputs that omit the parenthetical part

**STAGE 2: Merge Separated Labels**
- perp_tipo1 + perp_tipo2 → perp_tipo (list)
- proced_contacto1 + proced_contacto2 → proced_contacto (list)

**STAGE 3: Category Coarsening**
- Apply taxonomy.json `category_merging` mappings
- Consolidate fine-grained labels into coarser monophyletic categories
- e.g., All police types → "State-affiliated", All cartels → "Criminal-organization-affiliated"

**STAGE 4: Update Match Columns**
- Sort and deduplicate merged list columns
- Recalculate *_match columns based on post-processed values

### Results

| Label | Δ Accuracy | Δ F1 | Δ Kappa |
|-------|------------|------|---------|
| vic_grupo_social | +0.03 | +0.02 | +0.03 |
| captura_tipo | +0.09 | +0.13 | +0.01 |
| desenlace | +0.03 | +0.21 | +0.00 |
| desenlace_tipo | +0.04 | +0.03 | -0.00 |
| proced_sent_tipo | +0.01 | +0.08 | +0.02 |
| perp_tipo | -0.03 | -0.00 | +0.06 |
| proced_contacto | +0.03 | -0.04 | +0.02 |

**Conclusion**: Most labels improved after post-processing. Merged labels (perp_tipo, proced_contacto) show mixed results — the merge operation itself introduces complexity (comparing lists), and the original separation issue is not fully resolved by coarsening alone.


---

## captura_tipo / desenlace_tipo Classification Issues

### Original Categories (9)

1. "Places related to the victim (house, workplace, private property)"
2. "Economic, social, industrial, agricultural and service centers"
3. "Authorities (government offices, military facilities)"
4. "Educational and medical facilities"
5. "Places for free expression, association and gatherings"
6. "Unoccupied or barren public spaces"
7. "Means and routes of transport and places of connection"
8. "International and protected spaces"
9. "Special centers and barracks for detention"

### Problems

- **NON-EXCLUSIVE**: Same location can belong to multiple categories
- **DIMENSION MIXING**: Categories mix usage, ownership, and state
- **ANNOTATOR INTERPRETATION**: LLM and Human may choose different categories for same location, both valid

### Case 1

> The victim was an 8-year-old girl who was approached by an adult male while playing outside a bus route office in a residential area in Ciudad Juárez. The victim's mother contacted the Ciudad Juárez police department, and the victim's family subsequently conducted a search. The victim was found dead in an empty lot. The suspected adult male was arrested due to a prior record of sexual offenses and was sent to Topochico prison. The Nuevo León State Attorney General's Office is investigating the case, and there is currently no indication of civil society involvement.

**Location**: Bus route office in residential area

**Possible classifications**:
- LLM: Economic, social, industrial, agricultural and service centers (bus office is a service center)
- Human: Places related to the victim (victim lives in the residential area)

**Result**: Evaluated as incorrect → false negative. But both classifications are valid.

### Case 2

> The kidnapped children were part of a shelter for minors and were subjected to physical and psychological abuse, sexual abuse, and religious indoctrination. The children were kidnapped by their captors, who disguised and made them up to sell them. Julio César and Diana Lizeth were recovered by the authorities, but their sister Adriana Guadalupe remains missing. The Restored Christian Church and the cult of 'Los Perfectos' were involved in the kidnapping and their leaders, Jorge Erdely Graham and Sergio Humberto Canavati Ayub, were identified. The mothers of the kidnapped children contacted the authorities and the Subprocuraduría de Investigación Especializada en Delincuencia Organizada (Siedo) responded to the contacts.

**Location**: Shelter for minors

**Possible classifications**:
- LLM: Educational and medical facilities (shelter is an institutional care facility)
- Human: Places related to the victim (children lived there)

**Result**: Same issue - both classifications are valid, but evaluation marks as incorrect.

### Current Solution: Binary Classification

Due to severe taxonomy design issues, unable to achieve monophyletic standard through coarsening. Adopting binary classification:

```
→ Places related to the victim (as is)
→ Public and institutional spaces (Economic centers, Authorities, Educational/medical facilities, Free expression places, Unoccupied spaces, Transport routes, International spaces, Detention centers)
```

**Rationale**:
- "Places related to the victim" is the only category describing private/personal relationship
- Other 8 categories can all describe the same public location, merged to reduce false negatives

---

## vic_grupo_social Classification Issues

### Original Categories (9)

1. "Professionals (Entrepreneur, Engineer, Professor, Journalist, etc)"
2. "People that work in service industries (taxi driver, salesman, etc)"
3. "Civil servants (Police, mayor, public worker, etc)"
4. "Belonging to some sexual identity group (LGBTQ)"
5. "People associated with politics"
6. "Activists (political activist, human rights, etc)"
7. "Organized crime"
8. "Students"
9. "Land Worker"

### Problems

- **NON-EXCLUSIVE**: A person can belong to multiple categories (LGBTQ engineer, journalist activist)
- **DIMENSION MIXING**: Categories mix occupation, political role, identity, and affiliation
- **ANNOTATOR INTERPRETATION**: Mayor is both "Civil servants" and "People associated with politics"

### Analysis: Binary Thinking

**Split 1: Student vs Non-student**
- Students - clearly separable

**Split 2 (Non-students): By employer/affiliation**
- Government: Civil servants
- Non-government: Professionals, Service workers, Land Worker
- Illegal organization: Organized crime

**Problematic categories:**
- People associated with politics: Could be government, non-government, or student
- Activists: Cross-cuts all categories
- LGBTQ: Identity dimension, unrelated to occupation

### Current Solution: Partial Coarsening

Unable to achieve full monophyletic standard. Coarsen where possible, keep cross-dimensional categories as-is:

```
→ Students (as is)
→ Civil servants (Civil servants, People associated with politics)
→ Professionals (Professionals, People that work in service industries, Land Worker)
→ Organized crime (as is)
→ Activists (as is)
→ LGBTQ (as is)
```

**Rationale**:
- Students, Organized crime: clearly separable, keep as-is
- Civil servants + People associated with politics: overlapping (mayor is both), merge
- Professionals + Service workers + Land Worker: all non-government workers, merge
- Activists, LGBTQ: cross-dimensional, cannot merge without losing meaning, keep as-is
