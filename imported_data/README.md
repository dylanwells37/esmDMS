# Count-bearing DMS datasets — handoff README

Read this first. It tells you what the files are, how to use them, and the gotchas that
will bite you if you don't. For where the data came from, access dates, and every
assumption, see **`DATASETS.md`** in this folder. ClinVar snapshot metadata is in
**`CLINVAR_SNAPSHOT.json`**.

## What's here

Five deep-mutational-scanning (DMS) datasets, one CSV each. Every row is a **protein
variant**, and each file is **one row per unique protein sequence**.

| File | Gene | Measurement layer | functional_score | ClinVar key |
|---|---|---|---|---|
| `MV_BRCA1_Findlay_2018.csv` | BRCA1 | raw **counts** (8) | yes | coding HGVS |
| `MV_MSH2_Jia_2020.csv` | MSH2 | raw **counts** (10) | yes (16749/17746) | protein HGVS |
| `MV_BRCA2_Huang_2025.csv` | BRCA2 | raw **counts** (18) | **NO (empty)** | coding HGVS |
| `MV_VHL_Buckley_2024.csv` | VHL | raw **counts** (13) | yes | coding HGVS |
| `MV_TP53_Kotler_2018.csv` | TP53 | **frequencies** (24) | yes | protein HGVS |

## Column schema (all files)

```
mutant                            protein change, 1-based on the canonical isoform (e.g. "F1821L"; "*" = stop)
mutated_sequence                  FULL mutated protein sequence  <-- unique per row; use as the join key
has_clinvar                       "true" / "false"
clinvar_significance              free-text ClinVar germline description (e.g. "Likely pathogenic")
clinvar_significance_normalized   pathogenic | benign | uncertain significance | other
clinvar_review_status             ClinVar review status (star level)
clinvar_conditions                associated condition(s)
clinvar_variation_ids             pipe-joined ClinVar Variation IDs (union of the variant's underlying records)
count__<...>  OR  frequency__<...>   per-timepoint/replicate measurements (prefix tells you which; see gotcha 2)
functional_score                  assay score (last column; see gotchas 3 & 4)
```

## How to use

```python
import pandas as pd
df = pd.read_csv("MV_BRCA1_Findlay_2018.csv")

# join to embeddings keyed by protein sequence (1:1 — see gotcha 1)
merged = df.merge(emb_df, left_on="mutated_sequence", right_on="ProteinSequence", how="left")

# keep only clinically-labeled rows for a supervised task
labeled = df[df.clinvar_significance_normalized.isin(
    ["pathogenic", "benign", "uncertain significance"])]

# measurement columns
count_cols = [c for c in df.columns if c.startswith(("count__", "frequency__"))]
```

## Gotchas — read these

1. **`mutated_sequence` is the join key and is unique in every file.** Variants that share
   a protein sequence were already pooled to one row, so a `how="left"` merge on
   `mutated_sequence` gives exactly one row per variant. Do not join on `mutant` — that's a
   display label, not guaranteed to disambiguate across isoform conventions.

2. **TP53 is FREQUENCIES, the other four are raw COUNTS.** Check the column prefix
   (`frequency__` vs `count__`) before doing anything count-like (log-enrichment, depth
   normalization). Don't mix the two.

3. **BRCA2 Huang has NO functional_score** (empty for all rows) — its MaveDB score wasn't
   protein-joinable. It ships as **counts only**. If you need a score, derive it from the
   counts (e.g. D14-vs-library log-enrichment across R1–R6).

4. **Score polarity is NOT consistent across datasets.** Higher can mean *more* or *less*
   functional depending on the assay:
   - BRCA1 / VHL (SGE): **higher = more functional** (less damaging).
   - MSH2: **positive = loss-of-function** (opposite polarity).
   - TP53: Kotler relative-fitness score (see `DATASETS.md`).
   - Never pool or threshold scores across datasets without aligning direction first.

5. **`clinvar_significance_normalized` is your label column.** Values `pathogenic`,
   `benign`, `uncertain significance` are the usable classes; **`other`** means ClinVar had a
   record but it was conflicting / not-provided / risk-factor — treat as *unlabeled*, not as
   a fourth class. `has_clinvar == "true"` with `normalized == "other"` is common and expected.

6. **All ClinVar was queried on one snapshot: 2026-07-15** (see `CLINVAR_SNAPSHOT.json`), via
   NCBI E-utilities, per variant. The query key differs by assay type — coding HGVS for the
   SGE sets (BRCA1, BRCA2 Huang, VHL), protein HGVS for the codon-DMS sets (MSH2, TP53) —
   because that's what ClinVar can resolve for each. This is deliberate; don't "unify" it.

7. **Counts were summed, scores averaged, when pooling duplicate protein sequences.** TP53
   uses the **median** score (its duplicates are repeated sub-library measurements that often
   disagree). Count *totals* are unchanged from source. Details in `DATASETS.md`.

## Row counts (after collapse)

BRCA1 1956 · MSH2 17746 · BRCA2 Huang 4333 · VHL 1087 · TP53 3158.

## ClinVar coverage (has_clinvar / total ; pathogenic / benign / VUS)

- BRCA1: 1956 / 1956 ; 281 / 107 / 363
- BRCA2 Huang: 2152 / 4333 ; 265 / 210 / 976
- VHL: 633 / 1087 ; 189 / 21 / 304
- MSH2: 3133 / 17746 ; 142 / 285 / 1636
- TP53: 860 / 3158 ; 256 / 42 / 315
