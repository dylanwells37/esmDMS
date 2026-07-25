# Count-bearing DMS datasets — provenance & processing notes

This folder holds the **five** assembled DMS datasets that carry a per-timepoint
**count or frequency** layer (in addition to a functional score and ClinVar annotation).
One CSV per dataset, **one row per unique protein sequence**.

| File | Gene / assay | Measurement layer | Rows |
|---|---|---|--:|
| `MV_BRCA1_Findlay_2018.csv` | BRCA1, saturation genome editing (SGE) | raw **counts** (8 cols) | 1956 |
| `MV_MSH2_Jia_2020.csv` | MSH2, HAP1 LOF selection | raw **counts** (10 cols) | 17746 |
| `MV_BRCA2_Huang_2025.csv` | BRCA2, SGE | raw **counts** (18 cols) | 4333 |
| `MV_VHL_Buckley_2024.csv` | VHL, SGE | raw **counts** (13 cols) | 1087 |
| `MV_TP53_Kotler_2018.csv` | TP53 (p53 DBD), growth/fitness | **frequencies** (24 cols) | 3158 |

Note: four are raw counts; **TP53 is frequencies**, not counts.

---

## Column schema (all files)

```
mutant                            protein change, 1-based on the canonical isoform (e.g. "F1821L"; "*" = stop)
mutated_sequence                  full mutated protein sequence (unique per row)
has_clinvar                       true/false
clinvar_significance              free-text ClinVar germline description
clinvar_significance_normalized   pathogenic | benign | uncertain significance | other
clinvar_review_status             ClinVar review status (star level)
clinvar_conditions                condition(s) / record title
clinvar_variation_ids             pipe-joined ClinVar Variation IDs
count__<...>  or  frequency__<...>  per-timepoint/replicate measurement columns (see each dataset)
functional_score                  assay score (see per-dataset direction note; may be empty — see BRCA2 Huang)
```

---

## Processing applied to every dataset

1. **Wild-type-identical sequences removed.** Only variants whose translated protein
   sequence differs from wild-type are kept (missense + nonsense). Synonymous, intronic,
   and splice-silent variants — anything that leaves the protein sequence unchanged — were
   dropped upstream.
2. **Collapsed to one row per unique protein sequence.** Where the source data had several
   records producing the *same* protein sequence, they were pooled:
   - **Counts / frequencies:** **summed** across the pooled records (depth-weighted pooling).
   - **functional_score:** **mean**, except **TP53 = median** (see below).
   - **ClinVar:** a single representative record is chosen — among the pooled records
     carrying a kept label (pathogenic / benign / uncertain), the one with the **highest
     review-star**, ties broken by severity (pathogenic > benign > uncertain). "other"
     (conflicting / not-provided / risk-factor) is treated as unlabeled. `clinvar_variation_ids`
     keeps the union of all underlying IDs for provenance.
3. After collapse, `mutated_sequence` is **unique** in every file (verified), so joining to
   external sequence-level annotations is one-to-one.

Rows before → after collapse: **BRCA1** 2224→1956, **BRCA2 Huang** 4886→4333, **VHL**
1190→1087, **TP53** 4413→3158, **MSH2** 17746→17746 (no duplicates). For the SGE datasets
(BRCA1, BRCA2 Huang, VHL) the "before" count is nucleotide variants; multiple nucleotide
variants encoding one protein change are pooled. TP53 pools repeated sub-library
measurements of one protein change (see its note).

---

## ClinVar annotation — single same-day snapshot

All five datasets were annotated in **one pass on 2026-07-15** (recorded in
`CLINVAR_SNAPSHOT.json`) against ClinVar via direct NCBI E-utilities, **per variant**, so
they share one consistent snapshot and one method. The **query key differs by assay type**,
matching what ClinVar can actually resolve:

- **SGE datasets → coding HGVS** (exact nucleotide match). BRCA1 `NM_007294.3/.4:c.`; VHL
  `ENST00000256474.3:c.` → `NM_000551.4`; BRCA2 Huang `ENST00000380152.8:c.` → `NM_000059.4`
  (the BRCA2 coding HGVS is derived from the GEO codon columns and validated 1:1 against
  MaveDB's own `hgvs_nt`). Each nucleotide variant is queried, then the representative label
  is chosen on collapse.
- **codon-DMS datasets → protein HGVS.** MSH2 `NP_000242.1:p.`; TP53 `NP_000537.3:p.` (built
  from the mutant). These studies record no nucleotide identity, and coding-exact matching
  would undercount (library codon ≠ patient codon), so protein HGVS is the correct key.

Normalized labels are `pathogenic | benign | uncertain significance | other`, where "other"
(ClinVar "conflicting", "not provided", etc.) is treated as unlabeled downstream.

**Coverage (post-collapse, one row per protein change):**

| Dataset | Query key | has_clinvar | P / B / VUS |
|---|---|--:|--:|
| BRCA1 Findlay | coding HGVS | 1956 / 1956 | 281 / 107 / 363 |
| BRCA2 Huang | coding HGVS | 2152 / 4333 | 265 / 210 / 976 |
| VHL Buckley | coding HGVS | 633 / 1087 | 189 / 21 / 304 |
| MSH2 Jia | protein HGVS | 3133 / 17746 | 142 / 285 / 1636 |
| TP53 Kotler | protein HGVS | 860 / 3158 | 256 / 42 / 315 |

---

## Per-dataset provenance

All MaveDB score sets were pulled from the MaveDB API (`https://api.mavedb.org/api/v1`).
"Accessed" = download date of the local raw artifact (file timestamp).

### MV_BRCA1_Findlay_2018 — BRCA1 SGE
- **Source:** MaveDB `urn:mavedb:00000097-0-2` (Findlay et al. 2018, BRCA1 saturation genome editing in HAP1). Counts from the MaveDB **counts.csv** of that score set.
- **Accessed:** 2026-06-25.
- **Counts (8):** `count_day5_rep1/rep2`, `count_day11_rep1/rep2`, `count_library`, `count_negative_control`, `count_rna_rep1/rep2`.
- **Score:** SGE normalized function score — **higher = more functional** (less damaging).
- **ClinVar:** annotated by exact coding HGVS (`NM_007294.3/.4`, `c.` level); all 1956 rows carry a ClinVar record.
- **Collapse note:** 238 protein changes arose from >1 nucleotide variant (codon-degenerate missense); their counts were **summed** and scores averaged.

### MV_MSH2_Jia_2020 — MSH2 loss-of-function
- **Source:** MaveDB `urn:mavedb:00000050-a-1` (Jia et al., MSH2 variant function in HAP1). Counts from **GEO GSE162130**, file `GSE162130_MSH2_HAP1_raw_count.tsv.gz`.
- **Accessed:** MaveDB 2026-06-25; GEO count file 2026-06-25.
- **Counts (10):** replicates R1–R3 across conditions `D_P0` (day0/plasmid), `DB_P2`, `DB6_P2`, plus `WT_plasmid`.
- **Score:** LOF score — **positive = loss-of-function** (MaveDB convention; note this is the opposite polarity of the BRCA1/SGE "higher = functional" scores).
- **functional_score populated for 16749 / 17746 rows** (997 protein changes have counts but no score).
- **Collapse note:** no duplicate protein sequences; count layer unchanged.

### MV_BRCA2_Huang_2025 — BRCA2 SGE (counts only)
- **Source:** MaveDB `urn:mavedb:00001225-a-1` (Huang et al. 2025, BRCA2 SGE). Counts from **GEO GSE270424**, file `GSE270424_combined.raw.tsv.gz`.
- **Accessed:** MaveDB 2026-06-25; GEO count file 2026-06-25.
- **Counts (18):** replicates R1–R6 × `{lib, D5, D14}`.
- **Score:** the MaveDB score for this set is **not protein-joinable**, so `functional_score` is **empty for all rows** — this dataset provides **counts only**. Derive a score from the counts (e.g. D14 vs library log-enrichment) if one is needed.
- **ClinVar:** coding HGVS derived from the GEO codon columns (`ENST00000380152.8:c.` → `NM_000059.4`), queried per nucleotide variant; 2152 / 4333 protein changes carry a record.
- **Collapse note:** the build now keeps one row per nucleotide variant (4886 rows) so coding-HGVS ClinVar can match; 553 protein changes are encoded by >1 nucleotide variant and are pooled (counts summed) back to 4333 rows. Count totals are unchanged from source.

### MV_VHL_Buckley_2024 — VHL SGE
- **Source:** MaveDB `urn:mavedb:00000675-a-1` (Buckley et al. 2024, VHL SGE). Counts from the MaveDB **counts.csv** of that score set.
- **Accessed:** MaveDB scores/metadata 2026-06-29; counts.csv 2026-07-06.
- **Counts (13):** two selection arms `tHDR_*` and `rLD2_tHDR_*`, each with `lib / pre / post / post2 / rna / rna2` (and `tHDR_neg`).
- **Score:** VHL SGE function score. Where >1 nucleotide variant encodes one protein change, the collapsed score is their **mean**.
- **ClinVar:** exact coding HGVS (`ENST00000256474.3:c.` → `NM_000551.4`), queried per nucleotide variant; 633 / 1087 protein changes carry a record.
- **Collapse note:** the build now keeps one row per nucleotide variant (1190 rows); 103 protein changes are encoded by >1 nucleotide variant and are pooled (counts summed, score averaged) back to 1087 rows. Count totals are unchanged from source.

### MV_TP53_Kotler_2018 — p53 DNA-binding domain (frequencies)
- **Source:** MaveDB `urn:mavedb:00000059-a-1` (Kotler et al. 2018, p53 DBD relative-fitness screen in H1299). **Frequencies** from **GEO GSE115072** (H1299 "TC" time-course), files `GSE115072_H1299_TC_DBD{A,B,C,D}[_rep2].csv.gz`; variant identities from `GSE115072_H1299_RFS_sequence_variants.xlsx`.
- **Accessed:** MaveDB 2026-06-25; GEO files 2026-06-25.
- **Frequencies (24):** four synthesized sub-libraries **DBDA, DBDB, DBDC, DBDD** (DBDA/DBDB also have a `rep2`) × four timepoints **2d, 6d, 9d, 14d**.
- **Position offset:** MaveDB numbering carries a `protein_position_offset` of 101 (the score set covers the DBD); `mutant` here is on the full-length p53 numbering.
- **Score:** relative fitness score (Kotler growth screen), as published in MaveDB.
- **Collapse notes (important, TP53-specific):**
  - `hgvs_nt` is empty for this study — variants are identified **only at the protein level**. The 4413 source records collapse to 3158 protein sequences because **652 protein changes were measured independently in more than one sub-library** (up to 9 records for one variant, e.g. spiked-in nonsense controls).
  - These independent measurements often **disagree** (of the 652: ~176 differ by >1.0 on a scale with sd≈1.4; ~128 flip sign). The collapsed `functional_score` is therefore the **median** across a protein change's records (not the mean), to be robust to an outlier sub-library. For the 277 variants measured exactly twice, median = mean unavoidably.
  - Frequency columns are **per-sub-library**: a variant contributes to only the columns of the sub-library it appeared in, so pooling fills different columns rather than summing within one.
  - ClinVar coverage is lower here (860 / 3158 rows), consistent with the DBD-only footprint.

---

## Access-date summary

| Dataset | MaveDB score set | External count/frequency source | Accessed | ClinVar snapshot |
|---|---|---|---|---|
| BRCA1 Findlay 2018 | urn:mavedb:00000097-0-2 | MaveDB counts.csv | 2026-06-25 | 2026-07-15 |
| MSH2 Jia 2020 | urn:mavedb:00000050-a-1 | GEO GSE162130 | 2026-06-25 | 2026-07-15 |
| BRCA2 Huang 2025 | urn:mavedb:00001225-a-1 | GEO GSE270424 | 2026-06-25 | 2026-07-15 |
| VHL Buckley 2024 | urn:mavedb:00000675-a-1 | MaveDB counts.csv | 2026-06-29 / 2026-07-06 | 2026-07-15 |
| TP53 Kotler 2018 | urn:mavedb:00000059-a-1 | GEO GSE115072 | 2026-06-25 | 2026-07-15 |

All ClinVar annotations were queried in a single pass on **2026-07-15** via NCBI E-utilities
(per-variant), so the five datasets share one consistent snapshot. See the "ClinVar
annotation" section above for the per-dataset query key.

_Document generated 2026-07-15._
