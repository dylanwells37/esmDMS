# ESM-DMS architecture and analysis flow

This document describes the complete supported workflow, the data contracts
between stages, the numerical methods, and the repository changes that produced
the current structure. There is one data schema, one feature-artifact schema,
one CLI, and one analysis notebook.

## 1. Repository structure

```text
esmdms/
  schema.py       canonical Dataset and FeatureArtifact contracts
  import_data.py  imported_data CSV to canonical Dataset conversion
  features.py     protein embeddings, masked-marginal LLR, and SAE training
  inference.py    feature-basis popDMS inference and LLR-prior sweeps
  metrics.py      enrichment, functional score, ClinVar AUC, and Spearman metrics
  workflow.py     configuration-driven multi-dataset analysis
  cli.py          all command-line entry points

imported_data/                    supplied count/frequency-bearing source tables
examples/brca1_100_demo.py        resumable 100-variant end-to-end demonstration
multi_dataset_llr_prior.ipynb     interactive view of the same workflow API
analysis_config.example.json      multi-dataset analysis configuration
cluster/slurm/                    sharded Slurm jobs and resource estimator
tests/                            synthetic end-to-end tests
```

The previous simulation code, notebook-specific helpers, monolithic model
class, duplicated popDMS implementations, Slurm wrappers, old analysis
notebooks, logs, and generated Matplotlib cache were removed. Raw experimental
source data was retained. No pickle or legacy cache adapter is present.

## 2. End-to-end flow

```text
imported_data/*.csv
        |
        v
canonical Dataset
  variants.csv + trajectory.csv + dataset.json
        |
        +-----------------------------+
        |                             |
        v                             v
protein sequences                 reference sequence
        |                             |
        v                             v
PLM hidden states             masked-residue PLM logits
        |                             |
        v                             v
mean/max pooling                 mutant-vs-WT LLR
        |                             |
        v                             v
embedding artifact              LLR-prior artifact
        |
        +---- optional SAE ----> SAE artifact
        |
        v
zero-prior popDMS on embedding/SAE feature basis

LLR-prior artifact + substitution basis
        |
        v
prior-guided popDMS across alpha x gamma

all fitness outputs
        |
        v
cross-replicate consistency, ClinVar AUC, functional-score Spearman,
and functional/enrichment/regular-popDMS/raw-LLR baselines
```

The embedding/SAE and LLR branches are related but distinct. An SAE transforms
a vector embedding. The scalar LLR is not passed through the SAE. Instead, the
LLR supplies the mean of the Gaussian prior on amino-acid substitution
coefficients. Current prior-guided inference therefore uses the substitution
basis. Embedding and SAE inference use a zero-mean prior because their latent
dimensions do not correspond one-to-one with substitutions.

## 3. Processing `imported_data`

Run:

```bash
esmdms process imported_data --output datasets
```

Use `--datasets` followed by dataset stems to select a subset. Existing output
directories are protected unless `--force` is supplied.

### 3.1 Source validation and reference reconstruction

For every CSV, `esmdms.import_data`:

1. Requires `mutant` and `mutated_sequence`.
2. Parses each mutation as `WT + one-based position + mutant`, such as
   `I1855M` or `L2480*`.
3. Uses every non-stop row to reconstruct the wild-type sequence by replacing
   the declared mutant residue with the declared wild-type residue.
4. Requires every reconstructed candidate to be identical.
5. Requires every kept row to be exactly one declared amino-acid substitution
   from that reference.
6. Computes a stable `SequenceIndex` as
   `seq_ + sha256(protein_sequence)[:20]`.

Stop variants are counted in the manifest and excluded. Their translated
sequences may be truncated and cannot be represented by the fixed-length
protein embedding and substitution schemas.

The processor adds one `__wildtype__` row to `variants.csv`. It exists so that
reference-centered embedding and SAE operations have an explicit reference
vector. It is deliberately absent from `trajectory.csv`; no wild-type count is
invented.

### 3.2 Dataset-specific trajectory mapping

Only biological selection/time-course columns are used. Negative controls, RNA
measurements, and unrelated plasmid controls are not treated as time points.
Missing measurements in the supplied sparse/wide tables mean absence from that
library and become zero within that replicate.

| Dataset | Replicates | Generations | Mapping |
|---|---:|---|---|
| BRCA1 Findlay | 2 | 0, 5, 11 | shared `count_library`, then `day5_repN`, `day11_repN` |
| BRCA2 Huang | 6 | 0, 5, 14 | `RN_lib`, `RN_D5`, `RN_D14` |
| MSH2 Jia | 6 | 0, 2 | `RN_D_P0` paired separately with `RNDB_P2` and `RNDB6_P2` |
| VHL Buckley | 4 | 0, 1 | each arm's `pre` paired separately with `post` and `post2` |
| TP53 Kotler | 6 | 2, 6, 9, 14 | each GEO source file is one frequency replicate |

MSH2 has two endpoint conditions per experimental replicate. They are kept as
separate trajectories (`R1_DB`, `R1_DB6`, etc.) because combining them would
conflate distinct selections. VHL uses the same principle for `post` and
`post2`.

### 3.3 Imported annotations

The processor carries these fields into `variants.csv`:

- `functional_score`, including missing values;
- normalized `annotation`;
- `review_stars`, derived from the ClinVar review-status text;
- the original ClinVar significance, review status, conditions, and variation
  IDs;
- `has_clinvar`.

Review stars are assigned as follows: practice guideline = 4, expert panel = 3,
multiple submitters without conflicts = 2, criteria-provided non-conflicting
single submission = 1, and all other states = 0.

Each `dataset.json` records the source filename and SHA-256, shared ClinVar
snapshot, original/kept/stop row counts, sequence-ID rule, reference method,
and exact source-column mapping for every replicate and generation.

### 3.4 Real conversion audit

The processor was run against the supplied directory with these results:

| Dataset | Kept missense | Dropped stops | Trajectory rows | Replicates | Reference length |
|---|---:|---:|---:|---:|---:|
| MV_BRCA1_Findlay_2018 | 1,837 | 119 | 11,022 | 2 | 1,863 |
| MV_BRCA2_Huang_2025 | 4,064 | 269 | 73,152 | 6 | 3,418 |
| MV_MSH2_Jia_2020 | 17,746 | 0 | 212,952 | 6 | 934 |
| MV_TP53_Kotler_2018 | 2,967 | 191 | 71,208 | 6 | 393 |
| MV_VHL_Buckley_2024 | 1,035 | 52 | 8,280 | 4 | 213 |

Every generated replicate/generation had positive total measurement mass, and
every output reloaded through `Dataset.load` successfully.

## 4. Canonical dataset schema

Each processed dataset has exactly:

```text
dataset.json
variants.csv
trajectory.csv
```

### `dataset.json`

Dataset schema version 2 stores the reference sequence, selection direction,
source metadata, and an optional shared truncation. Truncation is either `null`
or a 1-based inclusive object such as
`{"start": 1371, "end": 3418}`. It must contain every assayed substitution.
Embedding and LLR code read this field from the same `Dataset` object.

### `variants.csv`

Required columns:

- `SequenceIndex`: stable unique row identity;
- `protein_sequence`: full model-ready protein sequence;
- `position`: one-based substitution position;
- `wt_aa`, `mutant_aa`;
- `is_synonymous`.

The schema verifies sequence length, reference residue, mutant residue, unique
IDs, and exact single-substitution reconstruction. Annotation and score columns
are optional to the schema but supplied by this processor.

### `trajectory.csv`

Required columns:

- `SequenceIndex`;
- `Replicate`;
- `Generation`;
- `Frequency`.

`Frequency` is the common numerical name for both raw counts and source
frequencies. `MeasurementKind` preserves which one it was, and `SourceColumns`
records its origin. The inference equations normalize within each time point,
so either non-negative count mass or already-normalized frequency mass is valid.

Rows must be unique by sequence, replicate, and generation. Each replicate must
have at least two generations, and every generation must have positive total
mass.

## 5. Canonical feature artifacts

All feature-like outputs use `FeatureArtifact` and the `.npz` format:

- a 2D finite numeric matrix;
- unique `SequenceIndex` rows;
- unique feature-column names;
- `kind`: `embedding`, `sae`, `llr_prior`, `basis`, or `fitness`;
- dataset name;
- JSON provenance.

Scalar LLR and fitness artifacts are matrices with one column. Artifact loading
never enables pickle.

## 6. Protein sequence to embedding

```bash
esmdms embed datasets/MV_BRCA1_Findlay_2018 \
  --model biohub/ESMC-300M \
  --layers 12 24 30 \
  --pooling max \
  --output artifacts/MV_BRCA1_Findlay_2018/300M
```

`features.embed` selects one fixed assay window, deduplicates identical protein
sequences within that window, tokenizes them with model special tokens, removes
special-token positions, and converts only the requested hidden layers to
NumPy. The current loader accepts ESM-C safetensors checkpoints, instantiates
the official ESM-C architecture directly, and requires an exact state-dictionary
match. There is no alternate model-cache or checkpoint adapter.

The default maximum window is 2,048 residues. It is centered to cover all
assayed substitutions and is shared by every variant and the wild type, so
vectors remain comparable. Proteins at or below the limit use their full
sequence. BRCA2 uses reference positions 1371-3418, which contain its complete
assayed range (2480-3186). `--window-size` can set a different limit; processing
fails rather than silently omitting mutations if the assayed span is too wide.
The optional `dataset.json` truncation overrides automatic placement with one
1-based inclusive interval. `--truncate START END` is an explicit one-run
override. Both embeddings and LLR use the resulting exact endpoints for every
row and mutation. Invalid bounds, intervals longer than `--window-size`, and
intervals that omit any assayed substitution are rejected.

Pooling converts the variable residue axis into one fixed vector:

- `mean`: arithmetic mean of each hidden dimension across all residues;
- `max`: maximum of each hidden dimension across all residues.

Layer 0 is the model's input embedding state; later indices are transformer
hidden states. One artifact is written per requested layer. Max pooling is the
CLI default because it retains strong localized activations, while mean pooling
represents global average context. They are separate artifacts and should be
benchmarked rather than combined implicitly.

The highest layer index returns the model's post-LayerNorm output state; every
lower index returns a raw transformer block output. Those are on different
scales, so layer 30 is not directly comparable with layers 12 and 24. Each
artifact records `final_layer_norm_applied` in its provenance to make this
explicit. The raw un-normalized final block output is not reachable through a
layer index.

## 7. Reference sequence to LLR prior

```bash
esmdms llr datasets/MV_BRCA1_Findlay_2018 \
  --model biohub/ESMC-300M \
  --output artifacts/MV_BRCA1_Findlay_2018/300M_llr.npz
```

For every assayed substitution position:

1. Select the same fixed assay window used for embeddings and tokenize that
   wild-type reference context.
2. Replace that residue token with the model's mask token.
3. Run the masked-language-model head.
4. Compute log-softmax over the full token vocabulary. The LLR is a difference
   of two log probabilities, so the normalizer cancels and restricting the
   softmax to amino-acid tokens would give the same value.
5. Store

   `LLR = log P(mutant | masked context) - log P(wild type | masked context)`.

More negative raw LLR means the model considers the mutant less compatible
with the wild-type context. The artifact records `orientation = raw_llr`.

When used as a popDMS selection prior, inference aligns its sign to the assay:
BRCA1, BRCA2, and VHL use raw LLR direction; MSH2 and TP53 use the opposite
direction because pathogenic variants lie on the high-selection end in those
assays. The direction is stored in `dataset.json`, not hard-coded in analysis
cells.

## 8. Optional sparse autoencoder

```bash
esmdms sae datasets/MV_BRCA1_Findlay_2018 \
  artifacts/MV_BRCA1_Findlay_2018/300M/layer_30.npz \
  --features 1920 \
  --mode batchtopk \
  --k 64 \
  --model-output artifacts/MV_BRCA1_Findlay_2018/300M_layer30_sae.pt \
  --output artifacts/MV_BRCA1_Findlay_2018/300M_layer30_sae.npz
```

Before fitting, the wild-type embedding is subtracted from every embedding by
default. Duplicate protein sequences are represented once in the training set.
The SAE uses a learned tied centering bias, a ReLU encoder, a linear decoder,
and unit-normalized decoder columns.

Sparsity modes:

- `normal`: all positive ReLU activations remain; an L1 activation penalty is
  added to reconstruction MSE;
- `topk`: for each sequence independently, only its `k` largest activations
  remain; no redundant L1 penalty is added;
- `batchtopk`: across a training batch, only `k * batch_size` activations remain
  globally, allowing samples to use different numbers of features; no L1
  penalty is added.

BatchTopK couples samples through a per-batch ranking, which would make a
sequence's stored activations depend on which other sequences were encoded
alongside it. After training, `calibrate_threshold` replaces the ranking with a
fixed scalar: the mean smallest surviving activation across training batches.
Evaluation then applies that threshold per sample, so encoding one sequence
alone gives the same result as encoding it inside the full matrix. The threshold
is a model buffer, saved in the checkpoint and recorded in provenance.

`train_fraction` holds sequences out of training. They are scored rather than
discarded: provenance records `train_sequences`, `validation_sequences`,
`train_reconstruction_mse`, and `validation_reconstruction_mse` alongside the
all-row `reconstruction_mse`.

After training, neurons that never fire or fire for essentially every sequence
are removed from the saved SAE artifact. The `.pt` checkpoint stores the full
model and active mask; the `.npz` artifact stores the active sequence features
used by inference. The active mask is stored as a tensor so the checkpoint loads
under `torch.load(weights_only=True)`.

## 9. popDMS inference

`inference.infer` accepts any feature matrix `Z` whose rows are sequences. For a
substitution basis, `Z` is one-hot by amino-acid change. For embedding or SAE
inference, `Z` contains the corresponding latent vectors.

### Sparse substitution basis

`substitution_basis` returns a `SubstitutionBasis`, which stores one column
index per variant row instead of a dense matrix. Every supported dataset has
exactly one assayed variant per substitution, so the dense form would be an
`n x n` permutation matrix: 1.26 GB for MSH2 alone, and a further copy per
replicate/generation inside the moment cache. Storing only the indices reduces
that to well under a megabyte and turns each covariance product into a gather
and a `bincount`. `SubstitutionBasis.dense()` and `.to_artifact()` materialize
the equivalent matrix for tests and small inspections.

The class exposes the same read-only attributes as `FeatureArtifact`
(`sequence_ids`, `feature_names`, `kind`, `dataset`, `provenance`), so inference
accepts either. Embedding and SAE artifacts continue to use the dense path.

### Reusing moments across a sweep

The replicate moments depend only on the trajectory and the feature basis, never
on `gamma` or the prior. `build_problem` computes them once and returns an
`InferenceProblem`; `InferenceProblem.solve(gamma=..., prior_values=...)` runs
one fit. The gamma baselines and the alpha-by-gamma sweep both build the problem
once per dataset rather than rebuilding it per grid point. `infer` remains the
single-fit convenience wrapper.

At each replicate and time point, the observed count/frequency mass is
normalized to probabilities. The method computes:

- population mean feature vector `x(t) = E[Z]`;
- endpoint change `dx = x(t_final) - x(t_initial)`;
- time-integrated covariance
  `C = integral (E[ZZ^T] - x(t)x(t)^T) dt`, using trapezoid weights.

For one replicate, selection coefficients solve:

```text
(C + gamma I) s = dx + gamma mu
```

where `gamma` is regularization/prior precision and `mu` is the coefficient
prior. Joint inference sums replicate `C` and `dx` and uses
`n_replicates * gamma`, so each replicate contributes the same prior precision.
The implementation uses a covariance `LinearOperator` and conjugate gradients;
it does not materialize a large dense covariance matrix.

Sequence fitness is:

```text
fitness(sequence) = 1 + Z(sequence) @ s_joint
```

### Zero-prior feature inference

Raw embedding and SAE benchmarks set `mu = 0`. Their coefficients describe
directions in a latent feature space.

### LLR-prior substitution inference

The LLR artifact is aligned to substitution-basis columns by
`position:mutant_aa`. The sweep uses:

```text
mu = alpha * assay_oriented_LLR
```

`alpha = 0` is the regularized zero-mean popDMS control. `alpha = 1` uses the
raw LLR scale after assay orientation. `gamma` controls confidence in that
prior. `prior_sweep` evaluates every configured alpha/gamma pair.

By default `run_analysis` (`alpha_mode = "matched"`) does not sweep raw alpha
multipliers. It scales the prior to the popDMS coefficient spread:
`inference.matched_alpha_grid` sets `s* = std(regular-popDMS coefficients at the
elbow gamma) / std(LLR)` and sweeps `alpha ∈ s*·{1/8…8 by 2}`, plus the unscaled
raw-LLR point (`alpha = 1`, flagged `unscaled_raw_llr`) and the `alpha = 0`
control. Each sweep row records `scale_multiple`, `matched_scale`, `sigma_coeff`,
and `sigma_prior`. `alpha_mode = "fixed"` restores the literal `alphas` list.

## 10. Analysis and baselines

Run the configured analysis with:

```bash
esmdms analyze analysis_config.example.json
```

The same call is used by `multi_dataset_llr_prior.ipynb`.

### Cross-replicate consistency

Each gamma/alpha configuration produces a coefficient vector per replicate.
Consistency is the mean Pearson correlation across every pair of replicate
coefficient vectors. For regular popDMS, embedding, and SAE baselines, the
reported gamma is selected at the **popDMS correlation elbow**, independently of
ClinVar labels: over the ascending grid `logspace(log10(1/max_reads), 4, 20)` the
consistency curve is fed to `regularization.get_best_regularization` (copied from
the canonical popDMS), which walks down from the peak to the elbow rather than
taking the over-regularized argmax. `baselines.csv` records the selected `gamma`
and its
`cross_replicate_consistency` on every row, so a published number can be
reproduced from that file alone. Methods without a gamma (enrichment ratio, DMS
functional score, raw LLR) leave both columns explicitly `NaN`.

### ClinVar AUC

Only exact normalized `benign` and `pathogenic` labels are used. Uncertain,
conflicting, other, and missing records are excluded. Results are computed at
configured minimum review-star thresholds.

Scores are oriented so higher always means more pathogenic before rank-based
ROC AUC is calculated. The code does not replace AUC with `max(AUC, 1-AUC)`.

`inference.assay_oriented_scores` is the single place a prior's `orientation`
provenance and the dataset's selection direction are combined. Prior-guided
inference and the raw-prior baseline both call it, so they cannot disagree. A
`raw_llr` artifact is negated for `pathogenic_high_selection` datasets; a prior
already recorded in selection units is used unchanged. The raw-prior baseline is
then evaluated with the dataset's own assay direction, exactly like every other
row, which also makes its Spearman rho sign-comparable across datasets.

### Functional-score Spearman

Where a finite source `functional_score` exists, inferred fitness is inner-joined
to those rows and Spearman correlation is reported. Partial MSH2 score coverage
is allowed; BRCA2 correctly has no functional-score baseline.

Spearman rho compares inferred fitness to the assay's own functional score and
never consults ClinVar review status, so it is emitted once as `spearman_rho`.
It is deliberately not suffixed per review-star cutoff: the cutoff filters only
the ClinVar columns (`auc`, `n_benign`, `n_pathogenic`), and a `_stars_N` suffix
on an unfiltered statistic would assert a filtering that never happened.

### Baselines

The workflow writes these comparable methods when inputs are available:

1. **DMS functional score**: supplied assay score on finite rows.
2. **Enrichment ratio**: mean replicate endpoint log2 enrichment with a
   pseudocount, computed only for trajectory-observed variants.
3. **Regular popDMS**: substitution-basis inference with zero prior; gamma is
   selected by replicate consistency.
4. **Raw embedding**: configured embedding artifact with zero prior; gamma is
   selected by replicate consistency.
5. **Raw SAE**: configured SAE artifact with zero prior; gamma is selected by
   replicate consistency.
6. **Raw LLR**: masked-marginal score before popDMS.
7. **LLR-prior popDMS**: alpha/gamma grid on the substitution basis.

The prior-sweep summary chooses the highest primary-cutoff AUC row per dataset
and prior model, with lower alpha and gamma as deterministic tie breakers. This
is a supervised model-selection summary and should be evaluated on held-out
data if used for a generalization claim.

Selection uses `head(1)` on the sorted sweep so one intact row survives.
`GroupBy.first()` must not be used: it returns the first non-null value of each
column independently, so a `NaN` in the winning row (most plausibly
`cross_replicate_consistency`) would be silently backfilled from a different
alpha/gamma, producing a summary row that never existed.

Because `alpha = 0` is the regularized zero-mean control, the AUC-maximizing row
may use no prior at all. The summary therefore also reports `auc_alpha0` (the
best `alpha = 0` AUC for the same dataset and prior), `auc_gain_over_alpha0`,
and the boolean `prior_used`, so a headline number cannot be read as evidence
for the prior when the prior contributed nothing.

### Output tables

The notebook and `esmdms analyze` read canonical datasets and feature artifacts
from disk and never create them. Both output directories are git-ignored, so a
fresh clone has neither; `esmdms process` and the feature commands must run
first. The notebook's preflight cell lists every missing configured path instead
of failing part-way through with a bare `FileNotFoundError`.

`workflow.run_analysis` writes:

- `baselines.csv`;
- `prior_sweeps.csv`;
- `summary.csv`;
- one regular-popDMS gamma table per dataset;
- one gamma table per configured embedding/SAE feature artifact;
- one alpha/gamma table per LLR prior.

## 11. Verification

The test suite is split so that the parts not needing a protein language model
run in any environment:

- `tests/test_core.py` covers schema and artifact persistence, the sparse
  substitution basis and its equivalence to the dense matrix, `InferenceProblem`
  reuse across gamma and alpha, prior alignment and assay orientation,
  per-replicate and joint inference, prior sweeps, enrichment and AUC metrics,
  summary-row integrity, and configuration-driven analysis. It imports no torch.
- `tests/test_features.py` covers the analysis window, SAE output and its
  held-out split, BatchTopK sample independence, and model-facing embedding/LLR
  generation with a deterministic test model. The whole module is skipped when
  torch is unavailable rather than failing collection for the entire suite.

`tests/conftest.py` holds the shared synthetic dataset and prior. The real
imported-data conversion is also audited against all five supplied files as
summarized above.

The checked BRCA1 demonstration is:

```bash
python3 examples/brca1_100_demo.py --prepare-only
python3 examples/brca1_100_demo.py --device cpu
```

It selects 100 missense variants in source order plus one wild-type reference
row. The real run produced a `101 x 960` final-layer max-pooled ESM-C 300M
embedding, a `101 x 237` active-feature artifact from one 256-feature BatchTopK
SAE, and a `100 x 1` masked-marginal LLR artifact. It then generated all
baseline, gamma-sweep, and alpha-by-gamma result tables. Repeating the command
reuses the three canonical artifacts and reruns only the inexpensive analysis.

## 12. Slurm execution

`esmdms embed-llr` is the cluster-facing joint model command. One process loads
one ESM-C checkpoint, embeds its assigned protein rows, then performs the masked
wild-type passes for its assigned substitution positions. Embedding and LLR
share model residency but not forward outputs because masking changes the model
input. Every embedding pass exposes all layers; pooling creates one canonical
artifact per selected layer.

For `N` array tasks, embedding rows use canonical order with stride `N`. Unique
LLR positions use the same deterministic stride, keeping every substitution at
one position together. `FeatureArtifact.merge` requires all declared shard
indices, identical feature/provenance contracts, no duplicate rows, and exact
dataset coverage before restoring canonical order.

The templates in `cluster/slurm/` form this dependency graph:

```text
embed/LLR array -> strict merge -> SAE -> SAE inference
                              +-> embedding inference
                              +-> LLR-prior substitution inference
```

Inference with embeddings or SAE activations remains zero-prior feature
inference. The LLR prior remains paired only with the substitution basis; the
Slurm inference template rejects an invalid simultaneous feature and prior
request.
