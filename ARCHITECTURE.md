# Architecture

esmDMS has one inference space: assayed amino-acid substitutions from a DMS
experiment. Protein language models contribute scalar masked-marginal LLRs only.

## Modules

- `schema.py` validates canonical datasets and scalar LLR/fitness artifacts.
- `import_data.py` converts supported count-bearing CSV files into canonical
  mutation trajectories.
- `features.py` loads ESM-C checkpoints and computes masked-marginal LLRs.
- `inference.py` constructs a sparse one-hot substitution basis and estimates
  per-substitution selection coefficients.
- `regularization.py` selects popDMS gamma by the replicate-correlation elbow.
- `metrics.py` evaluates inferred mutation fitness.
- `workflow.py` runs configured regular-popDMS and LLR-prior comparisons.
- `cli.py` exposes processing, LLR, merge, inference, sweep, and analysis
  commands.

## Data flow

```text
imported mutation counts -> canonical Dataset
                                  |
                                  +-> sparse substitution basis -> popDMS fitness
                                  |
reference sequence -> PLM masked LLR
                                  |
                                  +-> substitution prior --------^
```

Every substitution basis row stores only its feature-column index. Wild-type
and synonymous rows use `-1` and therefore project to zero. This avoids
materializing the otherwise near-permutation `n x n` design matrix.

For each replicate and generation, inference computes the mutation-frequency
mean and covariance action. Selection coefficients solve the regularized linear
system with conjugate gradients. A regular run uses a zero prior; an LLR-prior
run aligns the raw LLR sign to the assay direction and scales it by alpha.

## LLR semantics

At each assayed position, ESM-C receives the fixed reference-sequence window
with that residue masked. The saved value is:

```text
log P(mutant amino acid) - log P(wild-type amino acid)
```

One forward pass supplies every assayed mutant amino acid at that position.
The artifact records `orientation: raw_llr`, model identity, window endpoints,
and optional shard provenance. Merge restores canonical mutation order and
rejects missing, duplicate, or incompatible shards.

## Workflow outputs

Configured analyses write:

- a regular-popDMS gamma table for each dataset;
- one alpha-by-gamma table per LLR prior;
- `baselines.csv`, including enrichment, functional score when present,
  regular popDMS, and raw LLR;
- `prior_sweeps.csv` and `summary.csv`.

The multi-model experiment under `experiments/model_size_llr` calculates LLRs
for three ESM-C sizes, performs substitution-level analyses, aggregates tables,
and executes the reporting notebook.
