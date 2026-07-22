# ESM-DMS

This repository runs one analysis path for protein language-model embeddings,
sparse autoencoder (SAE) features, and LLR-prior popDMS inference. The Python
package contains the analysis; `multi_dataset_llr_prior.ipynb` is the single
interactive entry point.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed data conversion,
embedding, SAE, LLR-prior, inference, and evaluation flow.

## Install

```bash
python3 -m pip install -e '.[notebook,test]'
```

## Data Format

Convert the supplied `imported_data` directory first:

```bash
esmdms process imported_data --output datasets
```

The processor reconstructs each reference sequence, drops stop variants, adds a
trajectory-free wild-type row for reference-centered SAE training, maps each
assay's count/frequency columns to replicate trajectories, and validates the
result through the canonical schema. Use `--force` to replace existing outputs
or `--datasets MV_BRCA1_Findlay_2018 ...` to process a subset.

Each dataset is one directory with exactly three files:

```text
datasets/MV_BRCA1_Findlay_2018/
  dataset.json
  variants.csv
  trajectory.csv
```

`dataset.json`:

```json
{
  "schema_version": 2,
  "name": "MV_BRCA1_Findlay_2018",
  "reference_sequence": "M...",
  "pathogenic_high_selection": false,
  "truncation": null
}
```

Long proteins can set one shared 1-based inclusive interval directly in the
manifest, for example `"truncation": {"start": 1371, "end": 3418}`. The
canonical BRCA2 dataset already contains this setting. Dataset loading rejects
an interval that is invalid or omits an assayed substitution.

`variants.csv` has one row per variant. Required columns are
`SequenceIndex`, `protein_sequence`, `position`, `wt_aa`, `mutant_aa`, and
`is_synonymous`. Optional analysis columns are `functional_score`,
`annotation` (`benign` or `pathogenic`), and `review_stars`.

`trajectory.csv` is a tidy table with `SequenceIndex`, `Replicate`,
`Generation`, and non-negative `Frequency`. Every replicate needs at least two
generations. Positions in `variants.csv` are one-based. Stop variants must be
removed before creating a dataset because every row must contain a model-ready
protein sequence with the same length as the reference.

Create and validate a dataset:

```bash
esmdms dataset \
  --name MV_BRCA1_Findlay_2018 \
  --reference reference.txt \
  --variants variants.csv \
  --trajectory trajectory.csv \
  --output datasets/MV_BRCA1_Findlay_2018
```

Add `--truncate START END` when creating a long-protein dataset. Feature commands
read that value automatically; their own `--truncate` option is only an explicit
one-run override.

## Feature Artifacts

Embeddings, SAE activations, LLR priors, feature bases, and inferred fitness all
use `FeatureArtifact`. Artifacts are compressed `.npz` files containing a 2D
numeric matrix, unique `SequenceIndex` row ids, named feature columns, an
artifact kind, dataset name, and provenance. No pickle or method-specific cache
format is supported.

Embedding and LLR commands use one fixed assay window of at most 2,048 residues
by default. A configured `dataset.json` truncation takes precedence over
automatic placement and is shared across all variants and every LLR. Passing
`--truncate START END` to a feature command temporarily overrides the dataset's
1-based inclusive endpoints. The interval must cover every assayed substitution
and is never changed per mutation.

```bash
# Max-pooled embeddings, one artifact per selected layer
esmdms embed datasets/MV_BRCA1_Findlay_2018 --model biohub/ESMC-300M \
  --layers 12 24 30 --pooling max --output artifacts/MV_BRCA1_Findlay_2018/300M

# Masked-marginal prior
esmdms llr datasets/MV_BRCA1_Findlay_2018 --model biohub/ESMC-300M \
  --output artifacts/MV_BRCA1_Findlay_2018/300M_llr.npz

# Joint generation automatically reads the shared BRCA2 dataset truncation
esmdms embed-llr datasets/MV_BRCA2_Huang_2025 \
  --model biohub/ESMC-300M \
  --output artifacts/MV_BRCA2_Huang_2025/300M

# SAE features
esmdms sae datasets/MV_BRCA1_Findlay_2018 artifacts/MV_BRCA1_Findlay_2018/300M/layer_30.npz \
  --features 1920 --mode topk --k 64 \
  --model-output artifacts/MV_BRCA1_Findlay_2018/300M_layer30_sae.pt \
  --output artifacts/MV_BRCA1_Findlay_2018/300M_layer30_sae.npz

# One prior-guided popDMS fit
esmdms infer datasets/MV_BRCA1_Findlay_2018 \
  --prior artifacts/MV_BRCA1_Findlay_2018/300M_llr.npz \
  --alpha 1 --gamma 0.1 --output results/MV_BRCA1_Findlay_2018_fitness.npz
```

## Multi-Dataset Analysis

Edit `analysis_config.example.json`, then run:

```bash
esmdms analyze analysis_config.example.json
```

The workflow writes tidy baseline, alpha/gamma sweep, and best-configuration
tables. Raw embeddings and SAE artifacts can be listed under `features`; LLR
artifacts belong under `priors`. The notebook reads the same configuration and
calls the same workflow.

## BRCA1 100-Variant Demonstration

The runnable demonstration selects the first 100 BRCA1 missense rows, retains a
wild-type reference row, generates max-pooled final-layer ESM-C 300M
embeddings, trains one BatchTopK SAE, computes masked-marginal LLR priors, and
runs all inference and evaluation baselines:

```bash
# Generate and validate only the small canonical dataset
python3 examples/brca1_100_demo.py --prepare-only

# Run the complete pipeline on Apple Silicon
python3 examples/brca1_100_demo.py --device mps

# Or run on CUDA with reduced-precision model weights
python3 examples/brca1_100_demo.py --device cuda --dtype bf16
```

The full run writes `demo/brca1_100/analysis_config.json`, canonical `.npz`
artifacts, an SAE `.pt` checkpoint, tidy result CSVs, and `run_summary.json`.
Completed stages are reused; pass `--force` to recompute them. The ESM-C model
is downloaded from Hugging Face on the first full run and may require a Hugging
Face login in environments that do not already have access.

## Slurm

Standard Slurm templates are under [`cluster/slurm`](cluster/slurm/README.md).
The embedding array command loads ESM-C once per shard and calculates both all
requested embedding layers and masked-marginal LLRs. Separate dependent jobs
merge shards, train an SAE, and infer fitness from final-layer embeddings, SAE
features, and an LLR prior.

```bash
export REPO_ROOT=/shared/path/esmClone
export VENV=/shared/path/venvs/esmdms
export DATASET="$REPO_ROOT/datasets/MV_BRCA1_Findlay_2018"
export OUTPUT_ROOT="$SCRATCH/esmdms/brca1_300m"
export MODEL_SIZE=300m NUM_SHARDS=8 DEVICE_MODE=gpu
bash cluster/slurm/submit_pipeline.sh
```

GPU requests are added by the submission helper only when `DEVICE_MODE=gpu`;
the same array worker supports `DEVICE_MODE=cpu`. The cluster guide documents
manual submission, artifact layout, sharding rules, and BRCA1-length resource
estimates for 300M, 600M, custom 3B, and official 6B configurations.

## Tests

```bash
python3 -m pytest
```

Source code is GPL-3.0 licensed. Repository data and figures are CC0 licensed.
