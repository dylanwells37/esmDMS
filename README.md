# esmDMS

This repository estimates selection coefficients for assayed amino-acid
substitutions from deep mutational scanning (DMS) time courses. Protein language
models are used only to calculate masked-marginal log-likelihood ratios (LLRs),
which can be supplied as priors for the substitution-level inference.

## Setup

```bash
make setup
source .venv/bin/activate
```

Python 3.10 or newer is required. The default setup installs ESM-C, PyTorch,
NumPy, pandas, SciPy, and the test/notebook dependencies.

## Data model

A canonical dataset directory contains:

- `dataset.json`: dataset name, reference sequence, selection direction, and an
  optional 1-based inclusive PLM analysis interval.
- `variants.csv`: one row per assayed mutation, keyed by `SequenceIndex`.
- `trajectory.csv`: mutation frequencies by replicate and generation.

Create datasets from the imported count-bearing CSV files:

```bash
esmdms process imported_data --output datasets
```

Or validate and create one dataset directly:

```bash
esmdms dataset --name example --reference reference.txt \
  --variants variants.csv --trajectory trajectory.csv --output datasets/example
```

## LLR calculation

Calculate mutant-minus-wildtype masked-marginal LLRs:

```bash
esmdms llr datasets/MV_BRCA1_Findlay_2018 \
  --model biohub/ESMC-300M \
  --device cuda --dtype bf16 \
  --output artifacts/MV_BRCA1_Findlay_2018/300M_llr.npz
```

For array jobs, pass `--shard-index` and `--num-shards`, then merge the outputs:

```bash
esmdms merge datasets/MV_BRCA1_Findlay_2018 \
  artifacts/shard_0.npz artifacts/shard_1.npz \
  --output artifacts/MV_BRCA1_Findlay_2018/300M_llr.npz
```

The LLR window is fixed across all mutations. It is selected automatically to
cover the assayed sites, read from the dataset's `truncation` field, or
overridden with `--truncate START END`.

## Mutational-site inference

Run regular popDMS inference on the one-hot amino-acid substitution basis:

```bash
esmdms infer datasets/MV_BRCA1_Findlay_2018 \
  --gamma 0.1 --output results/fitness.npz
```

Add a PLM LLR prior with a scale factor:

```bash
esmdms infer datasets/MV_BRCA1_Findlay_2018 \
  --prior artifacts/MV_BRCA1_Findlay_2018/300M_llr.npz \
  --alpha 1.0 --gamma 0.1 --output results/llr_prior_fitness.npz
```

An alpha-by-gamma sweep is available through `esmdms sweep`. For a configured
multi-dataset analysis, copy `analysis_config.example.json` and run:

```bash
esmdms analyze analysis_config.json
```

The workflow compares enrichment, optional DMS functional scores, regular
popDMS, raw LLR, and LLR-prior popDMS. It writes tidy baseline, sweep, and
summary CSVs.

## Cluster experiment

`experiments/model_size_llr/submit_experiment.sh` launches the retained
multi-model LLR-prior benchmark. See `cluster/slurm/README.md` for the job graph
and environment variables.

## Tests

```bash
pytest
```

The tests cover dataset/artifact validation, substitution-level inference,
regularization, metrics, workflows, PLM checkpoint loading, fixed analysis
windows, masked-marginal LLR calculation, and LLR shard merging.
