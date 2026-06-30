# ClinProtGym Master Controller

`scripts/clinprotgym_master_controller.py` is a stateful orchestrator for the
ClinProtGym ESM-C SAE workflow. It coordinates the existing embedding, merge,
SAE, benchmark, ensemble, and summary commands without replacing them.

The Slurm entrypoint is:

```bash
job_scripts/run_clinprotgym_master_controller.sh
```

## What It Does

Each controller run is short-lived:

1. Read the controller state JSON.
2. Check whether tracked Slurm jobs are still running.
3. Inspect files on disk.
4. Submit the next needed stage.
5. Submit a future controller check with `sbatch --begin=now+Nminutes`.
6. Exit.

This avoids holding a Slurm allocation just to sleep.

The default state file is:

```text
data/clinprotgym_esmc_sae/tables/clinprotgym_master_controller_state.json
```

## Default Embedding Retry Policy

The controller starts with larger chunks than the original 40-chunk jobs:

| Attempt | `n_chunks` | Memory | Max active tasks per dataset/model |
|---:|---:|---:|---:|
| 1 | 120 | `64G` | 6 |
| 2 | 180 | `64G` | 6 |

When an embedding attempt finishes with missing chunks, the controller checks
the worst missing-chunk fraction for any dataset/model pair. If more than 10%
of chunks are missing for any pair, it archives active chunk files for the
incomplete pairs and increases `n_chunks` to the next policy. If 10% or fewer
chunks are missing, it resubmits only the missing chunks using the same policy.
Archived files are moved into timestamped directories inside each batch
directory; the controller does not delete outputs.

The per-dataset/model array limit starts at the policy value (`%6` by default).
If the remaining embedding rows would allow fewer than 20 active tasks in
aggregate, the submitter raises the per-row array limits for the remaining
rows so the total can approach 20 active embedding tasks. If fewer than 20
chunks remain total, it simply allows all remaining chunks to run.

Example archive directory:

```text
data/clinprotgym_esmc_sae/datasets/<dataset>/jobs/embedding_batches/<model>/old_outputs_controller_failed_<timestamp>_n120/
```

This matters because chunk index `0` under `n_chunks=40` is not the same set of
sequences as chunk index `0` under `n_chunks=120`.

## Start The Controller

From the repo root:

```bash
cd /net/dali/home/barton/dhw28/popDMS/esmDMS
sbatch job_scripts/run_clinprotgym_master_controller.sh
```

The wrapper passes `--resubmit-self`, so the controller will keep scheduling
follow-up checks until the pipeline is complete or blocked.

Check logs:

```bash
ls -ltr job_outs/clinpgym-master-*.out job_outs/clinpgym-master-*.err
```

Check state:

```bash
cat data/clinprotgym_esmc_sae/tables/clinprotgym_master_controller_state.json
```

## Useful Options

Run a one-shot check without scheduling the next controller job:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh --no-resubmit-self
```

Change the polling interval:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh --poll-minutes 60
```

Use a custom embedding retry policy:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh \
  --embedding-policy 120:64G:6 \
  --embedding-policy 180:64G:6
```

Change the escalation threshold:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh \
  --embedding-escalate-failure-fraction 0.05
```

Change the aggregate active-task target for embedding retries:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh \
  --embedding-target-active-total 16
```

Proceed to SAE/analysis using only complete datasets if embeddings remain
incomplete after all embedding policies are exhausted:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh --allow-partial-downstream
```

Disable ensemble jobs and stop after benchmark collection plus summaries:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh --no-run-ensemble
```

## Stage Details

### Embeddings

The controller runs:

```bash
python scripts/clinprotgym_esmc_sae_pipeline.py cache-status
python scripts/clinprotgym_esmc_sae_pipeline.py create-embedding-jobs --skip-complete --n-chunks <policy>
bash job_scripts/submit_clinprotgym_embedding_arrays.sh
bash job_scripts/submit_clinprotgym_embedding_merges.sh
```

It waits for embedding arrays and merge jobs via tracked Slurm job IDs before
checking again.

### SAE Training

After all requested models have complete merged embedding caches for all
datasets, the controller runs:

```bash
python scripts/clinprotgym_esmc_sae_pipeline.py create-sae-jobs --submit
python scripts/clinprotgym_esmc_sae_pipeline.py collect-sae
```

SAE jobs are created only for datasets with complete merged embeddings. Reruns
are idempotent when `--force-recompute` is not used: completed SAE feature and
inference outputs are reused.

### Count-Based Analysis

Benchmarks and ensembles are restricted to complete datasets with trajectory
count/frequency data in the processing manifest. The controller excludes
prepared datasets that have no counts.

It runs:

```bash
python scripts/clinprotgym_esmc_sae_pipeline.py create-benchmark-jobs --count-datasets-only --submit
python scripts/clinprotgym_esmc_sae_pipeline.py collect-benchmarks --count-datasets-only
python scripts/clinprotgym_esmc_sae_pipeline.py create-ensemble-jobs --count-datasets-only --submit
python scripts/clinprotgym_esmc_sae_pipeline.py collect-ensembles --count-datasets-only
python scripts/clinprotgym_esmc_sae_pipeline.py summarize --count-datasets-only
python scripts/clinprotgym_esmc_sae_pipeline.py cross-replicate-consistency --count-datasets-only --omit-empty-datasets
```

If no finite Fixed DeltaEmbSAE benchmark candidates exist, the ensemble stage is
skipped automatically.

## Resetting The Controller State

If you want to restart orchestration from the beginning, move the state file:

```bash
mv data/clinprotgym_esmc_sae/tables/clinprotgym_master_controller_state.json \
   data/clinprotgym_esmc_sae/tables/clinprotgym_master_controller_state.json.bak
```

Do not remove active Slurm jobs just by moving the state file. If jobs are still
running, let them finish or cancel them intentionally before restarting.

## Safety Notes

- The controller never deletes embedding chunk outputs. It archives them before
  changing `n_chunks`.
- Do not manually regenerate the same stage payload while a controller-submitted
  array from that stage is still running.
- The controller tracks only jobs it submits. If you submit extra embedding,
  merge, SAE, benchmark, or ensemble jobs manually, wait for them to finish
  before starting the controller.
- `--allow-partial-downstream` is useful if some datasets remain problematic,
  but the default is to wait for all requested embeddings to complete.
