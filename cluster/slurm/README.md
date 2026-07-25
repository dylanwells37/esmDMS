# Slurm workflows

The retained cluster workflow benchmarks ESM-C masked-marginal LLR priors for
selection coefficients defined on DMS amino-acid substitutions:

```text
experiment_controller.sbatch
  -> llr.sbatch (dataset x PLM jobs on GPU)
  -> analyze.sbatch (regular and LLR-prior popDMS on CPU)
  -> report.sbatch (aggregate tables and execute notebook)
```

Launch it with:

```bash
bash experiments/model_size_llr/submit_experiment.sh
```

The launcher creates per-dataset configs under
`results/model_size_llr/configs`, then submits the controller. The controller
processes missing canonical datasets, submits LLR jobs to the GPU cluster,
waits for their artifacts, and submits analysis and reporting jobs to the CPU
cluster.

The single-job scripts can also be used directly:

- `llr.sbatch` expects `REPO_ROOT`, `VENV`, `DATASET_DIR`, `MODEL_ID`, and
  `OUTPUT`.
- `infer.sbatch` expects `REPO_ROOT`, `VENV`, `DATASET`, and `OUTPUT`; optional
  variables are `PRIOR`, `ALPHA`, and `GAMMA`.
- `analyze.sbatch` expects `REPO_ROOT`, `VENV`, and `CONFIG`.

`estimate_resources.py` remains available for observing representative jobs.
Site-specific cluster, partition, memory, time, and GPU settings may be
overridden at submission.
