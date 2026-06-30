# ClinProtGym ESM-C SAE Pipeline

This pipeline adapts the ClinProtGym final CSVs in
`data/clin_dms_data/data/final` to the existing `esmDMS`
embedding, DeltaEmbSAE, popDMS-inference, and rank-ensemble code paths.

Main entry point:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py --help
```

Automatic Slurm controller:

```bash
sbatch job_scripts/run_clinprotgym_master_controller.sh
```

The controller checks embeddings, escalates failed chunk jobs with larger
`n_chunks`, submits merges, then runs SAE and count-ready analysis stages. See
`docs/clinprotgym_master_controller.md` for details.

Analysis notebook:

```text
clinprotgym_esmc_sae_analysis.ipynb
```

Default output root:

```text
data/clinprotgym_esmc_sae
```

The default models are `biohub/ESMC-300M` and `biohub/ESMC-600M`. A future
ESM-C 3B checkpoint can be added without changing the pipeline by passing both
the model id and layer count:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py create-embedding-jobs \
  --models biohub/ESMC-300M biohub/ESMC-600M FUTURE_ESMC_3B_ID \
  --model-layer-count FUTURE_ESMC_3B_ID=N
```

## Execution Order

1. Prepare all datasets:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py prepare
```

The default input is the repo-local ClinDMS final table directory. If raw
count/frequency layers are reconstructed upstream, first fold them into
`data/clin_dms_data/data/final` with the ClinDMS `assemble_final.py` workflow;
this pipeline treats those final CSVs as authoritative and will reprocess a
dataset automatically when its final CSV size or mtime changes. After a final
CSV changes, rerun downstream jobs from embeddings onward so caches and
benchmark rows correspond to the new sequence/count table.

2. Create and submit embedding jobs:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py create-embedding-jobs --skip-complete
```

Submit scripts are listed in:

```text
data/clinprotgym_esmc_sae/tables/clinprotgym_embedding_jobs.csv
```

After embedding chunks finish, submit the merge scripts listed in:

```text
data/clinprotgym_esmc_sae/tables/clinprotgym_embedding_merge_jobs.csv
```

Then verify caches:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py cache-status
```

3. Create and submit masked-marginal LLR jobs:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py create-llr-jobs
sbatch data/clinprotgym_esmc_sae/jobs/llr/submit_clinprotgym_llr_array.sh
```

4. Create and submit fixed DeltaEmbSAE jobs after embedding caches exist:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py create-sae-jobs
sbatch data/clinprotgym_esmc_sae/jobs/fixed_deltaembsae_layer_array/submit_clinprotgym_fixed_deltaembsae_layer_array.sh
python3 scripts/clinprotgym_esmc_sae_pipeline.py collect-sae
```

The SAE architecture matches the current reference best configuration:
`DeltaEmbSAE`, `batchtopk`, `k=64`, `n_features=12800`, `epochs=200`,
`batch_size=64`, `lr=1e-3`, `norm_scheme=none`.

5. Rebuild benchmark tasks after LLR, embedding, and SAE outputs exist, then
submit:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py create-benchmark-jobs
sbatch data/clinprotgym_esmc_sae/jobs/benchmark_row_analysis/submit_clinprotgym_benchmark_row_analysis_array.sh
python3 scripts/clinprotgym_esmc_sae_pipeline.py collect-benchmarks
```

6. Build the gamma-1 SAE ensemble from the top-12 single SAE rows by Spearman:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py create-ensemble-jobs --ensemble-gamma 1.0
sbatch data/clinprotgym_esmc_sae/jobs/sae_ensemble_gamma1/submit_clinprotgym_sae_ensemble_gamma1_array.sh
python3 scripts/clinprotgym_esmc_sae_pipeline.py collect-ensembles
```

For datasets with `functional_score` values, Spearman is agreement with those
functional scores. If a dataset lacks functional scores but has real
count/frequency trajectories, benchmark rows and SAE ensemble selection fall
back to agreement with the enrichment-ratio baseline. Each metrics row records
the target in `spearman_target`.

7. Summarize average Spearman and AUC across datasets:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py summarize
```

Summary output:

```text
data/clinprotgym_esmc_sae/tables/clinprotgym_average_spearman_auc_summary.csv
```

8. Plot cross-replicate consistency by dataset and method:

```bash
python3 scripts/clinprotgym_esmc_sae_pipeline.py cross-replicate-consistency
```

This writes:

```text
data/clinprotgym_esmc_sae/tables/clinprotgym_cross_replicate_consistency_by_dataset_method.csv
data/clinprotgym_esmc_sae/figures/clinprotgym_cross_replicate_consistency_by_dataset_method.png
```

The final cell of `clinprotgym_esmc_sae_analysis.ipynb` runs the same command
path through the pipeline helper.

This plot does not run popDMS, raw-embedding, SAE, or ensemble inference itself.
Those method dots appear only when their completed benchmark or ensemble
artifacts have been collected into the pipeline metrics tables.

## Data Availability

The adapter uses real count/frequency trajectories when present. Datasets
without real trajectories still get reference sequences, embeddings, SAEs, and
LLR scores, but enrichment ratio, popDMS, raw-embedding popDMS inference, and
SAE popDMS inference require real trajectories and are omitted for those
datasets. Separately, datasets without `functional_score` values contribute
Spearman against enrichment-ratio fitness when real trajectories are present,
and can still contribute ClinVar AUC. The summary reports `n_datasets`, finite
metric counts, and `spearman_targets` per method.
