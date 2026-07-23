# Slurm templates

These templates submit the expensive stages without adding a second analysis
path. Every job calls the same `esmdms` CLI used locally and reads or writes the
canonical dataset and `.npz` artifact schemas.

## Execution graph

```text
embed_llr.sbatch array (one full ESM-C model per shard)
                  |
                  v
merge_embed_llr.sbatch
          |                  |
          v                  v
      sae.sbatch       LLR-prior inference
          |                  |
          v                  v
     SAE inference     embedding inference
```

`embed-llr` loads ESM-C once in each array task. Variant embedding passes return
all transformer layers at once. LLR still requires additional wild-type passes,
one for every masked mutation position, so it cannot reuse the unmasked forward
outputs. Keeping both operations in one job avoids a second checkpoint load and
keeps the same model resident for both stages.

Embedding rows are assigned by canonical `SequenceIndex` order. LLR work is
assigned by unique mutation position, so all substitutions at one position
share one masked pass and stay in one shard. The number of shards must not
exceed either the number of protein rows or the number of unique eligible
mutation positions. Merging refuses missing, duplicate, or incompatible shards
and restores canonical row order.

## Cluster environment

Create the environment once on the cluster:

```bash
python3 -m venv /shared/path/venvs/esmdms
source /shared/path/venvs/esmdms/bin/activate
python3 -m pip install -e '/shared/path/esmClone[test]'
```

Use a shared persistent Hugging Face cache. If compute nodes cannot reach the
internet, populate it from a login or transfer node before submitting jobs:

```bash
export HF_HOME="$SCRATCH/huggingface"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('biohub/ESMC-300M')"
```

Do not place final artifacts under node-local `$TMPDIR`. The job scripts use
`$TMPDIR` only for Matplotlib cache files.

## ESM-C 6B GPU smoke test

The smoke test loads the official ESM-C 6B checkpoint once and sequentially
mean-pools the final residue embeddings for the first 100 full-length
(1,863-residue) BRCA1 sequences. It records the GPU model, peak CUDA memory,
load time, total inference time, and per-sequence timing statistics.

```bash
export REPO_ROOT=/shared/path/esmDMS
export VENV=/shared/path/venvs/esmdms
export OUTPUT="$SCRATCH/esmdms/esmc6b-smoke"

# Use your site's name for an 80 GB GPU, if the generic request is insufficient.
export GPU_RESOURCE=gpu:a100:1
bash cluster/slurm/submit_smoke_esmc_6b.sh
```

The helper exports these paths into the batch job. Direct submission also works
when run from the repository root; the job uses `SLURM_SUBMIT_DIR`, `.venv`, and
a job-specific directory under `results/` as defaults:

```bash
sbatch cluster/slurm/smoke_esmc_6b.sbatch
```

The defaults request 160 GB host RAM, one GPU, two hours, BF16 weights, and at
least 48 GiB of GPU memory. Sites that select GPU memory through a constraint
can instead set, for example, `GPU_CONSTRAINT=a100_80gb`. Override
`PARTITION`, `ACCOUNT`, `QOS`, `GPU_RESOURCE`, `GPU_CONSTRAINT`, `MEMORY`, or
`TIME_LIMIT` without editing the job. A successful run writes the following
beneath `$OUTPUT`:

- `brca1_first100_mean_embeddings.npy`: by default, a 100-by-2,560 matrix of
  mean-pooled final-layer embeddings. Set `NUM_SEQUENCES` to override the row
  count; the selected count is included in the filename.
- `metrics.json`: GPU identity, PyTorch peak allocated/reserved VRAM, and
  model-load, inference, and total Python times.
- `gpu_memory.csv`: one-second samples of total/used GPU memory, utilization,
  and power, including memory outside PyTorch's allocator.
- `process_resources.txt`: GNU `time -v` results, including elapsed time and
  maximum resident host memory.
- `slurm-*.out` and `slurm-*.err`: the complete job logs.

After the job completes, use Slurm accounting as a second host-memory and timing
measurement:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,ReqMem
```

## Submit the complete graph

The submission helper supplies dependencies and submits three distinct
inference jobs: final-layer embedding features, SAE features, and substitution
features with an LLR prior.

```bash
export REPO_ROOT=/shared/path/esmClone
export VENV=/shared/path/venvs/esmdms
export DATASET=/shared/path/esmClone/datasets/MV_BRCA1_Findlay_2018
export OUTPUT_ROOT="$SCRATCH/esmdms/brca1_300m"
export MODEL_SIZE=300m
export NUM_SHARDS=8
export DEVICE_MODE=gpu

bash cluster/slurm/submit_pipeline.sh
```

No truncation environment variable is needed. The worker loads the optional
1-based inclusive `truncation` from the dataset's `dataset.json`, using the same
interval for embeddings and every LLR. When `truncation` is `null`, it
automatically places one fixed window that covers all assayed substitutions.

The templates deliberately omit account, partition, QoS, and GPU-model
directives. Add site-specific defaults to the files or pass them through your
normal Slurm configuration. `GPU_RESOURCE` defaults to `gpu:1`; for example:

```bash
export GPU_RESOURCE=gpu:a100:1
bash cluster/slurm/submit_pipeline.sh
```

For CPU embedding, use the same worker without a GPU request:

```bash
export DEVICE_MODE=cpu
export NUM_SHARDS=16
bash cluster/slurm/submit_pipeline.sh
```

CPU array tasks each load a complete model. More shards improve throughput but
multiply simultaneous host-memory use and checkpoint reads across nodes.

## Submit stages manually

GPU embedding and LLR array:

```bash
sbatch --array=0-7 --gres=gpu:a100:1 --mem=32G \
  --export=ALL,REPO_ROOT="$REPO_ROOT",VENV="$VENV",DATASET="$DATASET",MODEL_ID=biohub/ESMC-300M,OUTPUT_ROOT="$OUTPUT_ROOT",NUM_SHARDS=8,DEVICE_MODE=gpu \
  cluster/slurm/embed_llr.sbatch
```

CPU embedding and LLR array:

```bash
sbatch --array=0-7 --mem=32G \
  --export=ALL,REPO_ROOT="$REPO_ROOT",VENV="$VENV",DATASET="$DATASET",MODEL_ID=biohub/ESMC-300M,OUTPUT_ROOT="$OUTPUT_ROOT",NUM_SHARDS=8,DEVICE_MODE=cpu \
  cluster/slurm/embed_llr.sbatch
```

Submit `merge_embed_llr.sbatch` with an `afterok` dependency on the array. SAE
and inference jobs should depend on the merge job. `submit_pipeline.sh` is the
reference for the exact variables and dependency syntax.

## BRCA1-length memory estimates

The estimates below use a 1,863-residue BRCA1 protein, an unbatched forward
pass, all hidden layers retained by the model, and BF16/FP16 GPU weights. Host
RAM remains important because the loader reads float32 safetensors and creates
the model on the CPU before moving it to a GPU.

| Size | Layers | Hidden | Parameters | Slurm host RAM | Minimum GPU VRAM | CPU-only RAM |
|---|---:|---:|---:|---:|---:|---:|
| ESM-C 300M | 30 | 960 | 333M | 16 GB | 8 GB | 16 GB |
| ESM-C 600M | 36 | 1,152 | 575M | 24 GB | 12 GB | 24 GB |
| Custom 3B estimate | 36 | 2,560 | 3.0B | 64 GB | 24 GB | 64 GB |
| ESM-C 6B | 80 | 2,560 | 6.35B | 128 GB | 48 GB | 128 GB |

These are initial requests, not guarantees. CUDA/PyTorch versions, the SDPA
attention backend, GPU architecture, and allocator fragmentation change peak
usage. FP32 GPU inference should start around 12, 16, 40, and 80 GB respectively.
Check Slurm `MaxRSS` and GPU high-water marks after the first job, then tune the
requests for the local cluster.

`submit_pipeline.sh` uses more conservative host-RAM defaults of 32, 48, 96,
and 160 GB for 300M, 600M, custom 3B, and 6B respectively. Override them with
`EMBED_MEM` after measuring representative jobs on the target cluster.

Biohub currently publishes ESM-C 300M, 600M, and 6B, not ESM-C 3B. The 3B row
is capacity planning for a compatible custom ESM-C checkpoint. The repository
loader requires `model_type = esmc`, so an ESM-2 3B checkpoint is not a drop-in
replacement. Use `MODEL_SIZE=3b MODEL_ID=<custom-checkpoint>` only when that
checkpoint follows the supported ESM-C safetensors layout.

Calculate the tensor components for another sequence count or length:

```bash
python3 cluster/slurm/estimate_resources.py \
  --model 300m --length 1863 --sequences 101 --dtype bf16
```

Sharding reduces wall time and pooled-artifact memory per task. It does not
reduce the model or per-sequence activation memory because every array task
loads a full checkpoint and processes one full sequence at a time.

## Output layout

```text
$OUTPUT_ROOT/
  shards/shard_000_of_008/{embeddings/layer_*.npz,llr.npz}
  merged/{embeddings/layer_*.npz,llr.npz}
  sae/{layer_N.npz,layer_N.pt}
  results/{embedding_fitness,llr_prior_fitness,sae_fitness}.npz
```
