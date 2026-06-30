#!/bin/bash
#SBATCH --job-name=clinpgym_embed_submit
#SBATCH -p any_cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --output=job_outs/clinpgym-embed-submit-%j.out
#SBATCH --error=job_outs/clinpgym-embed-submit-%j.err

set -euo pipefail

# Submit ClinProtGym embedding array jobs from the pipeline job table.
#
# By default this script only submits missing chunk indices. Rerun it after
# failures and it will submit only chunks that still do not have all expected
# chunk outputs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/data/clinprotgym_esmc_sae"
TABLE=""
PARTITION="dept_cpu"
MEM="128G"
CPUS="4"
TIME="24:00:00"
MAX_ACTIVE="2"
DRY_RUN=0
SUBMIT_ALL=0
DATASET_FILTER=""
MODEL_FILTER=""
SUBMISSION_MANIFEST=""

usage() {
    cat <<'USAGE'
Usage:
  bash job_scripts/submit_clinprotgym_embedding_arrays.sh [options]

Options:
  --output-root PATH     Pipeline output root. Default: data/clinprotgym_esmc_sae
  --table PATH           Embedding job table. Default: <output-root>/tables/clinprotgym_embedding_jobs.csv
  --partition NAME       Slurm partition override. Default: dept_cpu
  --mem AMOUNT           Slurm memory override. Default: 128G
  --cpus N               Slurm cpus-per-task override. Default: 4
  --time HH:MM:SS        Slurm time override. Default: 24:00:00
  --max-active N         Max active array tasks per dataset/model. Default: 2
  --dataset NAME         Only submit rows for this dataset. May be repeated.
  --model NAME           Only submit rows for this model or model_short. May be repeated.
  --all                  Submit all chunks, not only missing chunks
  --dry-run              Print jobs that would be submitted without calling sbatch
  -h, --help             Show this help

Examples:
  # Resubmit only missing chunks on dept_cpu with more memory than the
  # generated 64G scripts requested.
  bash job_scripts/submit_clinprotgym_embedding_arrays.sh --mem 128G

  # Try a second pass for stubborn chunks.
  bash job_scripts/submit_clinprotgym_embedding_arrays.sh --mem 192G --max-active 1

  # Submit every chunk for one dataset/model pair.
  bash job_scripts/submit_clinprotgym_embedding_arrays.sh \
    --dataset MV_MSH2_Jia_2020 --model ESMC-600M --all
USAGE
}

DATASET_FILTERS=()
MODEL_FILTERS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --table)
            TABLE="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --max-active)
            MAX_ACTIVE="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILTERS+=("$2")
            shift 2
            ;;
        --model)
            MODEL_FILTERS+=("$2")
            shift 2
            ;;
        --all)
            SUBMIT_ALL=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$REPO_ROOT"
mkdir -p job_outs

if [[ -z "$TABLE" ]]; then
    TABLE="$OUTPUT_ROOT/tables/clinprotgym_embedding_jobs.csv"
fi

if [[ ! -f "$TABLE" ]]; then
    echo "Missing embedding job table: $TABLE" >&2
    echo "Run create-embedding-jobs first." >&2
    exit 1
fi

SUBMISSION_MANIFEST="$OUTPUT_ROOT/tables/clinprotgym_embedding_array_submissions.csv"
mkdir -p "$(dirname "$SUBMISSION_MANIFEST")"

row_file="$(mktemp "${TMPDIR:-/tmp}/clinpgym_embed_rows.XXXXXX")"
trap 'rm -f "$row_file"' EXIT

dataset_filter_csv="$(IFS=,; echo "${DATASET_FILTERS[*]-}")"
model_filter_csv="$(IFS=,; echo "${MODEL_FILTERS[*]-}")"

python3 - "$TABLE" "$SUBMIT_ALL" "$MAX_ACTIVE" "$dataset_filter_csv" "$model_filter_csv" > "$row_file" <<'PY'
import csv
import pickle
import re
import sys
from pathlib import Path

table = Path(sys.argv[1])
submit_all = bool(int(sys.argv[2]))
max_active = str(sys.argv[3]).strip()
dataset_filters = {v for v in sys.argv[4].split(",") if v}
model_filters = {v for v in sys.argv[5].split(",") if v}

chunk_re = re.compile(r"_(mean_pool|max_pool|per_residue)_embeddings_chunk_(\d+)\.pkl$")

def compress_chunks(chunks: list[int]) -> str:
    # Keep this as a comma list instead of ranges. It is easy to audit in
    # Slurm logs and the chunk count is small for these jobs.
    base = ",".join(str(idx) for idx in chunks)
    if max_active and int(max_active) > 0 and len(chunks) > 1:
        return f"{base}%{max_active}"
    return base

with table.open(newline="") as handle:
    reader = csv.DictReader(handle)
    required = {"dataset", "model", "model_short", "script_path", "payload_path"}
    missing_cols = required - set(reader.fieldnames or [])
    if missing_cols:
        raise SystemExit(f"Embedding table is missing columns: {sorted(missing_cols)}")
    for row in reader:
        dataset = str(row.get("dataset", "")).strip()
        model = str(row.get("model", "")).strip()
        model_short = str(row.get("model_short", "")).strip()
        if dataset_filters and dataset not in dataset_filters:
            continue
        if model_filters and model not in model_filters and model_short not in model_filters:
            continue

        script_path = Path(str(row.get("script_path", "")).strip())
        payload_path = Path(str(row.get("payload_path", "")).strip())
        if not script_path.is_file() or not payload_path.is_file():
            continue

        with payload_path.open("rb") as payload_handle:
            payload = pickle.load(payload_handle)

        n_chunks = int(payload["n_chunks"])
        batch_dir = Path(payload["batch_dir"])
        present = {"mean_pool": set(), "max_pool": set(), "per_residue": set()}
        for path in batch_dir.glob("*_embeddings_chunk_*.pkl"):
            match = chunk_re.search(path.name)
            if match:
                present[match.group(1)].add(int(match.group(2)))

        complete_chunks = set.intersection(*present.values())
        chunks = list(range(n_chunks)) if submit_all else [
            idx for idx in range(n_chunks) if idx not in complete_chunks
        ]
        if not chunks:
            continue

        values = [
            dataset,
            model,
            model_short,
            str(script_path),
            str(payload_path),
            str(n_chunks),
            str(len(complete_chunks)),
            ",".join(str(idx) for idx in chunks),
            compress_chunks(chunks),
            str(row.get("embedding_window_method", "")),
        ]
        print("\t".join(values))
PY

n_rows="$(wc -l < "$row_file" | tr -d ' ')"
echo "Embedding table: $TABLE"
echo "Rows to submit: $n_rows"
echo "Partition: $PARTITION"
echo "Memory: $MEM"
echo "CPUs per task: $CPUS"
echo "Time: $TIME"
echo "Max active array tasks per row: $MAX_ACTIVE"
echo "Submission manifest: $SUBMISSION_MANIFEST"

printf 'dataset,model,model_short,embedding_window_method,script_path,payload_path,n_chunks,complete_chunks,submitted_chunks,array_spec,partition,mem,cpus,time,submitted,job_id,sbatch_output\n' > "$SUBMISSION_MANIFEST"

if [[ "$n_rows" == "0" ]]; then
    echo "No embedding chunks need submission."
    exit 0
fi

while IFS=$'\t' read -r dataset model model_short script_path payload_path n_chunks complete_chunks submitted_chunks array_spec window_method; do
    echo "Submitting embeddings: dataset=$dataset model=$model_short chunks=$submitted_chunks"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s,%s\n' \
            "$dataset" "$model" "$model_short" "$window_method" "$script_path" "$payload_path" \
            "$n_chunks" "$complete_chunks" "$submitted_chunks" "$array_spec" "$PARTITION" "$MEM" \
            "$CPUS" "$TIME" "False" "" "dry-run" >> "$SUBMISSION_MANIFEST"
        continue
    fi

    sbatch_output="$(
        sbatch --parsable \
            -p "$PARTITION" \
            --mem="$MEM" \
            --cpus-per-task="$CPUS" \
            --time="$TIME" \
            --array="$array_spec" \
            "$script_path"
    )"
    job_id="${sbatch_output%%;*}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s",%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" "$model" "$model_short" "$window_method" "$script_path" "$payload_path" \
        "$n_chunks" "$complete_chunks" "$submitted_chunks" "$array_spec" "$PARTITION" "$MEM" \
        "$CPUS" "$TIME" "True" "$job_id" "$sbatch_output" >> "$SUBMISSION_MANIFEST"
    echo "  Submitted batch job $job_id"
done < "$row_file"

echo "Done. Wrote $SUBMISSION_MANIFEST"
