#!/bin/bash
#SBATCH --job-name=clinpgym_merge_submit
#SBATCH -p any_cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --output=job_outs/clinpgym-merge-submit-%j.out
#SBATCH --error=job_outs/clinpgym-merge-submit-%j.err

set -euo pipefail

# Submit every merge script listed in the ClinProtGym ESM-C SAE merge table.
#
# Recommended from a login node:
#   bash job_scripts/submit_clinprotgym_embedding_merges.sh
#
# Or, if your cluster allows sbatch from inside jobs:
#   sbatch job_scripts/submit_clinprotgym_embedding_merges.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/data/clinprotgym_esmc_sae"
TABLE=""
DRY_RUN=0
DEPENDENCY=""
SUBMISSION_MANIFEST=""

usage() {
    cat <<'USAGE'
Usage:
  bash job_scripts/submit_clinprotgym_embedding_merges.sh [options]

Options:
  --output-root PATH     Pipeline output root. Default: data/clinprotgym_esmc_sae
  --table PATH           Merge job table. Default: <output-root>/tables/clinprotgym_embedding_merge_jobs.csv
  --dependency SPEC      Pass a Slurm dependency to every merge job, e.g. afterok:12345
  --dry-run              Print jobs that would be submitted without calling sbatch
  -h, --help             Show this help

The script reads the table written by:
  python scripts/clinprotgym_esmc_sae_pipeline.py create-embedding-jobs ...

Each table row is submitted as its own Slurm job using the generated
submit_embedding_merge.sh or submit_window_embedding_merge.sh script.
USAGE
}

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
        --dependency)
            DEPENDENCY="$2"
            shift 2
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
    TABLE="$OUTPUT_ROOT/tables/clinprotgym_embedding_merge_jobs.csv"
fi

if [[ ! -f "$TABLE" ]]; then
    echo "Missing merge job table: $TABLE" >&2
    echo "Run create-embedding-jobs first." >&2
    exit 1
fi

SUBMISSION_MANIFEST="$OUTPUT_ROOT/tables/clinprotgym_embedding_merge_submissions.csv"
mkdir -p "$(dirname "$SUBMISSION_MANIFEST")"

row_file="$(mktemp "${TMPDIR:-/tmp}/clinpgym_merge_rows.XXXXXX")"
trap 'rm -f "$row_file"' EXIT

python3 - "$TABLE" > "$row_file" <<'PY'
import csv
import sys
from pathlib import Path

table = Path(sys.argv[1])
with table.open(newline="") as handle:
    reader = csv.DictReader(handle)
    required = {"dataset", "model", "model_short", "script_path"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"Merge table is missing columns: {sorted(missing)}")
    for row in reader:
        script = str(row.get("script_path", "")).strip()
        if not script:
            continue
        if not Path(script).is_file():
            raise SystemExit(f"Missing merge script for {row.get('dataset')} {row.get('model')}: {script}")
        values = [
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            str(row.get("model_short", "")),
            script,
            str(row.get("embedding_window_method", "")),
        ]
        print("\t".join(values))
PY

n_rows="$(wc -l < "$row_file" | tr -d ' ')"
if [[ "$n_rows" == "0" ]]; then
    echo "No merge scripts found in $TABLE"
    exit 0
fi

echo "Merge table: $TABLE"
echo "Rows to submit: $n_rows"
echo "Submission manifest: $SUBMISSION_MANIFEST"
if [[ -n "$DEPENDENCY" ]]; then
    echo "Dependency applied to each merge job: $DEPENDENCY"
fi

printf 'dataset,model,model_short,embedding_window_method,script_path,submitted,job_id,sbatch_output\n' > "$SUBMISSION_MANIFEST"

while IFS=$'\t' read -r dataset model model_short script_path window_method; do
    echo "Submitting merge: dataset=$dataset model=$model_short script=$script_path"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$dataset" "$model" "$model_short" "$window_method" "$script_path" "False" "" "dry-run" \
            >> "$SUBMISSION_MANIFEST"
        continue
    fi

    sbatch_args=()
    if [[ -n "$DEPENDENCY" ]]; then
        sbatch_args+=(--dependency="$DEPENDENCY")
    fi
    sbatch_output="$(sbatch "${sbatch_args[@]}" "$script_path")"
    job_id="$(awk '{print $NF}' <<< "$sbatch_output")"
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" "$model" "$model_short" "$window_method" "$script_path" "True" "$job_id" "$sbatch_output" \
        >> "$SUBMISSION_MANIFEST"
    echo "  $sbatch_output"
done < "$row_file"

echo "Done. Wrote $SUBMISSION_MANIFEST"
