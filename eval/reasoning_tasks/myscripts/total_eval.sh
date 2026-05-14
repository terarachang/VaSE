#!/bin/bash
# Usage: bash myscripts/check_and_gen.sh <OUT_DIR> [extra summary_results.py args...]
# For every subdir under OUT_DIR that contains run_0/completions.jsonl:
#   - prints line count
#   - runs: python summary_results.py --limit <count> --output_dir <subdir> [extra args]

total_run=8

OUT_DIR=${1:?Usage: $0 <OUT_DIR> [extra summary_results.py args...]}
shift  # remaining args forwarded to summary_results.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

# Find all dirs that have run_0/completions.jsonl
while IFS= read -r filepath; do
    output_dir="$(dirname "$(dirname "$filepath")")"  # strip /run_0/completions.jsonl
    count=$(wc -l < "$filepath")

    # Extract data_name: first directory component of output_dir
    data_name="${output_dir%%/*}"

    echo "============================================================"
    echo "DIR: $output_dir"
    echo "Samples in run_0/completions.jsonl: $count"
    echo "Running: python summary_results.py --total_run $total_run --limit $count --output_dir $output_dir --data_name $data_name $@"
    echo "------------------------------------------------------------"

    python summary_results.py --total_run $total_run --limit "$count" --output_dir "$output_dir" --data_name "$data_name" "$@"

done < <(find "$OUT_DIR" -path "*/run_0/completions.jsonl" | sort)

python aggregate_results.py $OUT_DIR
