#!/bin/bash
# Usage: bash myscripts/total_eval_gsm8k.sh <OUT_DIR>
# For every experiment dir under OUT_DIR that contains run_0/completions.jsonl:
#   - counts examples from the completions file
#   - runs: python run_gsm8k.py --print_only --output_dir <dir> --limit <n> --total_run <k>
# Then aggregates all results with aggregate_results.py.

OUT_DIR=${1:?Usage: $0 <OUT_DIR>}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

while IFS= read -r filepath; do
    output_dir="$(dirname "$(dirname "$filepath")")"
    limit=$(wc -l < "$filepath")
    total_run=$(find "$output_dir" -maxdepth 1 -name "run_*" -type d | wc -l)

    echo "============================================================"
    echo "DIR: $output_dir"
    echo "Examples: $limit  Runs: $total_run"
    echo "------------------------------------------------------------"

    python run_gsm8k.py --print_only \
        --output_dir "$output_dir" \
        --limit "$limit" \
        --total_run "$total_run"

done < <(find "$OUT_DIR" -path "*/run_0/completions.jsonl" | sort)

python aggregate_results.py "$OUT_DIR"
