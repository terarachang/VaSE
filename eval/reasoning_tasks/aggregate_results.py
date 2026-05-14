#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path
assert len(sys.argv) == 2, 'file dir'

def parse_summary(filepath):
    data = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r'#\s*Examples:\s*(\d+)', line)
            if m:
                data['examples'] = int(m.group(1))
            m = re.match(r'Average Acc:\s*([\d.]+)%', line)
            if m:
                data['acc'] = float(m.group(1))
            m = re.match(r'Average generate length:\s*(\d+)', line)
            if m:
                data['avg_gen_len'] = int(m.group(1))
            m = re.match(r'No ans count:\s*(\d+)', line)
            if m:
                data['no_ans'] = int(m.group(1))
            m = re.match(r'Total_run:\s*(\d+)', line)
            if m:
                data['total_run'] = int(m.group(1))
    return data

rows = []
base = Path(sys.argv[1])

for summary_file in base.rglob('overall_summary.txt'):
    rel = summary_file.parent.relative_to(base)
    parts = rel.parts
    directory = parts[0]
    config = '/'.join(parts[1:]) if len(parts) > 1 else ''
    data = parse_summary(summary_file)
    examples = data.get('examples', 'N/A')
    acc = data.get('acc', None)
    avg_gen_len = data.get('avg_gen_len', None)
    no_ans = data.get('no_ans', None)
    total_run = data.get('total_run', None)
    rows.append((directory, config, examples, acc, avg_gen_len, no_ans, total_run))

rows.sort(key=lambda r: (r[0], r[1]))

# Auto-size columns
w_dir = max(len("Directory"),  max(len(r[0]) for r in rows))
w_cfg = max(len("Config"),     max(len(r[1]) for r in rows))

header = f"{'Directory':<{w_dir}}  {'Config':<{w_cfg}}  {'Avg Acc':>8}  {'Avg Len':>8}  {'No Ans':>6}  {'#':>4}  {'Runs':>4}"
sep    = f"{'-'*w_dir}  {'-'*w_cfg}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*4}  {'-'*4}"
print(header)
print(sep)
for directory, config, examples, acc, avg_gen_len, no_ans, total_run in rows:
    acc_str = f"{acc:.2f}%" if acc is not None else 'N/A'
    gen_len_str = str(avg_gen_len) if avg_gen_len is not None else 'N/A'
    no_ans_str = str(no_ans) if no_ans is not None else 'N/A'
    total_run_str = str(total_run) if total_run is not None else 'N/A'
    print(f"{directory:<{w_dir}}  {config:<{w_cfg}}  {acc_str:>8}  {gen_len_str:>8}  {no_ans_str:>6}  {str(examples):>4}  {total_run_str:>4}")
