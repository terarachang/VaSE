"""
Combine MathArena/hmmt_feb_2025 and MathArena/hmmt_nov_2025 datasets
and convert to the format of data/aime25/test.jsonl:
  {"problem": "...", "answer": "...", "id": "0"}
"""

import json
import os
from datasets import load_dataset

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "test.jsonl")

ds_feb = load_dataset("MathArena/hmmt_feb_2025")["train"]
ds_nov = load_dataset("MathArena/hmmt_nov_2025")["train"]

records = []
for ds in [ds_feb, ds_nov]:
    for row in ds:
        records.append({"problem": row["problem"].strip(), "answer": row["answer"]})

with open(output_path, "w") as f:
    for idx, rec in enumerate(records):
        rec["id"] = str(idx)
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(records)} records to {output_path}")
