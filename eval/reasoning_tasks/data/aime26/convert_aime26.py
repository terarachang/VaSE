"""Convert math-ai/aime26 HuggingFace dataset to data/aime26/test.jsonl"""
import json
import os
from datasets import load_dataset

ds = load_dataset("math-ai/aime26", split="test")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jsonl")
with open(out_path, "w") as f:
    for example in ds:
        f.write(json.dumps(example) + "\n")

print(f"Wrote {len(ds)} examples to {out_path}")
