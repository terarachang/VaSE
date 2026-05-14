#!/usr/bin/env python3
"""Upload *.zip files to HuggingFace dataset Ting-Yun/kv."""

import argparse
import glob
import os
from huggingface_hub import HfApi

REPO_ID = "Ting-Yun/kv"
REPO_TYPE = "dataset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    zip_files = ["aime25.zip", "aime26.zip", "gpqa.zip", "math.zip", "tokens.zip"]
    if not zip_files:
        print("No .zip files found in current directory.")
        return

    print(f"Found {len(zip_files)} zip file(s): {zip_files}")
    print(f"Uploading to {REPO_ID} ...")

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        print(f"  Uploading {filename} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print("done")

    print("All uploads complete.")

if __name__ == "__main__":
    main()
