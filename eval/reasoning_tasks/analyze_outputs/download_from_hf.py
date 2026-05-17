#!/usr/bin/env python3
"""Download *.zip files from HuggingFace dataset Ting-Yun/kv."""

import argparse
import fnmatch
import os
from huggingface_hub import HfApi

REPO_ID = "Ting-Yun/kv"
REPO_TYPE = "dataset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--output_dir", default="existing_results", help="Directory to save downloaded files")
    parser.add_argument("--files", nargs="+", metavar="PATTERN", default=["all_results.zip"],
                        help="Only download files matching these names or glob patterns (e.g. aime* gpqa.zip)")
    parser.add_argument("--list", action="store_true", help="List available zip files and exit")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    os.makedirs(args.output_dir, exist_ok=True)

    all_files = [f for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
                 if f.endswith(".zip")]

    if not all_files:
        print("No .zip files found in the dataset.")
        return

    if args.list:
        print(f"Available zip files ({len(all_files)}):")
        for f in all_files:
            print(f"  {f}")
        return

    if args.files:
        files = [f for f in all_files
                 if any(fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(os.path.basename(f), pat)
                        for pat in args.files)]
        if not files:
            print(f"No zip files matched the given patterns: {args.files}")
            print(f"Available files: {all_files}")
            return
    else:
        files = all_files

    print(f"Downloading {len(files)} zip file(s): {files}")
    for filename in files:
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        api.hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=args.output_dir,
        )
        print("done")

    print("All downloads complete.")

if __name__ == "__main__":
    main()
