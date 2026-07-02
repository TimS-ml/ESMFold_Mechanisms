#!/usr/bin/env python3
"""Upload data/ and results/ to HuggingFace Hub."""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ID = "kevinlu4588/ProteinFolding"  # Default HF Hub dataset repo for this project
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ is one level below the repo root
FOLDERS = ["data", "results"]  # Local folders synced to the Hub dataset repo
# Glob patterns excluded from every upload (transient/cache/editor files, not real data)
IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "*.swp",
    "*.swo",
    ".ipynb_checkpoints",
]
MAX_RETRIES = 5  # Max attempts per folder upload before giving up
RETRY_DELAY = 60  # Base backoff in seconds between retries (doubled for rate-limit errors)


def parse_args():
    """Parse command-line arguments for the upload script."""
    parser = argparse.ArgumentParser(
        description="Upload data/ and results/ to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id", default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Make the repo public (default: private)",
    )
    parser.add_argument(
        "--folders", nargs="+", default=FOLDERS,
        help="Which folders to upload (default: data and results)",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def ensure_authenticated(api: HfApi, token: str | None):
    """Ensure the HfApi client has valid credentials, prompting an interactive login if needed."""
    if token:
        api.token = token
        return
    try:
        # whoami() succeeds silently if a cached CLI/token login already exists
        api.whoami()
        return
    except Exception:
        pass
    print("Not logged in to HuggingFace. Launching interactive login...")
    login()


def upload_with_retry(api: HfApi, **kwargs):
    """Call api.upload_folder with retries on 504/429 errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.upload_folder(**kwargs)
            return
        except Exception as e:
            err = str(e)
            # 504 = gateway timeout, 429 = rate limited; both are transient and worth retrying
            if ("504" in err or "429" in err) and attempt < MAX_RETRIES:
                # Back off twice as long for rate limiting (429) as for a plain timeout
                delay = RETRY_DELAY * (2 if "429" in err else 1)
                print(f"    Error (attempt {attempt}/{MAX_RETRIES}), "
                      f"retrying in {delay}s...")
                print(e)
                time.sleep(delay)
            else:
                raise


def upload_folder_chunked(api: HfApi, folder_path: Path, folder_name: str,
                          repo_id: str, ignore_patterns: list[str]):
    """Upload per-subfolder to keep commits small."""
    skip_dirs = {"__pycache__", ".ipynb_checkpoints"}
    subdirs = sorted([d for d in folder_path.iterdir()
                      if d.is_dir() and d.name not in skip_dirs])
    # Loose top-level files (not inside any subfolder) get uploaded together in one commit
    files = [f for f in folder_path.iterdir()
             if f.is_file() and not any(f.match(p) for p in ignore_patterns)]

    if files:
        print(f"  Uploading {folder_name}/ (top-level files)...")
        upload_with_retry(
            api,
            folder_path=str(folder_path),
            path_in_repo=folder_name,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f.name for f in files],  # restrict this commit to just these files
            ignore_patterns=ignore_patterns,
        )

    # Upload each subfolder as its own separate commit so a failure only requires
    # retrying that one subfolder rather than the whole (potentially huge) folder
    for i, subdir in enumerate(subdirs, 1):
        print(f"  [{i}/{len(subdirs)}] Uploading {folder_name}/{subdir.name}/ ...")
        upload_with_retry(
            api,
            folder_path=str(subdir),
            path_in_repo=f"{folder_name}/{subdir.name}",
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=ignore_patterns,
        )


def main():
    """Authenticate, create/reuse the HF dataset repo, and upload each requested folder."""
    args = parse_args()
    api = HfApi()

    ensure_authenticated(api, args.token)
    user = api.whoami()["name"]
    print(f"Authenticated as: {user}")

    # exist_ok=True makes this safe to re-run against an already-created repo
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    print(f"Repo ready: {repo_url}")

    for folder in args.folders:
        folder_path = PROJECT_ROOT / folder
        if not folder_path.is_dir():
            print(f"WARNING: {folder_path} does not exist, skipping.")
            continue

        print(f"\nUploading {folder}/ ...")
        upload_folder_chunked(api, folder_path, folder, args.repo_id, IGNORE_PATTERNS)
        print(f"  Done: {folder}/")

    print(f"\nAll uploads complete!")
    print(f"View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()