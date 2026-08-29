"""
Download the training dataset used by the resume classifier.

Source: the Resume-Atlas dataset (13k+ labeled resumes across 43 categories),
mirrored on the Hugging Face Hub. This is a public download - no credentials
or Kaggle login required.

Usage:
    python scripts/download_data.py
"""

import os
import sys
from urllib.request import urlopen

DATA_URL = "https://huggingface.co/datasets/ahmedheakl/resume-atlas/resolve/main/train.csv"
DEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "resume_atlas.csv")


def download(url: str = DATA_URL, dest: str = DEST) -> str:
    """Download the dataset to ``dest`` if it is not already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"Already present: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    with urlopen(url) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("content-length", 0))
        read = 0
        while chunk := resp.read(1 << 20):  # 1 MB chunks
            out.write(chunk)
            read += len(chunk)
            if total:
                sys.stdout.write(f"\r  {read/1e6:5.1f} / {total/1e6:.1f} MB")
                sys.stdout.flush()
    print(f"\nDone: {os.path.getsize(dest)/1e6:.1f} MB")
    return dest


if __name__ == "__main__":
    download()
