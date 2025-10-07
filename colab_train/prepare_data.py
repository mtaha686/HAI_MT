#!/usr/bin/env python3
"""
Copies project data files into colab_train/data for packaging to Colab.
"""

import shutil
from pathlib import Path


def main():
    here = Path(__file__).parent
    root = here.parent
    src_dir = root / "data"
    dst_dir = here / "data"
    dst_dir.mkdir(exist_ok=True, parents=True)

    for name in ["train.jsonl", "test.jsonl", "full_dataset.jsonl"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            print(f"Copied {src} -> {dst_dir / name}")
        else:
            print(f"Skip missing {src}")


if __name__ == "__main__":
    main()



