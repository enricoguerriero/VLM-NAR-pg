#!/usr/bin/env python3
"""
Create a CSV (or JSON) with a row per video and one column per label.

Typical call:
    python build_resus_dataset.py \
        --root  /path/to/raw/videos \
        --out   /path/to/labels.csv
"""
import argparse, json, re, shutil
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# 1.  Label-extraction core (same logic as before, wrapped in a class)
# ──────────────────────────────────────────────────────────────────────────
class ResusVideoLabeler:
    _PATTERNS: Dict[str, re.Pattern] = {
        "Baby visible": re.compile(r"baby\s*visible", re.I),
        "Ventilation" : re.compile(r"(?:^|[_\-])P?-?(?:CPAP|PPV)(?:[_\-]|\.|$)", re.I),
        "Stimulation" : re.compile(r"(?:^|[_\-])P?-?Stimulation\s*(?:backnates|trunk)", re.I),
        "Suction"     : re.compile(r"(?:^|[_\-])P?-?Suction(?:[_\-]|\.|$)", re.I),
    }
    ORDER = ["Baby visible", "Ventilation", "Stimulation", "Suction"]

    @classmethod
    def labels_for_file(cls, fname: str | Path) -> Set[str]:
        name = Path(fname).name
        return {lbl for lbl, pat in cls._PATTERNS.items() if pat.search(name)}

    @classmethod
    def encode_multi_hot(cls, fname: str | Path) -> List[int]:
        present = cls.labels_for_file(fname)
        return [int(lbl in present) for lbl in cls.ORDER]


# ──────────────────────────────────────────────────────────────────────────
# 2.  Build a DataFrame, optionally copy/rename videos
# ──────────────────────────────────────────────────────────────────────────
def build_dataset(root: Path, copy_to: Path | None = None) -> pd.DataFrame:
    video_paths = sorted(root.rglob("*.avi"))
    records = []

    for idx, vid in enumerate(video_paths):
        labels = ResusVideoLabeler.labels_for_file(vid)
        multi_hot = ResusVideoLabeler.encode_multi_hot(vid)
        dest = None

        # Optional: copy (or move) the file to a clean “dataset” folder
        if copy_to is not None:
            copy_to.mkdir(parents=True, exist_ok=True)
            dest = copy_to / f"{idx:06d}{vid.suffix.lower()}"
            shutil.copy2(vid, dest)

        records.append(
            dict(
                file=str(dest if dest else vid),
                **{k: v for k, v in zip(ResusVideoLabeler.ORDER, multi_hot)},
                # keep the raw text labels too (handy for debugging)
                labels=list(labels),
            )
        )

    return pd.DataFrame.from_records(records)


# ──────────────────────────────────────────────────────────────────────────
# 3.  CLI entry-point
# ──────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="Directory that contains the raw *.avi files (walked recursively)")
    p.add_argument("--out",  required=True, type=Path,
                   help="Where to write the labels file (extension .csv or .json)")
    p.add_argument("--copy_videos_to", type=Path,
                   help="(Optional) folder that will receive a clean, sequential copy "
                        "of every video—good for reproducible ML splits")
    args = p.parse_args()

    df = build_dataset(args.root, args.copy_videos_to)

    if args.out.suffix.lower() == ".csv":
        df.to_csv(args.out, index=False)
    elif args.out.suffix.lower() == ".json":
        df.to_json(args.out, orient="records", indent=2)
    else:
        raise ValueError("Use .csv or .json as the output filename")

    print(f"✓ Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
