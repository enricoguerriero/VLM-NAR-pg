#!/usr/bin/env python3
"""
Convert the neonatal-resuscitation CSV exported by your clip-generator
into a JSON file where every row becomes an object like

{
  "video": "clips/clip_00001.mp4",
  "labels": {
    "baby_visible": 1,
    "ventilation": 0,
    "stimulation": 1,
    "suction": 0
  }
}
"""

import argparse
import csv
import json
import os
from pathlib import Path


# ────────────────────────────── helpers ───────────────────────────── #

def build_video_path(csv_path: str, index: int) -> str:
    """
    Transform the 'file' column into the value you want under
    the `video` key.

    The default behaviour below:
      • strips “data/” if present
      • changes the extension to .mp4
      • rewrites the name as clips/clip_00001.mp4, clips/clip_00002.mp4 …

    If you'd rather keep the original filename (or devise your own
    naming rule), just replace the body of this function.
    """
    # Example: clips/clip_00001.mp4
    return f"clips/clip_{index:05d}.mp4"


def row_to_record(row: dict, index: int) -> dict:
    """Map one CSV row to the JSON schema."""
    return {
        "video": build_video_path(row["file"], index),
        "labels": {
            "baby_visible": int(row["Baby visible"]),
            "ventilation": int(row["Ventilation"]),
            "stimulation": int(row["Stimulation"]),
            "suction": int(row["Suction"]),
        },
    }


# ────────────────────────────── driver ───────────────────────────── #

def convert(csv_file: Path, json_file: Path) -> None:
    with csv_file.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        records = [row_to_record(r, i + 1) for i, r in enumerate(reader)]

    # Write a pretty-printed JSON array (JSON Lines is one-liner change if preferred)
    with json_file.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"✓ Wrote {len(records)} records to {json_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert neonatal-resuscitation CSV → JSON")
    ap.add_argument("csv_in", type=Path, help="input CSV file")
    ap.add_argument("json_out", type=Path, help="output JSON file")
    args = ap.parse_args()

    convert(args.csv_in, args.json_out)
