#!/usr/bin/env python3
"""
Create a 40-example balanced sample from a JSONL file of multilabel
video clips.  Assumes each line looks like

{"video": "...", "labels": {"baby_visible": 0/1, "ventilation": 0/1,
                            "stimulation": 0/1, "suction": 0/1}}
"""

import json, random, argparse
from collections import defaultdict, Counter
from pathlib import Path

LABELS           = ["baby_visible", "ventilation", "stimulation", "suction"]
TARGET_PER_LABEL = 50          # 40 examples ÷ 4 labels
NEGATIVES = 10

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def balanced_sample(records, labels, target_per_label, negatives, seed=42):
    random.seed(seed)
    # index positives for each label
    positives = {lbl: [i for i, r in enumerate(records) if r["labels"][lbl]==1]
                 for lbl in labels}
    all_negatives = [i for i, r in enumerate(records)
                     if all(r["labels"][lbl] == 0 for lbl in labels)]

    selected = set()
    # round-robin selection so we avoid bias toward any label
    while any(len([i for i in selected if records[i]["labels"][lbl]==1]) < target_per_label
              for lbl in labels):
        for lbl in labels:
            have = [i for i in selected if records[i]["labels"][lbl]==1]
            need = target_per_label - len(have)
            if need <= 0:              # this label is satisfied
                continue
            pool = [idx for idx in positives[lbl] if idx not in selected]
            if not pool:               # ran out of unseen positives → reuse
                pool = positives[lbl]
            random.shuffle(pool)
            selected.update(pool[:need])
            if len(selected) >= target_per_label*len(labels):  # reached 40
                break
    if all_negatives:
        random.shuffle(all_negatives)
        selected.update(all_negatives[:negatives])

    # If we overshot a little, trim down to exactly 40 examples
    selected = list(selected)[:target_per_label*len(labels)]
    return [records[i] for i in selected]

def main(in_path: Path, out_path: Path):
    records  = load_jsonl(in_path)
    sample   = balanced_sample(records, LABELS, TARGET_PER_LABEL, NEGATIVES)
    write_jsonl(sample, out_path)

    # quick sanity check
    counts = Counter()
    for r in sample:
        for lbl in LABELS:
            counts[lbl] += r["labels"][lbl]
    print(f"Wrote {len(sample)} records to {out_path}")
    for lbl in LABELS:
        print(f"  {lbl:12s}: {counts[lbl]} positives")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create a balanced 40-row sample")
    p.add_argument("-i", "--input",  type=Path, default="data/clips/test.jsonl")
    p.add_argument("-o", "--output", type=Path, default="data/clips/test_balanced_40.jsonl")
    args = p.parse_args()
    main(args.input, args.output)
