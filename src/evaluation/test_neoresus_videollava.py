#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference / test script for neonatal-resuscitation multi-label classification
with Video-LLaVA.

New in this version
-------------------
1. **W&B integration** (metrics are logged when --labels is provided).
2. Full accuracy / precision / recall / F1 per class + macro averages.

Run examples
------------
# Zero-shot baseline
python test_neoresus_videollava.py \
    --model LanguageBind/Video-LLaVA-7B-hf \
    --video data/clips/clip_00001.mp4

# Fine-tuned LoRA checkpoint with metrics logging
python test_neoresus_videollava.py \
    --model checkpoints/neoresus_lora \
    --list  data/clips/testlist.txt \
    --labels data/clips/test.jsonl \
    --project videollava_eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Sequence

import av
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

import wandb  # noqa: F401 – required for auto-logging

# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "USER: <video>\n"
    "You are an expert neonatal instructor.\n"
    "Possible events:\n"
    "A. Baby mannequin visible on the table.\n"
    "B. A caregiver holds a ventilation (PPV/CPAP) mask on the baby's face.\n"
    "C. A caregiver rubs the baby's trunk/back (stimulation).\n"
    "D. A suction tube is inserted in the baby's mouth or nose.\n\n"
    "Rules:\n"
    "1. B, C and D can only happen if A happens.\n"
    "2. B and D are mutually exclusive.\n\n"
    "TASK → Respond ONLY with the JSON dictionary:\n"
    '{"baby_visible":<yes/no>,"ventilation":<yes/no>,'
    '"stimulation":<yes/no>,"suction":<yes/no>} '
    "ASSISTANT:"
)

MAX_NEW_TOKENS = 64
NUM_FRAMES = 8
_FIELDS = ["baby_visible", "ventilation", "stimulation", "suction"]

# ---------------------------------------------------------------------------


def sample_frames(path: str, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Uniformly sample exactly `num_frames` RGB24 frames from an MP4."""
    container = av.open(path)
    total = container.streams.video[0].frames
    if total < num_frames:
        raise RuntimeError(f"{path}: expected ≥{num_frames} frames, found {total}")
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)

    frames: List[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > idxs[-1]:
            break
        if i in idxs:
            frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if len(frames) != num_frames:
        raise RuntimeError(f"{path}: could not decode {num_frames} frames.")
    return np.stack(frames)


_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_json_from_text(txt: str) -> Dict[str, str] | None:
    """Return first JSON object found in text (or None)."""
    m = _JSON_RE.search(txt)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def enforce_rules(js: dict | None) -> Dict[str, str]:
    """Post-hoc constraint solver (matches training prompt rules)."""
    def _empty():
        return {
            "baby_visible": "no",
            "ventilation": "no",
            "stimulation": "no",
            "suction": "no",
        }

    if not js:
        return _empty()

    js = {k: str(v).lower() for k, v in js.items()}
    yes = lambda v: v in {"yes", "true", "1"}

    if not yes(js.get("baby_visible", "no")):
        return _empty()

    # mutually exclusive
    if yes(js.get("ventilation", "no")) and yes(js.get("suction", "no")):
        js["suction"] = "no"

    for k in ("ventilation", "stimulation", "suction"):
        js.setdefault(k, "no")
    return js


def _dict2vec(d: Dict[str, str]) -> np.ndarray:
    return np.array([1 if d[k] == "yes" else 0 for k in _FIELDS], dtype=np.int8)


# ---------------------------------------------------------------------------


def classify_clip(
    model: VideoLlavaForConditionalGeneration,
    processor: VideoLlavaProcessor,
    video_path: str,
    apply_rules: bool = True,
):
    frames = sample_frames(video_path)
    inputs = processor(
        text=PROMPT_TEMPLATE, videos=frames, return_tensors="pt", padding=True
    ).to(model.device)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
    js = parse_json_from_text(decoded)
    if apply_rules:
        js = enforce_rules(js)
    return js, decoded


def load_model(model_ckpt: str):
    processor = VideoLlavaProcessor.from_pretrained(model_ckpt)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_ckpt, device_map="auto"
    )
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------


def compute_and_log_metrics(
    preds: Sequence[Dict[str, str]],
    gts: Sequence[Dict[str, str]],
    project: str,
):
    """Compute per-class + macro metrics and send them to W&B."""
    y_pred = np.vstack([_dict2vec(p) for p in preds])
    y_true = np.vstack([_dict2vec(t) for t in gts])

    metrics: Dict[str, float] = {}
    precs, recs, f1s = [], [], []

    for i, fld in enumerate(_FIELDS):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        metrics.update(
            {
                f"{fld}_accuracy": acc,
                f"{fld}_precision": prec,
                f"{fld}_recall": rec,
                f"{fld}_f1": f1,
            }
        )
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    metrics["macro_precision"] = float(np.mean(precs))
    metrics["macro_recall"] = float(np.mean(recs))
    metrics["macro_f1"] = float(np.mean(f1s))

    run = wandb.init(project=project, job_type="test")
    wandb.log(metrics)                                               # :contentReference[oaicite:0]{index=0}
    run.finish()


# ---------------------------------------------------------------------------


def main(argv: List[str]):
    ap = argparse.ArgumentParser(
        description="Inference tester for Video-LLaVA neonatal-resuscitation model"
    )
    ap.add_argument("--model", required=True, help="Checkpoint name or path")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Path to a single .mp4 video")
    g.add_argument("--list", help="Text file with one video path per line")

    ap.add_argument(
        "--labels",
        help="JSONL file with ground-truth labels (same format as training data). "
        "If supplied, metrics are computed and logged to W&B.",
    )
    ap.add_argument(
        "--project",
        default="videollava_neoresus",
        help="Weights & Biases project name (default: %(default)s)",
    )
    ap.add_argument(
        "--no-rules", action="store_true", help="Skip rule-enforcement post-processing"
    )

    args = ap.parse_args(argv)

    # ---------- load model ----------
    model, processor = load_model(args.model)

    # ---------- collect videos ----------
    videos: List[str]
    if args.video:
        videos = [args.video]
    else:
        videos = [v.strip() for v in open(args.list) if v.strip()]

    # ---------- optional ground-truth ----------
    gt_map: Dict[str, Dict[str, str]] = {}
    if args.labels:
        ds = load_dataset("json", data_files=args.labels, split="train")
        gt_map = {rec["video"]: enforce_rules(rec["labels"]) for rec in ds}

    preds, gts = [], []

    # ---------- inference ----------
    for vp in tqdm(videos, desc="videos"):
        try:
            pred_js, raw = classify_clip(
                model, processor, vp, apply_rules=not args.no_rules
            )
            print(json.dumps({"video": vp, "pred": pred_js}, ensure_ascii=False))

            preds.append(pred_js)
            if vp in gt_map:
                gts.append(gt_map[vp])
        except Exception as e:
            print(
                json.dumps({"video": vp, "error": str(e)}, ensure_ascii=False),
                file=sys.stderr,
            )

    # ---------- metrics ----------
    if args.labels and gts:
        compute_and_log_metrics(preds, gts, project=args.project)
    elif args.labels:
        print(
            "⚠️  No ground-truth entries matched the provided videos – metrics skipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
