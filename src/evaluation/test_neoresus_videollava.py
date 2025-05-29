#!/usr/bin/env python
"""
Test-and-metrics script for neonatal-resuscitation multi-label classification with **Video-LLaVA**.

v3 – bug-fixed
--------------
* Fixed indentation + newline bug in streaming loop.
* Each JSON line now contains both the prediction and, *when available*, the
  ground-truth labels (key: `label`).
* All other functionality (W&B, metrics) unchanged.

Run examples remain the same – see the header of the file.
"""

import argparse, json, re, sys, os, numpy as np, av, torch
from typing import List, Dict
from tqdm import tqdm
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

LABELS = ["baby_visible", "ventilation", "stimulation", "suction"]
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
    "{\"baby_visible\":<yes/no>,\"ventilation\":<yes/no>,"
    "\"stimulation\":<yes/no>,\"suction\":<yes/no>} "
    "ASSISTANT:"
)
MAX_NEW_TOKENS = 64
NUM_FRAMES     = 8

def sample_frames(path: str, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Uniformly sample exactly `num_frames` RGB24 frames from a video."""
    container = av.open(path)
    total = container.streams.video[0].frames
    if total < num_frames:
        raise RuntimeError(f"{path}: expected ≥{num_frames} frames, found {total}")
    idx = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = [f.to_ndarray(format="rgb24") for i, f in enumerate(container.decode(video=0)) if i in idx]
    container.close()
    if len(frames) != num_frames:
        raise RuntimeError(f"Could not decode {num_frames} frames from {path}")
    return np.stack(frames)

def parse_json_from_text(txt: str):
    """Return the first JSON object found in `txt` or `None`."""
    m = re.search(r"\{.*?\}\s*", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def enforce_rules(js: Dict[str, str]):
    """Apply dependency & mutual-exclusion rules post-hoc."""
    def _no():
        return {l: "no" for l in LABELS}

    if not js:
        return _no()

    js = {k: str(v).lower() for k, v in js.items()}
    yes = lambda v: v in {"yes", "true", "1"}

    if not yes(js.get("baby_visible", "no")):
        return _no()

    if yes(js.get("ventilation", "no")) and yes(js.get("suction", "no")):
        js["suction"] = "no"  # favour ventilation

    for k in LABELS:
        js.setdefault(k, "no")
    return js

def classify_clip(model, processor, video_path: str, apply_rules: bool = True):
    frames = sample_frames(video_path)
    inputs = processor(text=PROMPT_TEMPLATE, videos=frames, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    js  = parse_json_from_text(out)
    if apply_rules:
        js = enforce_rules(js)
    return js, out

def load_model(model_ckpt: str):
    processor = VideoLlavaProcessor.from_pretrained(model_ckpt)
    model     = VideoLlavaForConditionalGeneration.from_pretrained(model_ckpt, device_map="auto")
    model.eval()
    return model, processor

def add_metric(acc, prec, rec, f1, lbl, metrics):
    metrics[f"acc_{lbl}"] = acc
    metrics[f"prec_{lbl}"] = prec
    metrics[f"rec_{lbl}"]  = rec
    metrics[f"f1_{lbl}"]   = f1

def compute_metrics(y_true: Dict[str, List[int]], y_pred: Dict[str, List[int]]):
    metrics = {}
    for lbl in LABELS:
        acc = accuracy_score(y_true[lbl], y_pred[lbl])
        p,r,f,_ = precision_recall_fscore_support(y_true[lbl], y_pred[lbl], average='binary', zero_division=0)
        add_metric(acc,p,r,f,lbl,metrics)
    # macro averages
    macro_acc = np.mean([metrics[f"acc_{l}"] for l in LABELS])
    yt = np.array([y_true[l] for l in LABELS]).T
    yp = np.array([y_pred[l] for l in LABELS]).T
    p_macro,r_macro,f_macro,_ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    metrics.update({
        "acc_macro": macro_acc,
        "prec_macro": p_macro,
        "rec_macro": r_macro,
        "f1_macro": f_macro,
    })
    return metrics

def print_metrics_table(metrics: Dict[str, float]):
    """Pretty print per-class and macro metrics."""
    sep = "-"*66
    print(sep)
    print(f"| {'Label':<13}| {'Acc':>6}| {'Prec':>6}| {'Rec':>6}| {'F1':>6}|")
    print(sep)
    for lbl in LABELS:
        print(f"| {lbl:<13}| {metrics[f'acc_{lbl}']:.3f}| {metrics[f'prec_{lbl}']:.3f}| {metrics[f'rec_{lbl}']:.3f}| {metrics[f'f1_{lbl}']:.3f}|")
    print(sep)
    print(f"| {'MACRO':<13}| {metrics['acc_macro']:.3f}| {metrics['prec_macro']:.3f}| {metrics['rec_macro']:.3f}| {metrics['f1_macro']:.3f}|")
    print(sep)


def main(argv: List[str]):
    ap = argparse.ArgumentParser(description="Inference & metrics for Video-LLaVA neonatal model")
    ap.add_argument("--model", required=True, help="Checkpoint name or path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Path to a single video file (no metrics)")
    g.add_argument("--list", help="Text file with one video path per line (no metrics)")
    g.add_argument("--jsonl", help="HF-style jsonl file; may contain 'labels' for metrics")
    ap.add_argument("--out", help="Write predictions to this file (jsonl); else stdout")
    ap.add_argument("--no_rules", action="store_true", help="Skip rule enforcement")
    # W&B
    ap.add_argument("--wandb_project", default=None, help="W&B project name (logs metrics)")
    ap.add_argument("--wandb_run",     default=None, help="W&B run name")
    ap.add_argument("--no_wandb", action="store_true", help="Disable W&B even if project is set")
    args = ap.parse_args(argv)

    wb = None
    if args.wandb_project and not args.no_wandb:
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run, config={"model": args.model})

    model, processor = load_model(args.model)

    # Collect records
    if args.video:
        records = [{"video": args.video, "labels": None}]
    elif args.list:
        vids = [v.strip() for v in open(args.list) if v.strip()]
        records = [{"video": v, "labels": None} for v in vids]
    else:  # jsonl
        records = [json.loads(l) for l in open(args.jsonl)]
        for r in records:
            r.setdefault("labels", None)

    # containers for metrics
    y_true = {lbl: [] for lbl in LABELS}
    y_pred = {lbl: [] for lbl in LABELS}

    sink = open