#!/usr/bin/env python
"""
Test script for neonatal‑resuscitation multi‑label classification with Video‑LLaVA.

It can be run in two modes:
1. **Zero‑shot / pre‑training** – supply the base checkpoint id/name.
2. **Fine‑tuned**           – supply the path to the directory produced by `trainer.save_model()`.

Examples
--------
# 1. Baseline (no fine‑tuning)
python test_neoresus_videollava.py \
    --model LanguageBind/Video-LLaVA-7B-hf \
    --video data/clips/clip_00001.mp4

# 2. After LoRA fine‑tuning
python test_neoresus_videollava.py \
    --model checkpoints/neoresus_lora \
    --video data/clips/clip_00001.mp4

You may also pass a text file with one video path per line:
python test_neoresus_videollava.py --model checkpoints/neoresus_lora --list testlist.txt

Dependencies
------------
- torch 2.2+
- transformers 4.52.3+
- av, numpy, peft, tqdm

"""

import argparse, json, re, sys, os, numpy as np, av, torch
from typing import List
from tqdm import tqdm
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

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
    """Uniformly sample exactly `num_frames` RGB24 frames from an mp4."""
    container = av.open(path)
    total = container.streams.video[0].frames
    if total < num_frames:
        raise RuntimeError(f"{path}: expected ≥{num_frames} frames, found {total}")
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)


def parse_json_from_text(txt: str):
    """Extract the first JSON object appearing in `txt`. Return dict or None."""
    m = re.search(r"\{.*?\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def enforce_rules(js: dict):
    """Post‑hoc constraint solver as described in the paper."""
    def _no():
        return {"baby_visible": "no", "ventilation": "no", "stimulation": "no", "suction": "no"}

    if not js:
        return _no()

    # Normalise values to lowercase yes/no
    js = {k: (str(v).lower()) for k, v in js.items()}
    yes = lambda v: v in {"yes", "true", "1"}

    if not yes(js.get("baby_visible", "no")):
        return _no()

    # Make mutually exclusive: ventilation vs suction
    if yes(js.get("ventilation", "no")) and yes(js.get("suction", "no")):
        js["suction"] = "no"  # keep ventilation by default
    # Ensure keys exist
    for k in ("ventilation", "stimulation", "suction"):
        js.setdefault(k, "no")
    return js


def classify_clip(model, processor, video_path: str):
    frames = sample_frames(video_path)
    inputs = processor(text=PROMPT_TEMPLATE, videos=frames, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    js = parse_json_from_text(out)
    js = enforce_rules(js)
    return js, out


def load_model(model_ckpt: str):
    processor = VideoLlavaProcessor.from_pretrained(model_ckpt)
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_ckpt, device_map="auto")
    model.eval()
    return model, processor


def main(argv: List[str]):
    ap = argparse.ArgumentParser(description="Inference tester for Video‑LLaVA neonatal resuscitation model")
    ap.add_argument("--model", required=True, help="Checkpoint name or path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Path to a single .mp4 video")
    g.add_argument("--list", help="Text file with one video path per line")
    ap.add_argument("--no-rules", action="store_true", help="Skip rule enforcement post‑processing")
    args = ap.parse_args(argv)

    model, processor = load_model(args.model)

    videos = [args.video] if args.video else [v.strip() for v in open(args.list)]

    for vp in tqdm(videos, desc="videos"):
        try:
            js, raw = classify_clip(model, processor, vp)
            if args.no_rules:
                js = parse_json_from_text(raw)
            print(json.dumps({"video": vp, "pred": js}, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"video": vp, "error": str(e)}), file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1:])
