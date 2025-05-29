#!/usr/bin/env python3
"""test_llava_next_video_multilabel.py

Zero‑shot evaluation script for LLaVA‑NeXT‑Video on a multi‑label video classification task.

The script expects:
    1. A JSONL file where each line contains:
           {
             "video": "<path/to/video>",
             "labels": {"baby_visible": 1, "ventilation": 0, "stimulation": 0, "suction": 0}
           }
    2. A list of label‑specific textual prompts.  If not provided, defaults are generated.

It will:
    • Load the model / processor (HF hub id overridable).
    • Sample N frames uniformly with PyAV.
    • Ask the model each prompt, forcing it to answer **only `0` or `1`**.
    • Aggregate predictions and compute accuracy, precision, recall and F1
      (per‑class and macro) using scikit‑learn.

Example
--------
python test_llava_next_video_multilabel.py \
       --jsonl data/clips/test.jsonl \
       --model llava-hf/LLaVA-NeXT-Video-7B-hf \
       --batch-size 1 \
       --num-frames 8
"""

import json
import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import av                      # pip install av
import numpy as np             # pip install numpy
import torch                   # pip install torch --index-url https://download.pytorch.org/whl/cu121
from tqdm.auto import tqdm     # pip install tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support   # pip install scikit-learn
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)


# ---------- Video helpers -------------------------------------------------- #

def _read_video_pyav(filepath: str, num_frames: int = 8) -> np.ndarray:
    """Decode *num_frames* RGB frames, uniformly sampled across the clip."""
    container = av.open(filepath)
    total = container.streams.video[0].frames
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != num_frames:     # Pad if video too short
        frames.extend(frames[-1:] * (num_frames - len(frames)))
    return np.stack(frames)            # (T, H, W, 3)


# ---------- Prompt helpers ------------------------------------------------- #

def _default_prompt(label: str) -> str:
    natural = label.replace("_", " ")
    return (
        f"Does this clip show {natural}? "  # question
        "Reply with **1** for Yes, **0** for No – just the digit."  # force format
    )


def build_conversation(prompt: str) -> List[Dict]:
    """Wrap a single turn + video into LLaVA format."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        }
    ]


# ---------- Prediction per clip/label ------------------------------------- #
@torch.inference_mode()

def predict_binary(
    model,
    processor,
    video: np.ndarray,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 4,
) -> int:
    conv = build_conversation(prompt)
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt_text, videos=video, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # Find first explicit 0/1
    for ch in decoded:
        if ch in {"0", "1"}:
            return int(ch)
    return 0  # fallback / model refused


# ---------- Main ----------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Zero‑shot multi‑label evaluation with LLaVA‑NeXT‑Video.")
    parser.add_argument("--jsonl", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--model", default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="HF model id or local path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching (currently 1).")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled per clip.")
    parser.add_argument("--prompts", type=str, default=None, help="JSON file mapping label->prompt.")
    parser.add_argument("--half", action="store_true", help="Load model in fp16.")
    args = parser.parse_args()

    # Load dataset
    with open(args.jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Extract label set
    label_names = list(samples[0]["labels"].keys())

    # Build prompts
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as fp:
            custom = json.load(fp)
        prompts = {lbl: custom.get(lbl, _default_prompt(lbl)) for lbl in label_names}
    else:
        prompts = {lbl: _default_prompt(lbl) for lbl in label_names}

    # Model / processor
    dtype = torch.float16 if args.half else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    processor = LlavaNextVideoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model.eval()

    # Prediction loop -------------------------------------------------------
    y_true = {lbl: [] for lbl in label_names}
    y_pred = {lbl: [] for lbl in label_names}

    for sample in tqdm(samples, desc="Evaluating", unit="clip"):
        video = _read_video_pyav(sample["video"], args.num_frames)

        for lbl in label_names:
            pred = predict_binary(model, processor, video, prompts[lbl], device)
            y_pred[lbl].append(pred)
            y_true[lbl].append(sample["labels"][lbl])

    # Metrics ---------------------------------------------------------------
    print("\n===== Per‑class metrics =====")
    macro_prec, macro_rec, macro_f1 = [], [], []
    for lbl in label_names:
        acc = accuracy_score(y_true[lbl], y_pred[lbl])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[lbl], y_pred[lbl], average="binary", zero_division=0
        )
        macro_prec.append(prec)
        macro_rec.append(rec)
        macro_f1.append(f1)
        print(f"{lbl:15s}  acc={acc:5.3f}  prec={prec:5.3f}  rec={rec:5.3f}  f1={f1:5.3f}")

    print("\n===== Macro‑average =====")
    print(
        f"acc={np.mean([accuracy_score(y_true[l], y_pred[l]) for l in label_names]):5.3f}  "  # over labels
        f"prec={np.mean(macro_prec):5.3f}  rec={np.mean(macro_rec):5.3f}  f1={np.mean(macro_f1):5.3f}"
    )


if __name__ == "__main__":
    main()
