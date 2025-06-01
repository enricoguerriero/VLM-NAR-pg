#!/usr/bin/env python3
"""test_llava_next_video_multilabel_single_prompt.py

Zero‑shot evaluation script for LLaVA‑NeXT‑Video on a multi‑label video classification task,
using **one single prompt per clip** that requests the model to output all labels at once
in a strict JSON dictionary format.

Expected dataset format (same as the binary‑per‑label variant):
    Each line in the JSONL input file contains:
        {
          "video": "<path/to/video>",
          "labels": {"baby_visible": 1, "ventilation": 0, "stimulation": 0, "suction": 0}
        }

Differences from the original script
------------------------------------
* Exactly one model call per clip.
* The prompt forces the assistant to output **only** a JSON object like:
      {"baby_visible":1,"ventilation":0,"stimulation":0,"suction":1}
  – no spaces, newlines or additional text.
* Robust parsing that tolerates minor formatting drift (e.g. spaces/newlines).

Example
-------
python test_llava_next_video_multilabel_single_prompt.py \
       --jsonl data/clips/test.jsonl \
       --model llava-hf/LLaVA-NeXT-Video-7B-hf \
       --batch-size 1 \
       --num-frames 8
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import av                      
import numpy as np             
import torch                   
from tqdm.auto import tqdm     
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

def build_prompt(label_names: List[str]) -> str:
    """Return a fixed instruction asking for all labels at once in JSON format."""

    natural_labels = [lbl.replace("_", " ") for lbl in label_names]
    joined = ", ".join(natural_labels)

    fmt_example = "{" + ",".join([f'\"{lbl}\":1' for lbl in label_names]) + "}"

    return (
        "You will be shown a short video clip. For each of the following items — "
        f"{joined} — decide whether it is present (1) or absent (0).\n\n"
        "Reply with **only** a JSON dictionary _without spaces or newlines_ whose keys are the label names "
        "and whose values are 0 or 1.\n\n"
        "Example of required format (order must match exactly):\n"
        f"{fmt_example}\n\n"
        "Do not output anything except that JSON object."
    )


def build_conversation(prompt: str) -> List[Dict]:
    """Wrap the single turn + video into LLaVA chat template."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        }
    ]

# ---------- Prediction per clip ------------------------------------------- #
@torch.inference_mode()
def predict_multilabel(
    model,
    processor,
    video: np.ndarray,
    prompt: str,
    label_names: List[str],
    device: torch.device,
    max_new_tokens: int = 50,
) -> Dict[str, int]:
    """Return dict(label -> 0/1) after one model call."""
    conv = build_conversation(prompt)
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt_text, videos=video, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(f"Response: {decoded}")

    # Extract JSON substring – find first '{' and last '}'.
    try:
        start = decoded.index("{")
        end = decoded.rindex("}") + 1
        json_str = decoded[start:end]
        pred_dict = json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        # Fallback: assume first |label_names| digits appear in correct order
        digits = [c for c in decoded if c in "01"][: len(label_names)]
        pred_dict = {lbl: int(digits[i]) if i < len(digits) else 0 for i, lbl in enumerate(label_names)}

    # Ensure all keys present and cast to int 0/1
    for lbl in label_names:
        pred_dict[lbl] = int(bool(pred_dict.get(lbl, 0)))
    return pred_dict


# ---------- Main ----------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Zero‑shot multi‑label evaluation with a single prompt and forced JSON output."
    )
    parser.add_argument("--jsonl", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--model", default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="HF model id or local path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching (currently 1).")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled per clip.")
    parser.add_argument("--half", action="store_true", help="Load model in fp16.")
    args = parser.parse_args()

    # Load dataset
    with open(args.jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    label_names = list(samples[0]["labels"].keys())
    prompt = build_prompt(label_names)
    # prompt = "You are seeing a video clip of a simulation of a newborn resuscitation. You are asked to identify the presence of certain actions in the video."

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
        print(f"\n[VIDEO] {sample['video']}", flush=True)
        video = _read_video_pyav(sample["video"], args.num_frames)

        preds = predict_multilabel(model, processor, video, prompt, label_names, device)
        for lbl in label_names:
            y_true[lbl].append(sample["labels"][lbl])
            y_pred[lbl].append(preds[lbl])
            print(f"  {lbl:15s} | GT: {sample['labels'][lbl]} | Pred: {preds[lbl]}", flush=True)

    # Metrics ---------------------------------------------------------------
    print("\n===== Per‑class metrics =====", flush=True)
    macro_prec, macro_rec, macro_f1 = [], [], []
    for lbl in label_names:
        acc = accuracy_score(y_true[lbl], y_pred[lbl])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[lbl], y_pred[lbl], average="binary", zero_division=0
        )
        macro_prec.append(prec)
        macro_rec.append(rec)
        macro_f1.append(f1)
        print(f"{lbl:15s}  acc={acc:5.3f}  prec={prec:5.3f}  rec={rec:5.3f}  f1={f1:5.3f}", flush=True)

    print("\n===== Macro‑average =====", flush=True)
    print(
        f"acc={np.mean([accuracy_score(y_true[l], y_pred[l]) for l in label_names]):5.3f}  "  # over labels
        f"prec={np.mean(macro_prec):5.3f}  rec={np.mean(macro_rec):5.3f}  f1={np.mean(macro_f1):5.3f}", flush=True
    )


if __name__ == "__main__":
    main()
