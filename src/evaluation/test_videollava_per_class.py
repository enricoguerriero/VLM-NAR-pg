#!/usr/bin/env python3
"""
test_video_llava_multilabel.py

Zero-shot evaluation script for **Video-LLaVA** on a multi-label video
classification task (binary labels per class).

Expected input
--------------
1. A JSONL file where each line looks like:
       {
         "video": "<path/to/video>",
         "labels": {"baby_visible": 1, "ventilation": 0, ...}
       }
2. (Optional) a JSON file mapping each label to a custom textual prompt.

What the script does
--------------------
• Loads the Video-LLaVA model and processor (HF hub id overridable).  
• Uniformly samples *N* RGB frames with PyAV (defaults to 8, as required by
  the model). Missing frames are padded with the last available frame.  
• Asks the model each label-specific yes/no question, **forcing it to answer
  only “0” or “1”.**  
• Aggregates predictions, printing accuracy, precision, recall and F1
  (per-class and macro) via scikit-learn.

Example
-------
python test_video_llava_multilabel.py \
       --jsonl data/clips/test.jsonl \
       --model LanguageBind/Video-LLaVA-7B-hf \
       --batch-size 1 \
       --num-frames 8
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import av                      # pip install av
import numpy as np             # pip install numpy
import torch                   # pip install torch --index-url https://download.pytorch.org/whl/cu121
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)                              # pip install scikit-learn
from tqdm.auto import tqdm     # pip install tqdm
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
)


# ---------- Video helpers -------------------------------------------------- #
def _read_video_pyav(filepath: str, num_frames: int = 8) -> np.ndarray:
    """
    Decode *num_frames* RGB frames, sampled uniformly across the clip.

    Video-LLaVA was trained with 8-frame inputs; if you choose a different
    --num-frames the script will still work, but 8 is recommended.
    """
    container = av.open(filepath)
    total = container.streams.video[0].frames
    # robust against <num_frames videos too
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    # Pad short clips with last frame
    if len(frames) != num_frames and len(frames) > 0:
        frames.extend(frames[-1:] * (num_frames - len(frames)))
    return np.stack(frames)  # (T, H, W, 3)


# ---------- Prompt helpers ------------------------------------------------- #
def _default_prompt(label: str) -> str:
    natural = label.replace("_", " ")
    return (
        f"Does this clip show {natural}? "
        "Reply with **1** for Yes, **0** for No – just the digit."
    )


def build_conversation(prompt: str) -> str:
    """
    Build a single-turn multimodal chat in Video-LLaVA format.

    apply_chat_template then converts this to:
        USER: <video>\n<prompt> ASSISTANT:
    """
    return "USER: <video>\n" + prompt + " ASSISTANT:"


# ---------- Single prompt inference --------------------------------------- #
@torch.inference_mode()
def predict_binary(
    model: VideoLlavaForConditionalGeneration,
    processor: VideoLlavaProcessor,
    video: np.ndarray,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> int:
    conv = build_conversation(prompt)
    # prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=conv, videos=video, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    print(f"Response: {decoded}")
    answer = decoded.split("ASSISTANT:")[-1].strip()
    # Return the first explicit 0/1 the model outputs
    for ch in answer:
        if ch in {"0", "1"}:
            return int(ch)
    return 0  # fallback if the model refuses / hallucinates


# ---------- Main ----------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot multi-label evaluation with Video-LLaVA."
    )
    parser.add_argument("--jsonl", required=True, help="Path to dataset jsonl.")
    parser.add_argument(
        "--model",
        default="LanguageBind/Video-LLaVA-7B-hf",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Reserved for future batching (currently 1).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Frames sampled per clip (8 recommended).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Optional JSON mapping label → prompt.",
    )
    parser.add_argument(
        "--half", action="store_true", help="Load model in fp16 (saves memory)."
    )
    args = parser.parse_args()

    # ---------- Load dataset ---------------------------------------------- #
    with open(args.jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    label_names = list(samples[0]["labels"].keys())

    # ---------- Build prompts --------------------------------------------- #

    prompts = {
        "baby_visible": (
            "Is there a baby doll on the table? 1 for Yes, 0 for No. "
            # "You are in a simulation of a newborn resuscitation. The camera is on "
            # "a table, where there can or cannot be a baby or a mannequin representing "
            # "a baby. If you see a subject representing a baby on the table, reply "
            # "with 1; if the table is empty reply with 0. Be sure to only reply "
            # "with 0 or 1, nothing else."
        ),
        "ventilation": (
            "You are in a simulation of a newborn resuscitation. The camera is on "
            "a table, where there can or cannot be a baby or a mannequin. If the "
            "baby/mannequin receives ventilation via a mask, reply with 1; "
            "otherwise reply with 0 (also reply 0 if no baby is visible). "
            "Answer only 0 or 1."
        ),
        "stimulation": (
            "You are in a simulation of a newborn resuscitation. The camera is on "
            "a table, where there can or cannot be a baby or a mannequin. If the "
            "baby/mannequin receives stimulation (rubbing the back, nates, or trunk), "
            "reply 1; otherwise 0 (or 0 if no baby visible). Answer only 0 or 1."
        ),
        "suction": (
            "You are in a simulation of a newborn resuscitation. The camera is on "
            "a table, where there can or cannot be a baby or a mannequin. If the "
            "baby/mannequin receives suctioning with a catheter, reply 1; otherwise "
            "0 (or 0 if no baby visible). Answer only 0 or 1."
        ),
    }

    # ---------- Model & processor ----------------------------------------- #
    dtype = torch.float16 if args.half else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    processor = VideoLlavaProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model.eval()

    # ---------- Prediction loop ------------------------------------------- #
    y_true = {lbl: [] for lbl in label_names}
    y_pred = {lbl: [] for lbl in label_names}

    for sample in tqdm(samples, desc="Evaluating", unit="clip"):
        print(f"\n[VIDEO] {sample['video']}", flush=True)
        video = _read_video_pyav(sample["video"], args.num_frames)

        for lbl in label_names:
            pred = predict_binary(model, processor, video, prompts[lbl], device)
            gt = sample["labels"][lbl]
            y_pred[lbl].append(pred)
            y_true[lbl].append(gt)
            print(f"  {lbl:15s} | GT: {gt} | Pred: {pred}", flush=True)

    # ---------- Metrics --------------------------------------------------- #
    print("\n===== Per-class metrics =====", flush=True)
    macro_prec, macro_rec, macro_f1 = [], [], []
    for lbl in label_names:
        acc = accuracy_score(y_true[lbl], y_pred[lbl])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[lbl], y_pred[lbl], average="binary", zero_division=0
        )
        macro_prec.append(prec)
        macro_rec.append(rec)
        macro_f1.append(f1)
        print(
            f"{lbl:15s}  acc={acc:5.3f}  prec={prec:5.3f}  "
            f"rec={rec:5.3f}  f1={f1:5.3f}",
            flush=True,
        )

    print("\n===== Macro-average =====", flush=True)
    print(
        f"acc={np.mean([accuracy_score(y_true[l], y_pred[l]) for l in label_names]):5.3f}  "
        f"prec={np.mean(macro_prec):5.3f}  rec={np.mean(macro_rec):5.3f}  "
        f"f1={np.mean(macro_f1):5.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
