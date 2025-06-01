#!/usr/bin/env python3
"""test_llava_next_video_multilabel_list_prompt.py

Zero‑shot evaluation script for LLaVA‑NeXT‑Video on a multi‑label video classification task
using **one natural‑language list prompt per clip**.

The model is asked to output *only* the names of the actions that are present, separated by
commas, e.g. ::

    Baby visible, Ventilation

If no action appears, the model must reply with exactly ``none``.

The script then converts that list into binary labels and computes accuracy, precision,
recall and F‑score – just like the previous JSON version.

Dataset format
--------------
Identical to the other scripts – each line of the input JSONL must look like::

    {"video": "path/to/clip.avi",
     "labels": {"baby_visible": 1, "ventilation": 0, "stimulation": 0, "suction": 0}}

Example
-------
::

    python test_llava_next_video_multilabel_list_prompt.py \
           --jsonl data/clips/test.jsonl \
           --model llava-hf/LLaVA-NeXT-Video-7B-hf \
           --num-frames 8
"""

import argparse
import json
from typing import Dict, List, Set

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

def _natural_name(label: str) -> str:
    """Convert snake_case to capitalised natural language ("baby_visible" -> "Baby visible")."""
    words = label.split("_")
    return " ".join([words[0].capitalize()] + words[1:])


def build_prompt(label_names: List[str]) -> str:
    """Return the single instruction prompt."""
    natural = [_natural_name(lbl) for lbl in label_names]
    joined = ", ".join(natural)
    return (
        "You will be shown a short video clip of a simulation of a Newborn resuscitation."
        "Decide which of the following actions are present: "
        f"{joined}.\n\n"
        "Keep in mind that there are dependecies between actions:\n"
        "- If the baby is not visible, then no other action can be present.\n"
        "- Stimulation can occurr together with ventilation or suction. \n"
        "- Ventilation and suction can not happen at the same time.\n"
        "- All the actions can occur if and only if the baby is visible.\n\n"
        "Based on what you see in the video, reply with a comma‑separated list containing **only** the names of the actions you see, "
        "using exactly the spellings given above (capitalisation may vary). If none of them appear, "
        "reply with the single word `none`. Do not add any other text."
    )


def build_conversation(prompt: str) -> List[Dict]:
    """Wrap user prompt + video into LLaVA chat format."""
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
    max_new_tokens: int = 30,
) -> Dict[str, int]:
    """Return dict(label -> 0/1) from a single model call."""

    conv = build_conversation(prompt)
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt_text, videos=video, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,            # greedy decoding for determinism
        temperature=0.2,
    )
    decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(f"Response: {decoded}")

    # Extract assistant reply (everything after last 'ASSISTANT:')
    reply = decoded.split("ASSISTANT:")[-1].strip()

    # Normalise reply: lower‑case, strip spaces around commas
    reply_norm = reply.lower().replace(",", ", ")  # ensure split works
    tokens = [t.strip() for t in reply_norm.split(",") if t.strip()]

    present: Set[str]
    if tokens == ["none"]:
        present = set()
    else:
        # Map every token back to snake_case label if possible
        mapping = { _natural_name(lbl).lower(): lbl for lbl in label_names }
        present = { mapping[token] for token in tokens if token in mapping }

    # Build binary dict
    return { lbl: int(lbl in present) for lbl in label_names }


# ---------- Main ----------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Zero‑shot multi‑label evaluation with a single list prompt and minimal output."
    )
    parser.add_argument("--jsonl", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--model", default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="HF model id or local path.")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled per clip.")
    parser.add_argument("--half", action="store_true", help="Load model in fp16 (saves GPU memory).")
    args = parser.parse_args()

    # Load dataset
    with open(args.jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    label_names = list(samples[0]["labels"].keys())
    prompt = build_prompt(label_names)

    # Model / processor
    dtype = torch.float16 if args.half else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
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
        f"acc={np.mean([accuracy_score(y_true[l], y_pred[l]) for l in label_names]):5.3f}  "
        f"prec={np.mean(macro_prec):5.3f}  rec={np.mean(macro_rec):5.3f}  f1={np.mean(macro_f1):5.3f}", flush=True
    )


if __name__ == "__main__":
    main()
