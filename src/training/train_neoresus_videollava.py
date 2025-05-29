#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for Video-LLaVA on neonatal-resuscitation events.

Fixes & features compared with the original:
1. Stops the “No label_names provided …” warning by passing `label_names=["labels"]`
   to `TrainingArguments`.
2. Adds full W&B logging with accuracy, precision, recall and F1 for each of the
   four classes plus macro averages on *validation* (every epoch) and *test*.
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List

import av
import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data.dataloader import default_collate
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    TrainingArguments,
    Trainer,
)

import wandb  # noqa: F401 – only imported so the W&B callback can find it

# ---------------------------------------------------------------------------
# CONFIG (edit here)
# ---------------------------------------------------------------------------
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
MAX_NEW_TOKENS = 64
NUM_FRAMES = 8  # Video-LLaVA was trained with 8 frames
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

_FIELDS = ["baby_visible", "ventilation", "stimulation", "suction"]  # metric order
_YESNO = re.compile(r'"(yes|no)"', re.I)  # loose cleaner for JSON decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_frames(container: av.container.input.InputContainer, num_frames: int = 8) -> np.ndarray:
    """
    Uniformly sample `num_frames` RGB24 frames from a video and return an
    array of shape (T, H, W, 3).
    """
    total = container.streams.video[0].frames
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames: List[np.ndarray] = []

    container.seek(0)
    for i, f in enumerate(container.decode(video=0)):
        if i > idxs[-1]:
            break
        if i in idxs:
            frames.append(f.to_ndarray(format="rgb24"))

    if len(frames) != num_frames:
        raise RuntimeError("Could not decode enough frames from video.")
    return np.stack(frames)


def labels_to_json(lbl: Dict[str, int]) -> str:
    yesno = lambda b: "yes" if b else "no"
    return json.dumps(
        {
            "baby_visible": yesno(lbl["baby_visible"]),
            "ventilation": yesno(lbl["ventilation"]),
            "stimulation": yesno(lbl["stimulation"]),
            "suction": yesno(lbl["suction"]),
        }
    )


def load_split(path: str) -> Dataset:
    """Load a JSONL file into a 🤗 Dataset (single ‘train’ split)."""
    return load_dataset("json", data_files=path, split="train")


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
class VideoSFTCollator:
    """
    Prepares batches of (video, prompt) → labels for supervised fine-tuning.
    """

    def __init__(self, processor: VideoLlavaProcessor, num_frames: int):
        self.processor = processor
        self.num_frames = num_frames
        # Use left padding (recommended for causal LM fine-tuning)
        self.original_pad_side = processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, batch):
        videos, prompts, answers = [], [], []

        # ----- build prompt + ground-truth answer -----
        for item in batch:
            container = av.open(item["video"])
            frames = sample_frames(container, self.num_frames)
            videos.append(frames)

            prompts.append(PROMPT_TEMPLATE)
            answers.append(labels_to_json(item["labels"]))

        # ----- first pass: tokenise prompt only (to compute prompt length) -----
        prompt_ids = [
            self.processor.tokenizer(p).input_ids for p in prompts
        ]  # list of list[int]

        # ----- second pass: prompt + answer, build supervision labels -----
        inputs, attn_masks, lbls = [], [], []
        for i, (video, prompt, ans) in enumerate(zip(videos, prompts, answers)):
            full = f"{prompt} {ans}"
            enc = self.processor(text=full, videos=[video], return_tensors="pt")

            ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]

            lab = ids.clone()
            lab[: len(prompt_ids[i])] = -100  # ignore prompt tokens in loss

            inputs.append(ids)
            attn_masks.append(attn)
            lbls.append(lab)

        # ---- final batch dict ----
        return {
            "input_ids": torch.stack(inputs),
            "attention_mask": torch.stack(attn_masks),
            "labels": torch.stack(lbls),
            "videos": torch.stack([torch.from_numpy(v) for v in videos]),
        }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _tok2json(tokenizer, ids):
    """
    Decode token IDs → dict with the 4 yes/no fields.
    Returns `None` if the JSON cannot be parsed.
    """
    txt = tokenizer.decode(ids, skip_special_tokens=True)
    j0, j1 = txt.find("{"), txt.rfind("}")
    if j0 == -1 or j1 == -1:
        return None
    snip = txt[j0 : j1 + 1]
    try:
        return json.loads(snip)
    except json.JSONDecodeError:
        # Try loose cleaning (capitalisation, trailing tokens, …)
        try:
            return json.loads(_YESNO.sub(lambda m: f'"{m.group(1).lower()}"', snip))
        except json.JSONDecodeError:
            return None


def _dict2vec(d: Dict[str, str]) -> np.ndarray:
    """
    Convert the yes/no dict to a 4-element int vector.
    """
    return np.array([1 if d[k].lower() == "yes" else 0 for k in _FIELDS], dtype=np.int8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    train_json: str,
    valid_json: str,
    test_json: str,
    output_dir: str,
    batch_per_gpu: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    # ---------- model & processor ----------
    processor = VideoLlavaProcessor.from_pretrained(MODEL_ID)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.tie_weights()  # keep proj ↔ embed weights in sync

    # ---------- LoRA ----------
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # ---------- data ----------
    train_ds = load_split(train_json)
    valid_ds = load_split(valid_json)
    if os.path.exists(test_json):
        test_ds = load_split(test_json)
    else:
        test_ds = None

    collator = VideoSFTCollator(processor, NUM_FRAMES)

    # ---------- metrics ----------
    tok = processor.tokenizer

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids

        y_pred, y_true = [], []
        for p_ids, l_ids in zip(preds, labels):
            # Replace -100 (ignored positions) so we can decode ground truth
            l_ids = np.where(l_ids == -100, tok.pad_token_id, l_ids)

            p_dict = _tok2json(tok, p_ids)
            l_dict = _tok2json(tok, l_ids)

            if p_dict is None or l_dict is None:
                continue  # skip if JSON parse failed

            y_pred.append(_dict2vec(p_dict))
            y_true.append(_dict2vec(l_dict))

        if not y_true:
            return {}

        y_pred, y_true = np.vstack(y_pred), np.vstack(y_true)

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

        # macro
        metrics["macro_precision"] = float(np.mean(precs))
        metrics["macro_recall"] = float(np.mean(recs))
        metrics["macro_f1"] = float(np.mean(f1s))
        return metrics

    # ---------- training args ----------
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_per_gpu,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        report_to="wandb",
        run_name="videollava_neoresus",
        label_names=["labels"],  # suppresses Trainer warning
    )

    # ---------- trainer ----------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # ---------- train ----------
    trainer.train()

    # ---------- test split ----------
    if test_ds is not None:
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        wandb.log(test_metrics)

    # ---------- save ----------
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Video-LLaVA on neonatal-resuscitation event detection."
    )
    parser.add_argument("--train_json", type=str, default="data/clips/train.jsonl")
    parser.add_argument("--valid_json", type=str, default="data/clips/validation.jsonl")
    parser.add_argument("--test_json", type=str, default="data/clips/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="runs/videollava_neoresus")
    parser.add_argument("--batch_per_gpu", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    main(
        train_json=args.train_json,
        valid_json=args.valid_json,
        test_json=args.test_json,
        output_dir=args.output_dir,
        batch_per_gpu=args.batch_per_gpu,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
