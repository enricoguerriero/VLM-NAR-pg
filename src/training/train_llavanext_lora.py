import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging
from peft import LoraConfig, TaskType, get_peft_model

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# --------------------------- Constants --------------------------- #
LABEL_LIST = ["baby_visible", "ventilation", "stimulation", "suction"]
LABEL2ID: Dict[str, int] = {n: i for i, n in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: n for n, i in LABEL2ID.items()}
logger.info("Using hard‑coded label mapping: %s", LABEL2ID)

# --------------------------- Data helpers --------------------------- #

def sample_frames(path: str, num_frames: int = 8):
    """Uniformly sample *num_frames* RGB frames with Decord."""
    import decord

    vr = decord.VideoReader(str(path))
    total = len(vr)
    idx = (
        np.linspace(0, total - 1, num=num_frames, dtype=int)
        if total >= num_frames
        else np.pad(np.arange(total), (0, num_frames - total), mode="edge")
    )
    return vr.get_batch(idx).asnumpy()


class VideoDataset(torch.utils.data.Dataset):
    """Dataset wrapper that converts the `labels` dict into a fixed‑order vector."""

    def __init__(self, hf_ds: Dataset, processor, num_frames: int = 8):
        self.ds = hf_ds
        self.processor = processor
        self.num_frames = num_frames
        self.num_labels = len(LABEL_LIST)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        vid = sample_frames(ex["video"], self.num_frames)
        lbl_vec = np.zeros(self.num_labels, dtype=np.float32)
        for k, v in ex["labels"].items():
            if v and k in LABEL2ID:
                lbl_vec[LABEL2ID[k]] = 1
        return {"video": vid, "text": "Classify the video.", "labels": lbl_vec}


def collate_fn(batch, processor):
    vids = [b["video"] for b in batch]
    txts = [b["text"] for b in batch]
    y = torch.tensor([b["labels"] for b in batch])
    inp = processor(txts, videos=vids, padding=True, return_tensors="pt")
    inp["labels"] = y
    return inp

# --------------------------- Model --------------------------- #

class LlavaNextVideoForMultiLabel(nn.Module):
    def __init__(
        self,
        base_model: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: tuple = ("q_proj", "v_proj"),
    ):
        super().__init__()
        self.base = LlavaNextVideoForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        for p in self.base.parameters():
            p.requires_grad = False
        self.base = get_peft_model(
            self.base,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
        self.classifier = nn.Linear(self.base.config.text_config.hidden_size, len(LABEL_LIST))
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        out = self.base(**kwargs, output_hidden_states=True, use_cache=False)
        pooled = out.hidden_states[-1][:, 0]
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# --------------------------- Metrics --------------------------- #

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    m = {}
    for i, name in enumerate(LABEL_LIST):
        m[f"accuracy_{name}"] = accuracy_score(labels[:, i], preds[:, i])
        m[f"precision_{name}"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
        m[f"recall_{name}"] = recall_score(labels[:, i], preds[:, i], zero_division=0)
        m[f"f1_{name}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
    m["accuracy_macro"] = np.mean([m[f"accuracy_{n}"] for n in LABEL_LIST])
    m["precision_macro"] = precision_score(labels, preds, average="macro", zero_division=0)
    m["recall_macro"] = recall_score(labels, preds, average="macro", zero_division=0)
    m["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
    return {k: float(v) for k, v in m.items()}

# --------------------------- Main --------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    # Optional W&B
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=Path(args.output_dir).name, config=vars(args))
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    train_ds = load_dataset("json", data_files=str(Path(args.data_dir) / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(Path(args.data_dir) / "validation.jsonl"), split="train")

    processor = LlavaNextVideoProcessor.from_pretrained(args.model_name)
    processor.video_processor.do_center_crop = False
    processor.image_processor.do_center_crop = False  # safeguard
    train_set = VideoDataset(train_ds, processor, num_frames=args.num_frames)
    val_set = VideoDataset(val_ds, processor, num_frames=args.num_frames)

    model = LlavaNextVideoForMultiLabel(
        base_model=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.base.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=report_to,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=lambda b: collate_fn(b, processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
