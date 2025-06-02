# train_lora_llava_video.py
"""
Fine‑tune LLaVA‑NeXT with LoRA for multi‑label video classification
===================================================================

Dataset
-------
* ``train.jsonl``, ``val.jsonl`` and ``test.jsonl`` with the format::
    {"video": "path/to/file.avi", "labels": {"baby_visible": 1, "ventilation": 1, "stimulation": 0, "suction": 0}}

Classes are **hard‑coded** as ``["baby_visible", "ventilation", "stimulation", "suction"]`` and **order matters**.

Main steps
~~~~~~~~~~
1. Frame sampling (default 8 uniformly spaced frames) & prompt construction
2. Processor → model (``LLaVA‑NeXT``) forward pass
3. Mean‑pool **only the image/vision tokens** from the last hidden layer
4. One‑layer classifier head (``Linear(hidden, 4)``)
5. Weighted binary‑cross‑entropy with pos‑weights computed from train set
6. Metrics (per‑class + macro Accuracy, Precision, Recall, F1) logged to **Weights & Biases**
7. Toggle: ``--train_classifier_only`` to freeze everything except classifier, or jointly train LoRA + classifier

Run example
~~~~~~~~~~~
```bash
python train_lora_llava_video.py \
    --train_json data/clips/train.jsonl \
    --val_json   data/clips/val.jsonl   \
    --test_json  data/clips/test.jsonl  \
    --output_dir runs/exp1 \
    --model_id  llava-hf/llava-next-large-430k \
    --batch_size 2 --epochs 3 \
    --caption_prompt "Describe the video briefly." \
    --train_classifier_only false
```
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig, get_peft_model

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

# ------------------------
# 1.  CONSTANTS & HELPERS
# ------------------------
CLASSES = ["baby_visible", "ventilation", "stimulation", "suction"]
NUM_LABELS = len(CLASSES)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def uniform_frame_indices(num_frames: int, num_sampled: int) -> List[int]:
    """Return <num_sampled> indices uniformly spaced across <num_frames>."""
    if num_frames <= num_sampled:
        return list(range(num_frames))
    interval = num_frames / num_sampled
    return [int(i * interval) for i in range(num_sampled)]


def collate_fn(batch: List[Dict]):
    # batch: list of dicts returned by VideoJsonlDataset.__getitem__
    pixel_values_videos = torch.stack([item["pixel_values_videos"] for item in batch])  # (B, F, C, H, W)
    input_ids = torch.stack([item["input_ids"] for item in batch])        # (B, T)
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch]).float()
    video_token_mask = torch.stack([item["video_token_mask"] for item in batch])  # (B, T)
    return {
        "pixel_values_videos": pixel_values_videos,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "video_token_mask": video_token_mask,
    }


# ------------------------
# 2.  DATASET
# ------------------------
class VideoJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor: LlavaNextProcessor,
        prompt_template: str,
        num_frames: int = 8,
        max_length: int = 128,
    ):
        super().__init__()
        self.records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
        self.processor = processor
        self.prompt_template = prompt_template
        self.num_frames = num_frames
        self.max_length = max_length
        # Pre‑compute index of image placeholder token once
        self.image_token = self.processor.tokenizer.video_token_id

    def __len__(self):
        return len(self.records)

    def _read_frames(self, video_path: str) -> List[torch.Tensor]:
        video, _, _ = read_video(video_path, pts_unit="sec")  # shape: (T, H, W, C)
        indices = uniform_frame_indices(len(video), self.num_frames)
        frames = video[indices]  
        return frames

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = self._read_frames(rec["video"])  # list of tensors

        # Prompt
        prompt = self.prompt_template
        prompt_text = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        processed = self.processor(text=prompt_text, videos=frames, return_tensors="pt")
        print("Processor output keys:", processed.keys(), flush=True)
        # processor returns dict with pixel_values (F, C, H, W) & tokenized text
        pixel_values_videos = processed["pixel_values_videos"]  # (F, C, H, W)
        input_ids = processed["input_ids"].squeeze(0)  # (T,)
        attention_mask = processed["attention_mask"].squeeze(0)  # (T,)

        # Identify vision tokens (== image_token_id)
        video_token_mask = (input_ids == self.video_token).long()

        label_vec = torch.tensor([rec["labels"][c] for c in CLASSES], dtype=torch.float)

        return {
            "pixel_values_videos": pixel_values_videos,  # (F, C, H, W)
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_vec,
            "video_token_mask": video_token_mask,
        }


# ------------------------
# 3.  MODEL WRAPPER
# ------------------------
class LlavaVideoClassifier(nn.Module):
    def __init__(self, backbone: LlavaNextForConditionalGeneration, hidden_size: int, num_labels: int, train_classifier_only: bool):
        super().__init__()
        self.backbone = backbone  # with (optionally) LoRA
        self.classifier = nn.Linear(hidden_size, num_labels)
        if train_classifier_only:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
        else:
            for n, p in self.backbone.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False

    @torch.no_grad()
    def _get_video_token_indices(self, video_token_mask):
        # img_token_mask: (B, T) binary mask, 1 where image token
        return video_token_mask.bool()

    def forward(self, pixel_values_videos, input_ids, attention_mask, video_token_mask):
        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos,  # (B, F, C, H, W)
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, T, hidden)
        # Mean‑pool over vision tokens only
        vision_mask = self._get_video_token_indices(video_token_mask)
        vision_mask_exp = vision_mask.unsqueeze(-1).expand_as(last_hidden)  # (B, T, hidden)
        summed = (last_hidden * vision_mask_exp).sum(dim=1)
        counts = vision_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        vision_feat = summed / counts  # (B, hidden)

        logits = self.classifier(vision_feat)  # (B, num_labels)
        return logits


# ------------------------
# 4.  TRAIN / EVAL UTILS
# ------------------------

def compute_pos_weights(loader: DataLoader, device: str = "cpu") -> torch.Tensor:
    """Compute positive class weights (neg/pos) over an entire DataLoader."""
    total = torch.zeros(NUM_LABELS, device=device)
    pos = torch.zeros(NUM_LABELS, device=device)
    for batch in loader:
        labels = batch["labels"].to(device)
        total += labels.shape[0]
        pos += labels.sum(dim=0)
    neg = total - pos
    pos_weight = neg / pos.clamp(min=1)
    return pos_weight


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray):
    """Return dict of class‑wise and macro precision/recall/f1/accuracy."""
    assert y_true.shape == y_pred.shape
    y_true_bin = y_true > 0.5
    y_pred_bin = y_pred > 0.5

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    acc_m = accuracy_score(y_true_bin, y_pred_bin)

    metrics = {f"{CLASSES[i]}/precision": prec[i] for i in range(NUM_LABELS)}
    metrics.update({f"{CLASSES[i]}/recall": rec[i] for i in range(NUM_LABELS)})
    metrics.update({f"{CLASSES[i]}/f1": f1[i] for i in range(NUM_LABELS)})
    metrics.update({
        "macro/precision": prec_m,
        "macro/recall": rec_m,
        "macro/f1": f1_m,
        "macro/accuracy": acc_m,
    })
    return metrics

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


# ------------------------
# 5.  MAIN LOOP
# ------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    caption_prompt = "You are in a simulation of a neonatal resuscitation scenario. Give a caption for the video; be explicit about:\n-Who is present in the video.\n-What is happening in the video.\n-What actions are being performed.\n-What equipment is being used."
    # WandB init
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    prompt = build_conversation(caption_prompt)

    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    backbone = LlavaNextForConditionalGeneration.from_pretrained(args.model_id)

    # Apply LoRA unless train_classifier_only
    if not args.train_classifier_only:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, lora_cfg)

    model = LlavaVideoClassifier(
        backbone=backbone,
        hidden_size=backbone.config.text_config.hidden_size,
        num_labels=NUM_LABELS,
        train_classifier_only=args.train_classifier_only,
    ).to(device)

    # Datasets & loaders
    train_ds = VideoJsonlDataset(args.train_json, processor, prompt, num_frames=args.num_frames)
    val_ds   = VideoJsonlDataset(args.val_json,   processor, prompt, num_frames=args.num_frames)
    test_ds  = VideoJsonlDataset(args.test_json,  processor, prompt, num_frames=args.num_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Loss with pos weights
    pos_weight = compute_pos_weights(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            pixel_values_videos = batch["pixel_values_videos"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            video_mask = batch["video_token_mask"].to(device)
            labels = batch["labels"].to(device)

            optim.zero_grad()
            logits = model(pixel_values_videos, input_ids, attn, video_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)

        # ----- Validation -----
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values_videos = batch["pixel_values_videos"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                video_mask = batch["video_token_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(pixel_values_videos, input_ids, attn, video_mask)
                all_logits.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

        y_pred = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_labels).numpy()
        val_metrics = metrics_from_preds(y_true, y_pred)
        val_metrics["loss"] = train_loss
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

        # Save best
        if val_metrics["macro/f1"] > best_f1:
            best_f1 = val_metrics["macro/f1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))
            wandb.run.summary["best_val_f1"] = best_f1

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_macro_f1={val_metrics['macro/f1']:.4f}")

    # ----- Test -----
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.pt")))
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values_videos = batch["pixel_values_videos"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            video_mask = batch["video_token_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(pixel_values_videos, input_ids, attn, video_mask)
            all_logits.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    test_metrics = metrics_from_preds(y_true, y_pred)
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    wandb.run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    print("Test results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


# ------------------------
# 6.  ARGPARSE
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‑tune LLaVA‑NeXT with LoRA for video multi‑label classification")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json",   type=str, required=True)
    parser.add_argument("--test_json",  type=str, required=True)
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-next-large-430k")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_frames", type=int, default=8, help="Frames sampled per video")
    parser.add_argument("--train_classifier_only", action="store_true", help="Freeze backbone & LoRA; train only classifier")

    # LoRA params
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="llava-video")
    parser.add_argument("--run_name", type=str, default="exp")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
