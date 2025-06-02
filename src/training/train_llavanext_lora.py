import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import av
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, get_peft_model

import wandb
from tqdm import tqdm

# ------------------------
# 1.  CONSTANTS & HELPERS
# ------------------------
CLASSES = ["baby_visible", "ventilation", "stimulation", "suction"]
NUM_LABELS = len(CLASSES)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uniform_frame_indices(num_frames: int, num_sampled: int) -> List[int]:
    """Return <num_sampled> indices uniformly spaced across <num_frames>."""
    if num_frames <= num_sampled:
        return list(range(num_frames))
    interval = num_frames / num_sampled
    return [int(i * interval) for i in range(num_sampled)]


def collate_fn(batch: List[Dict]):
    # batch: list of dicts returned by VideoJsonlDataset.__getitem__
    pixel_values_videos = torch.cat([item["pixel_values_videos"] for item in batch])  # (B, F, C, H, W)
    input_ids = torch.cat([item["input_ids"] for item in batch])        # (B, T)
    attention_mask = torch.cat([item["attention_mask"] for item in batch])
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
        processor: LlavaNextVideoProcessor,
        prompt_template: List[Dict],
        num_frames: int = 8,
        max_length: int = 128,
    ):
        super().__init__()
        self.records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
        self.processor = processor
        self.prompt_template = prompt_template
        self.num_frames = num_frames
        self.max_length = max_length
        # Pre-compute index of image placeholder token once
        self.video_token = self.processor.tokenizer.video_token_id

    def __len__(self):
        return len(self.records)

    def _read_frames(self, filepath: str) -> np.ndarray:
        """Decode <num_frames> RGB frames, uniformly sampled across the clip."""
        container = av.open(filepath)
        total = container.streams.video[0].frames
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) != self.num_frames:     # Pad if video too short
            frames.extend(frames[-1:] * (self.num_frames - len(frames)))
        return np.stack(frames)            # (T, H, W, 3)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = self._read_frames(rec["video"])  # (T, H, W, 3)

        # Build processor input: chat-style prompt + video
        prompt_text = self.processor.apply_chat_template(self.prompt_template, add_generation_prompt=True)
        processed = self.processor(text=prompt_text, videos=frames, return_tensors="pt")

        pixel_values_videos = processed["pixel_values_videos"]  # (F, C, H, W)
        input_ids = processed["input_ids"]  # (T,)
        attention_mask = processed["attention_mask"]  # (T,)

        # Identify vision tokens (== image_token_id)
        video_token_mask = (input_ids == self.video_token).long()

        label_vec = torch.tensor([rec["labels"][c] for c in CLASSES], dtype=torch.float)

        return {
            "pixel_values_videos": pixel_values_videos,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_vec,
            "video_token_mask": video_token_mask,
        }


# ------------------------
# 3.  MODEL WRAPPER
# ------------------------
class LlavaVideoClassifier(nn.Module):
    def __init__(self, backbone: LlavaNextVideoForConditionalGeneration, hidden_size: int, num_labels: int, train_classifier_only: bool):
        super().__init__()
        self.backbone = backbone
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
        return video_token_mask.bool()

    def forward(self, pixel_values_videos, input_ids, attention_mask, video_token_mask):
        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos,  # (B, F, C, H, W)
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  
                    
        video_token_id = self.backbone.config.video_token_index
        
        video_mask = (input_ids == video_token_id).to(device)
        
        pooled_video = (last_hidden * video_mask.unsqueeze(-1)).sum(1) / \
               video_mask.sum(1, keepdim=True).clamp(min=1)# (B, hidden)

        logits = self.classifier(pooled_video.float())  # (B, num_labels)
        return logits


# ------------------------
# 4.  METRICS & TRAIN UTILITIES
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
    """Return dict of class-wise and macro precision/recall/f1/accuracy."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
# 5.  MAIN LOOP (TRAIN ONLY, WITH METRICS)
# ------------------------
def main(args):

    caption_prompt = (
        "You are in a simulation of a neonatal resuscitation scenario. "
        "Give a caption for the video; be explicit about:\n"
        "-Who is present in the video.\n"
        "-What is happening in the video.\n"
        "-What actions are being performed.\n"
        "-What equipment is being used."
    )
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    prompt = build_conversation(caption_prompt)

    processor = LlavaNextVideoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(args.model_id)

    # Apply LoRA unless training classifier only
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

    # Only train dataset & loader
    train_ds = VideoJsonlDataset(args.train_json, processor, prompt, num_frames=args.num_frames)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Compute class weights and set up loss/optimizer
    pos_weight = compute_pos_weights(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | Training...")
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", unit="batch"):
            pixel_values_videos = batch["pixel_values_videos"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            video_mask = batch["video_token_mask"].to(device)
            labels = batch["labels"].to(device)

            optim.zero_grad()
            logits = model(pixel_values_videos, input_ids, attn, video_mask)  # (B, num_labels)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        wandb.log({"train/loss": train_loss}, step=epoch)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f}")

        # ----- Compute metrics on the entire training set -----
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in train_loader:
                pixel_values_videos = batch["pixel_values_videos"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                video_mask = batch["video_token_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(pixel_values_videos, input_ids, attn, video_mask)
                probs = torch.sigmoid(logits)  # (B, num_labels)
                all_logits.append(probs.cpu())
                all_labels.append(labels.cpu())

        y_pred = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_labels).numpy()
        train_metrics = metrics_from_preds(y_true, y_pred)
        # Log each metric with prefix "train/"
        wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)

        # Print a summary to console
        summary = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items() if k.startswith("macro/")])
        print(f"  [Training metrics] {summary}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")


# ------------------------
# 6.  ARGPARSE
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaVA-NeXT classifier on video (train-only, with metrics).")
    parser.add_argument("--train_json", type=str, required=True, help="Path to training JSONL file")
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
    parser.add_argument("--run_name", type=str, default="train_only_metrics")

    args = parser.parse_args()

    main(args)
