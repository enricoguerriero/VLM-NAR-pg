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

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# ------------------------
# 1. CONSTANTS & HELPERS
# ------------------------
CLASSES = ["baby_visible", "ventilation", "stimulation", "suction"]
NUM_LABELS = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uniform_frame_indices(num_frames: int, num_sampled: int) -> List[int]:
    """Return <num_sampled> indices uniformly spaced across <num_frames>."""
    if num_frames <= num_sampled:
        return list(range(num_frames))
    interval = num_frames / num_sampled
    return [int(i * interval) for i in range(num_sampled)]


def collate_fn(batch: List[Dict]):
    """Collate function to batch video tensors and text inputs."""
    pixel_values_videos = torch.cat([item["pixel_values_videos"] for item in batch])  # (B, F, C, H, W)
    input_ids = torch.cat([item["input_ids"] for item in batch])        # (B, T)
    attention_mask = torch.cat([item["attention_mask"] for item in batch])  # (B, T)
    labels = torch.stack([item["labels"] for item in batch]).float()      # (B, num_labels)
    video_token_mask = torch.stack([item["video_token_mask"] for item in batch])  # (B, T)
    return {
        "pixel_values_videos": pixel_values_videos,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "video_token_mask": video_token_mask,
    }


# ------------------------
# 2. DATASET
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
        if len(frames) != self.num_frames:  # Pad if video too short
            frames.extend(frames[-1:] * (self.num_frames - len(frames)))
        return np.stack(frames)  # (T, H, W, 3)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = self._read_frames(rec["video"])  # (T, H, W, 3)

        # Build processor input: chat-style prompt + video token
        prompt_text = self.processor.apply_chat_template(self.prompt_template, add_generation_prompt=True)
        processed = self.processor(text=prompt_text, videos=frames, return_tensors="pt")

        pixel_values_videos = processed["pixel_values_videos"]  # (F, C, H, W)
        input_ids = processed["input_ids"].squeeze(0)  # (T,)
        attention_mask = processed["attention_mask"].squeeze(0)  # (T,)

        # Identify vision tokens (== video_token_id)
        video_token_mask = (input_ids == self.video_token).long()

        label_vec = torch.tensor([rec["labels"][c] for c in CLASSES], dtype=torch.float)

        return {
            "pixel_values_videos": pixel_values_videos.unsqueeze(0),  # add batch dim: (1, F, C, H, W)
            "input_ids": input_ids.unsqueeze(0),                       # (1, T)
            "attention_mask": attention_mask.unsqueeze(0),             # (1, T)
            "labels": label_vec,                                        # (num_labels,)
            "video_token_mask": video_token_mask.unsqueeze(0),         # (1, T)
        }


# ------------------------
# 3. MODEL WRAPPER
# ------------------------
class LlavaVideoClassifier(nn.Module):
    def __init__(self, backbone: LlavaNextVideoForConditionalGeneration, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)

    @torch.no_grad()
    def _get_video_token_indices(self, video_token_mask):
        return video_token_mask.bool()

    def forward(self, pixel_values_videos, input_ids, attention_mask, video_token_mask):
        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos,  # (B, F, C, H, W)
            input_ids=input_ids,                      # (B, T)
            attention_mask=attention_mask,            # (B, T)
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, T, hidden_size)

        video_token_id = self.backbone.config.video_token_index
        video_mask = (input_ids == video_token_id).to(device)  # (B, T)

        # Pool over video tokens
        pooled_video = (last_hidden * video_mask.unsqueeze(-1)).sum(1) / \
                       video_mask.sum(1, keepdim=True).clamp(min=1)  # (B, hidden_size)

        logits = self.classifier(pooled_video.float())  # (B, num_labels)
        return logits


# ------------------------
# 4. METRICS
# ------------------------
def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return dict of class-wise and macro precision/recall/f1/accuracy."""
    y_true_bin = y_true > 0.5
    y_pred_bin = y_pred > 0.5

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    acc_m = accuracy_score(y_true_bin, y_pred_bin)

    metrics = {f"{CLASSES[i]}/precision": float(prec[i]) for i in range(NUM_LABELS)}
    metrics.update({f"{CLASSES[i]}/recall": float(rec[i]) for i in range(NUM_LABELS)})
    metrics.update({f"{CLASSES[i]}/f1": float(f1[i]) for i in range(NUM_LABELS)})
    metrics.update({
        "macro/precision": float(prec_m),
        "macro/recall": float(rec_m),
        "macro/f1": float(f1_m),
        "macro/accuracy": float(acc_m),
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
# 5. MAIN EVALUATION LOOP
# ------------------------
def main(args):
    # Load processor & backbone
    processor = LlavaNextVideoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(args.model_id)

    # If the trained model used LoRA, wrap backbone with same config
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, lora_cfg)

    # Build classifier and load state_dict
    model = LlavaVideoClassifier(
        backbone=backbone,
        hidden_size=backbone.config.text_config.hidden_size,
        num_labels=NUM_LABELS,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare prompt (must match training prompt)
    caption_prompt = (
        "You are in a simulation of a neonatal resuscitation scenario. "
        "Give a caption for the video; be explicit about:\n"
        "-Who is present in the video.\n"
        "-What is happening in the video.\n"
        "-What actions are being performed.\n"
        "-What equipment is being used."
    )
    prompt = build_conversation(caption_prompt)

    # Prepare test dataset & loader
    test_ds = VideoJsonlDataset(
        jsonl_path=args.test_json,
        processor=processor,
        prompt_template=prompt,
        num_frames=args.num_frames,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            pixel_values_videos = batch["pixel_values_videos"].to(device)       # (B, F, C, H, W)
            input_ids = batch["input_ids"].to(device)                           # (B, T)
            attn = batch["attention_mask"].to(device)                           # (B, T)
            video_mask = batch["video_token_mask"].to(device)                   # (B, T)
            labels = batch["labels"].to(device)                                  # (B, num_labels)
            
            print(f"\nPixel values shape: {pixel_values_videos.shape}", flush=True)
            print(f"Input IDs shape: {input_ids.shape}", flush=True)
            print(f"Attention mask shape: {attn.shape}", flush=True)
            print(f"Video token mask shape: {video_mask.shape}", flush=True)
            print(f"Labels shape: {labels.shape}", flush=True)
            
            logits = model(pixel_values_videos, input_ids, attn, video_mask)    # (B, num_labels)
            probs = torch.sigmoid(logits)                                       # (B, num_labels)

            all_logits.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    test_metrics = metrics_from_preds(y_true, y_pred)

    # Print metrics
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Optionally save metrics to JSON
    if args.output_metrics:
        os.makedirs(Path(args.output_metrics).parent, exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nSaved test metrics to {args.output_metrics}")


# ------------------------
# 6. ARGPARSE
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LlavaVideoClassifier on a test JSONL dataset."
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to the test JSONL file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model state_dict (e.g., outputs/model_final.pt)."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-next-large-430k",
        help="Pretrained Llava model ID (should match training)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for testing."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample per video (must match training)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether the trained model used LoRA adapters."
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (only required if --use_lora)."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (only required if --use_lora)."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (only required if --use_lora)."
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default=None,
        help="Path to save the test metrics JSON (optional)."
    )

    args = parser.parse_args()
    main(args)
