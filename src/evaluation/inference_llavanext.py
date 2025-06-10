#!/usr/bin/env python
"""
Run sliding-window inference on a long video with a LlavaVideoClassifier.

Example
-------
python infer_long_video.py \
    --video_path demo/baby_resus.mp4 \
    --model_path outputs/model_final.pt \
    --save_csv demo/baby_resus_probs.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import av
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, get_peft_model

# ---------- constants ----------
CLASSES = ["baby_visible", "ventilation", "stimulation", "suction"]
NUM_LABELS = len(CLASSES)
PROMPT_TXT = (
    "You are in a simulation of a neonatal resuscitation scenario. "
    "Give a caption for the video; be explicit about:\n"
    "-Who is present in the video.\n"
    "-What is happening in the video.\n"
    "-What actions are being performed.\n"
    "-What equipment is being used."
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- helper ----------------
def build_conversation(text: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "video"},
            ],
        }
    ]


def uniform_indices(total: int, n: int) -> List[int]:
    """n indices spanning [0, total)."""
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


# ---------- model wrapper ----------
class LlavaVideoClassifier(nn.Module):
    def __init__(self, backbone: LlavaNextVideoForConditionalGeneration, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden, NUM_LABELS)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        video_token_mask: torch.Tensor,
    ):
        out = self.backbone(
            pixel_values_videos=pixel_values_videos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]  # (B, T, hidden)
        # pool only the <video> tokens
        pooled = (last_hidden * video_token_mask.unsqueeze(-1)).sum(1) / (
            video_token_mask.sum(1, keepdim=True).clamp(min=1)
        )
        logits = self.classifier(pooled.float())
        return torch.sigmoid(logits)  # (B, 4)


# ---------- inference loop ----------
def main(args):
    # ---------- load model ----------
    processor = LlavaNextVideoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(args.model_id)
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, lora_cfg)

    model = LlavaVideoClassifier(
        backbone, hidden=backbone.config.text_config.hidden_size
    ).to(DEVICE)
    state = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ---------- open video ----------
    container = av.open(args.video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    total_frames = stream.frames  # may be 0 for some codecs; fallback later
    duration_sec = stream.duration * stream.time_base if stream.duration else None

    if total_frames == 0 or duration_sec is None:
        # Fallback: decode to count frames & duration
        frames_tmp = [f for f in container.decode(video=0)]
        total_frames = len(frames_tmp)
        duration_sec = total_frames / fps
        container.seek(0)

    clip_len_sec = 3.0
    overlap_sec = 2.0
    stride_sec = clip_len_sec - overlap_sec  # == 1 s
    clip_len_frames = int(round(clip_len_sec * fps))
    stride_frames = int(round(stride_sec * fps))

    # Pre-extract every frame as rgb array to allow random indexing
    print("Decoding video …")
    frames_rgb = [f.to_ndarray(format="rgb24") for f in tqdm(container.decode(video=0), total=total_frames)]
    H, W = frames_rgb[0].shape[:2]

    # ---------- sliding window ----------
    starts = list(range(0, total_frames - clip_len_frames + 1, stride_frames))
    prompt_template = build_conversation(PROMPT_TXT)
    vid_token_id = processor.tokenizer.video_token_id

    results = []  # list of dicts
    print(f"Running inference on {len(starts)} clips …")
    with torch.no_grad():
        for s in tqdm(starts):
            # collect & uniformly subsample frames for this clip
            clip_frames_full = frames_rgb[s : s + clip_len_frames]
            idxs = uniform_indices(len(clip_frames_full), args.num_frames)
            clip_frames = np.stack([clip_frames_full[i] for i in idxs])  # (F, H, W, 3)

            # processor → tensors
            proc = processor(
                text=processor.apply_chat_template(prompt_template, add_generation_prompt=True),
                videos=clip_frames,
                return_tensors="pt",
            )
            pixel_vals = proc["pixel_values_videos"].unsqueeze(0).to(DEVICE)  # (1, F, C, H, W)
            input_ids = proc["input_ids"].to(DEVICE)
            attn = proc["attention_mask"].to(DEVICE)
            vid_mask = (input_ids == vid_token_id).long().to(DEVICE)

            probs = model(pixel_vals, input_ids, attn, vid_mask).cpu().squeeze(0).tolist()

            start_sec = s / fps
            end_sec = start_sec + clip_len_sec
            results.append(
                {
                    "clip_idx": len(results),
                    "start_s": round(start_sec, 3),
                    "end_s": round(end_sec, 3),
                    **{f"p_{c}": p for c, p in zip(CLASSES, probs)},
                }
            )

    # ---------- output ----------
    if args.save_csv:
        keys = ["clip_idx", "start_s", "end_s"] + [f"p_{c}" for c in CLASSES]
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved probabilities to {args.save_csv}")

    # pretty print a few lines
    for r in results[: min(5, len(results))]:
        print(
            f"[{r['start_s']:.1f}-{r['end_s']:.1f}s] "
            + ", ".join(f"{c}:{r[f'p_{c}']:.3f}" for c in CLASSES)
        )


# ---------- argparse ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser("Sliding-window inference for LlavaVideoClassifier")
    p.add_argument("--video_path", required=True, help="Path to a long video file (mp4, mov …)")
    p.add_argument("--model_path", required=True, help="state_dict file produced during training")
    p.add_argument(
        "--model_id",
        default="llava-hf/llava-next-large-430k",
        help="HF hub ID of the pretrained LLaVA-Next backbone",
    )
    p.add_argument("--num_frames", type=int, default=8, help="Frames sampled per 3-s clip")
    p.add_argument("--save_csv", type=str, default=None, help="Optional path to write CSV results")
    # LoRA
    p.add_argument("--use_lora", action="store_true", help="If your checkpoint uses LoRA")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    args = p.parse_args()
    main(args)
