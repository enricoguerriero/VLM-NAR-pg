# vlm_video_dataset.py  (v2)
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from PIL import Image


class VLMVideoDataset(Dataset):
    """
    Vision-Language dataset that yields FOUR separate samples per clip,
    one for each prompt / target label.

    Parameters
    ----------
    csv_file        : Path to labels.csv (must contain the 4 binary columns)
    processor       : Any HF processor (e.g. LlavaProcessor)
    prompts         : *List of exactly four* prompt strings - order must match label_cols
    system_message  : System string prepended to *every* prompt
    frames          : int (# of frames) or "first"
    frame_sample    : "uniform" | "random"
    """
    # label order is fixed here ↓↓↓ (same order as build_resus_dataset.py)
    label_cols = ["Baby visible", "Ventilation", "Stimulation", "Suction"]

    def __init__(
        self,
        csv_file: str | Path,
        processor,
        prompts: List[str],
        system_message: str = "",
        frames: int | str = 1,
        frame_sample: str = "uniform",
    ):
        if len(prompts) != 4:
            raise ValueError("Need exactly four prompts - one per label column.")
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.prompts = prompts
        self.system = system_message
        self.frames = frames
        self.frame_sample = frame_sample.lower()

    # ─────────────────── frame helpers ───────────────────
    @staticmethod
    def _sample_indices(n_total: int, n_needed: int, mode: str):
        if n_needed >= n_total:
            return list(range(n_total))
        if mode == "uniform":
            step = n_total / n_needed
            return [int(step / 2 + i * step) for i in range(n_needed)]
        if mode == "random":
            import random
            return sorted(random.sample(range(n_total), n_needed))
        raise ValueError

    def _load_frames(self, path: str):
        vid, _, _ = read_video(path, pts_unit="sec")  # (T,H,W,C)
        idxs = (
            [0]
            if self.frames == "first"
            else self._sample_indices(len(vid), self.frames, self.frame_sample)
        )
        return [Image.fromarray(vid[i].numpy()) for i in idxs]

    # ─────────────────── PyTorch API ───────────────────
    def __len__(self):
        return len(self.df) * 4  # 4 prompts per clip

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        clip_idx, prompt_idx = divmod(idx, 4)
        row = self.df.iloc[clip_idx]

        videos = self._load_frames(row.file)
        user_prompt = self.prompts[prompt_idx]
        full_prompt = f"{self.system}\n{user_prompt}"

        # processor builds the model-ready dict
        model_inputs = self.processor(
            videos=videos,
            text=full_prompt,
            return_tensors="pt",
            padding=True,
        )
        # squeeze batch-dim (1)
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # single binary target (float32 → ready for BCE loss)
        target_val = torch.tensor(float(row[self.label_cols[prompt_idx]]))

        model_inputs["label"] = target_val
        model_inputs["label_name"] = self.label_cols[prompt_idx]
        model_inputs["file"] = row.file

        return model_inputs
