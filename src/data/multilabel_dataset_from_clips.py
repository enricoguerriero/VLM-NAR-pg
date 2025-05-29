# vlm_video_multilabel_dataset.py  (v2)
import random
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from PIL import Image


class VLMVideoMultiLabelDataset(Dataset):
    """
    Vision-Language dataset that returns ONE sample per clip, suitable for
    multi-label (4-way) classification with BCEWithLogitsLoss.

    Parameters
    ----------
    csv_file        : Path to labels.csv (must contain the 4 binary columns)
    processor       : Any HF processor (e.g. LlavaProcessor)
    prompt          : Single user-prompt string (applied to every clip)
    system_message  : Optional system string prepended to the prompt
    frames          : int (# of frames) or "first"
    frame_sample    : "uniform" | "random"
    """

    # Fixed label order (same as build_resus_dataset.py)
    label_cols = ["Baby visible", "Ventilation", "Stimulation", "Suction"]

    # ─────────────────── statistics helper ───────────────────
    @classmethod
    def compute_class_stats(
        cls, csv_file: str | Path
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-class positive weights and prior probabilities.

        Returns
        -------
        pos_weight : torch.Tensor, shape (4,)
            (n_neg / n_pos) for each class - ready for BCEWithLogitsLoss.
        prior      : torch.Tensor, shape (4,)
            Empirical P(label == 1) for each class.
        """
        df = pd.read_csv(csv_file)
        pos = torch.tensor(df[cls.label_cols].sum().values, dtype=torch.float32)
        total = torch.tensor(len(df), dtype=torch.float32)

        # avoid division by zero in extremely skewed datasets
        eps = 1e-6
        prior = pos / (total + eps)
        pos_weight = (total - pos) / (pos + eps)

        return pos_weight, prior

    # ─────────────────── init & rest of the class ───────────────────
    def __init__(
        self,
        csv_file: str | Path,
        processor,
        prompt: str,
        system_message: str = "",
        frames: int | str = 1,
        frame_sample: str = "uniform",
    ):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.prompt = prompt
        self.system = system_message
        self.frames = frames
        self.frame_sample = frame_sample.lower()

        # Make the stats easily available from an instance too
        self.pos_weight, self.prior = self.compute_class_stats(csv_file)

    # ─────────────────── frame helpers ───────────────────
    @staticmethod
    def _sample_indices(n_total: int, n_needed: int, mode: str):
        if n_needed >= n_total:
            return list(range(n_total))
        if mode == "uniform":
            step = n_total / n_needed
            return [int(step / 2 + i * step) for i in range(n_needed)]
        if mode == "random":
            return sorted(random.sample(range(n_total), n_needed))
        raise ValueError(f"Unknown frame_sample mode: {mode}")

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
        return len(self.df)  # one prompt ⇒ one sample per clip

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # video → frames
        videos = self._load_frames(row.file)

        # build text prompt once
        full_prompt = f"{self.system}\n{self.prompt}" if self.system else self.prompt

        # processor assembles everything (returns batch-dim 1)
        model_inputs = self.processor(
            videos=videos,
            text=full_prompt,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # multi-label target vector (float32 → BCEWithLogitsLoss ready)
        targets = torch.tensor([float(row[c]) for c in self.label_cols])

        model_inputs["labels"] = targets          # shape (4,)
        model_inputs["label_names"] = self.label_cols
        model_inputs["file"] = row.file

        return model_inputs
