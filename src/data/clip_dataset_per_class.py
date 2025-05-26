import torch
from torch.utils.data import Dataset
from src.data import ClipDataset

class BinaryClipDataset(Dataset):
    """
    Flatten a multi-label ClipDataset into binary-per-class examples.

    Each sample is one clip (frames tensor) plus:
      - class_idx: which class (0..num_classes-1)
      - label: 0/1 for that class
    """
    def __init__(self,
                 video_folder: str,
                 annotation_folder: str,
                 clip_length: int,
                 overlapping: float,
                 frame_per_second: int,
                 num_classes: int = 4,
                 transform=None):
        # 1) build the underlying multi-label ClipDataset
        self.clip_ds = ClipDataset(
            video_folder, annotation_folder,
            clip_length, overlapping,
            frame_per_second, transform
        )

        # 2) make a flat index of (clip_idx, class_idx)
        self.num_classes = num_classes
        self.entries = [
            (clip_idx, class_idx)
            for clip_idx in range(len(self.clip_ds))
            for class_idx in range(self.num_classes)
        ]

        # 3) compute per-class counts / weights / priors
        # first, force the ClipDataset to compute its pos/prior
        self.clip_ds.compute_pos_weight()
        self.clip_ds.compute_prior()

        # total number of clips
        n_clips = len(self.clip_ds)

        # from ClipDataset:
        #   clip_ds.pos_weight = neg/pos per class
        #   clip_ds.prior_probability = pos/total per class
        self.pos_weight = self.clip_ds.pos_weight
        self.prior_probability = self.clip_ds.prior_probability

        # if you also want raw counts:
        self.pos_counts = self.prior_probability * n_clips
        self.neg_counts = n_clips - self.pos_counts
        self.counts_per_class = torch.full_like(self.pos_counts, n_clips, dtype=torch.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        clip_idx, class_idx = self.entries[idx]

        # get the full multi-label clip example
        clip_data = self.clip_ds[clip_idx]
        frames = clip_data['frames']          # [T, C, H, W]
        labels = clip_data['labels']          # tensor shape [num_classes]

        # extract just the one binary label
        label = labels[class_idx]

        return {
            'frames':     frames,
            'class_idx':  torch.tensor(class_idx, dtype=torch.long),
            'label':      torch.tensor(label,   dtype=torch.long),
        }
