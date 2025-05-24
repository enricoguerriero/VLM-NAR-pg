import os
import re
import torch
from torch.utils.data import Dataset

class BinaryTokenDataset(Dataset):
    """
    A Dataset for binary-processed inputs saved as .pt files.
    Each file corresponds to one prompt/class and has its binary label in the filename:
    video_{i}_clip_{j}_class_{k}_label_{b}.pt
    """
    def __init__(self, data_dir: str, num_classes: int):
        self.data_dir = data_dir
        # List and sort .pt files by their numeric components
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.pt')],
            key=lambda x: [int(n) for n in re.findall(r"\d+", x)]
        )
        self.num_classes = num_classes

        # Compute class-wise positive and total counts for weighting
        self.pos_counts = torch.zeros(self.num_classes, dtype=torch.float32)
        self.counts_per_class = torch.zeros(self.num_classes, dtype=torch.float32)
        for fname in self.files:
            m = re.search(r'class_(\d+)_label_(\d+)\.pt$', fname)
            if not m:
                raise ValueError(f"Filename {fname} does not match expected pattern.")
            class_idx = int(m.group(1))
            label = int(m.group(2))
            self.pos_counts[class_idx] += label
            self.counts_per_class[class_idx] += 1

        self.neg_counts = self.counts_per_class - self.pos_counts
        raw_weight = self.neg_counts / (self.pos_counts + 1e-6)
        self.raw_weight = raw_weight
        self.pos_weight = torch.clamp(raw_weight, min=0.0, max=10.0)
        self.prior_probability = self.pos_counts / self.counts_per_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.data_dir, fname)
        data = torch.load(path, weights_only=False)

        m = re.search(r'class_(\d+)_label_(\d+)\.pt$', fname)
        class_idx = int(m.group(1))
        label_value = int(m.group(2))

        data['label'] = torch.tensor(label_value, dtype=torch.long)
        data['class_idx'] = torch.tensor(class_idx, dtype=torch.long)
        return data
