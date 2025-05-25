import os
import torch
from torch.utils.data import Dataset, DataLoader
import re

class TokenDataset(Dataset):
    """
    A Dataset for already-processed inputs saved as .pt files.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(
            (f for f in os.listdir(data_dir) if f.endswith('.pt')),
            key=lambda x: int(re.search(r'clip_(\d+)', x).group(1))
        )
        self.n_classes = 4
        self.pos_counts = torch.zeros(self.n_classes, dtype=torch.float32)
        self._total_samples = len(self.files)
        
        for f in self.files:
            data = torch.load(os.path.join(data_dir, f), weights_only=False)
            if isinstance(data, dict):
                label = data['labels']
            elif hasattr(data, 'data') and 'labels' in data.data:
                label = data.data['labels']
            else:
                raise KeyError(f"'labels' key not found in file {f}")
            label = data['labels']
            self.pos_counts += label.float()
        neg_counts = self._total_samples - self.pos_counts
        raw_weight = neg_counts / (self.pos_counts + 1e-6)
        self.raw_weight = raw_weight
        self.pos_weight = torch.clamp(raw_weight, min=0.0, max=10.0)
        self.prior_probability = self.pos_counts / self._total_samples
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, weights_only=False)
        return data