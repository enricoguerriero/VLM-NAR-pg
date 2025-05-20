import os
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(f for f in os.listdir(data_dir) if f.endswith('.pt'))
        self.n_classes = 4
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path = os.path.join(self.data_dir, self.files[idx])
            data = torch.load(file_path, weights_only=False)
            return data
        except Exception as e:
            print(f"Error loading file {self.files[idx]}: {e}")
            return None

    def weight_computation(self):
        """
        Compute the positive class weights for BCEWithLogitsLoss.
        """
        pos_counts = torch.zeros(self.n_classes, dtype=torch.float32)
        total_samples = len(self.files)
        
        for f in self.files:
            data = torch.load(os.path.join(self.data_dir, f), weights_only=False)
            label = data["labels"]
            pos_counts += label.squeeze(0).float()
        
        neg_counts = total_samples - pos_counts
        raw_weight = neg_counts / (pos_counts + 1e-6)
        pos_weight = torch.clamp(raw_weight, min=0.0, max=10.0)
        prior_prob = pos_counts / total_samples
        
        return pos_weight, prior_prob