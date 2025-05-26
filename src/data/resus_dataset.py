import torch, torchvision
from pathlib import Path
import pandas as pd

class ResusDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video, _, _ = torchvision.io.read_video(row.file, pts_unit="sec")
        label = torch.tensor(row[["Baby visible", "Ventilation",
                                  "Stimulation", "Suction"]].values,
                             dtype=torch.float32)
        if self.transforms:
            video = self.transforms(video)
        return video, label
