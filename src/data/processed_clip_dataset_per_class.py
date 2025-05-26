import torch
from torch.utils.data import Dataset
from src.data import BinaryClipDataset

class VLMBinaryClipDataset(BinaryClipDataset):
    """
    Takes the same video/annotation setup as BinaryClipDataset,
    but for each (clip, class) returns a processed VLM input:
      - tokenized text = system_message + prompts[class_idx]
      - pixel/video frames run through the HF processor
    along with 'label' and 'class_idx'.
    """
    def __init__(self,
                 video_folder: str,
                 annotation_folder: str,
                 clip_length: int,
                 overlapping: float,
                 frame_per_second: int,
                 num_classes: int,
                 system_message: str,
                 prompts: list,
                 processor,
                 transform=None):
        # build the underlying binary‐flattened dataset
        super().__init__(
            video_folder, annotation_folder,
            clip_length, overlapping,
            frame_per_second, num_classes,
            transform
        )
        if len(prompts) != num_classes:
            raise ValueError(f"Expected {num_classes} prompts, got {len(prompts)}")
        self.system_message = system_message
        self.prompts = prompts
        self.processor = processor

    def __getitem__(self, idx):
        # get raw frames + class_idx + label
        sample = super().__getitem__(idx)
        frames = sample['frames']            # torch.Tensor [T, C, H, W]
        class_idx = int(sample['class_idx']) # scalar
        label = sample['label']              # 0/1 tensor

        # build the full text prompt
        text = self.system_message.strip()
        prompt = self.prompts[class_idx].strip()
        full_text = f"{text} {prompt}"

        # convert frames to numpy: (T, H, W, C)
        video_np = frames.permute(0, 2, 3, 1).cpu().numpy()

        # run through HF processor
        encoding = self.processor(
            text=full_text,
            video=video_np,
            return_tensors="pt",
            padding=True
        )

        # remove the leading batch‐dim (so DataLoader can stack cleanly)
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                encoding[k] = v.squeeze(0)

        # attach our binary label and class index
        encoding['label'] = label        # shape: () so collate→(B,)
        encoding['class_idx'] = torch.tensor(class_idx, dtype=torch.long)

        return encoding
