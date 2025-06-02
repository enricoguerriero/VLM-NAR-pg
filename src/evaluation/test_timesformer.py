import json, math, torch, numpy as np
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TimesformerConfig,
)
from sklearn.metrics import average_precision_score, f1_score
import av  

# ---------- 1.  Config --------------------------------------------------------

JSONL_TEST   = "data/clips/test_balanced.jsonl"          # ← your test file
PRETRAINED   = "models/TimeSformer/model.pth"  
MODEL_ID = "facebook/timesformer-base-finetuned-ssv2"  # base model for TimeSformer
BATCH_SIZE   = 4
NUM_FRAMES   = 8                          # must match dataset + model
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["baby_visible", "ventilation", "stimulation", "suction"]  # keep the same ordering

# ---------- 2.  Dataset & Dataloader -----------------------------------------

class VideoJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor: AutoImageProcessor,
        num_frames: int = 8,
    ):
        super().__init__()
        self.records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
        self.processor = processor
        self.num_frames = num_frames

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
        # if len(frames) != self.num_frames:     # Pad if video too short
        #     frames.extend(frames[-1:] * (self.num_frames - len(frames)))
        return np.stack(frames)            # (T, H, W, 3)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = self._read_frames(rec["video"])  # (T, H, W, 3)

        processed = self.processor([frame for frame in frames], return_tensors="pt")

        pixel_values = processed["pixel_values"]  # (F, C, H, W)

        label_vec = torch.tensor([rec["labels"][c] for c in CLASSES], dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "labels": label_vec
        }

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
test_ds   = VideoJsonlDataset(JSONL_TEST, processor, num_frames=NUM_FRAMES)

def collate_fn(batch):
    """
    - stack pixel_values 
    - stack label vectors
    """
    pix = torch.cat([item["pixel_values"] for item in batch])  
    lab = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pix, "labels": lab}

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# ---------- 3.  Model ---------------------------------------------------------
class TimeSformer(nn.Module):
    
    def __init__(self,  
                 checkpoint: str = "facebook/timesformer-base-finetuned-ssv2",
                 base_model_id: str = "facebook/timesformer-base-finetuned-ssv2", 
                 device: str = "cuda", 
                 num_classes: int = 4,
                 num_frames: int = 8):
        """
        Initialize the TimeSformer model.
        """
        super().__init__()
        self.model_name = "TimeSformer"
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = TimesformerConfig.from_pretrained(base_model_id)
        self.processor = AutoImageProcessor.from_pretrained(base_model_id)
        self.backbone = TimesformerForVideoClassification.from_pretrained(base_model_id)
        self.backbone.classifier = nn.Identity()
        
        if num_frames != 8:  # 8 = default number of frames for SSV2 pretraining
            self.interpolate_pos_encoding(num_frames=num_frames)
        
                
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes, bias=True)
        self.backbone.to(self.device)
        self.classifier.to(self.device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_state_dict(state_dict)

        
    def forward(self, pixel_values, labels = None, loss_fct = None): 
        """
        Forward pass through the model.
        """
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        features = hidden_states.mean(dim=1)
        logits = self.classifier(features)
        loss = None
        if labels is not None and loss_fct is not None:
            loss = loss_fct(logits, labels.float())
        
        return {"loss": loss, "logits": logits}

model = TimeSformer(
    checkpoint=PRETRAINED,
    base_model_id=MODEL_ID,
    device=DEVICE,
    num_classes=len(CLASSES),
    num_frames=NUM_FRAMES
)

# ---------- 4.  Evaluation loop ----------------------------------------------

all_logits, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"].to(DEVICE)

        out    = model(pixel_values=pixel_values)
        logits = out["logits"]    # (B, |C|)
        loss  = out["loss"] if "loss" in out else None
        assert logits.shape == labels.shape
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

logits = torch.cat(all_logits)            # (N, |C|)
labels = torch.cat(all_labels)            # (N, |C|)
probas = logits.sigmoid().numpy()         # convert to probabilities
y_true = labels.numpy()

# ---------- 5.  Metrics -------------------------------------------------------

threshold = 0.50
y_pred_bin = (probas >= threshold).astype(int)

# mean Average Precision (mAP)
ap_per_class = [
    average_precision_score(y_true[:, i], probas[:, i])
    for i in range(len(CLASSES))
]
map_score = np.mean(ap_per_class)

# micro / macro F1
f1_micro = f1_score(y_true, y_pred_bin, average="micro")
f1_macro = f1_score(y_true, y_pred_bin, average="macro")

print(f"mAP : {map_score:6.4f}")
print(f"F1-micro : {f1_micro:6.4f}")
print(f"F1-macro : {f1_macro:6.4f}")

# Optional: per-class AP table
for c, ap in zip(CLASSES, ap_per_class):
    print(f"{c:20s}  {ap:6.4f}")
