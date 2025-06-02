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

def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray):
    """Return dict of class-wise and macro precision/recall/f1/accuracy."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    assert y_true.shape == y_pred.shape
    y_true_bin = y_true > 0.5
    y_pred_bin = y_pred > 0.5

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    acc_m = accuracy_score(y_true_bin, y_pred_bin)

    metrics = {f"{CLASSES[i]}/precision": prec[i] for i in range(4)}
    metrics.update({f"{CLASSES[i]}/recall": rec[i] for i in range(4)})
    metrics.update({f"{CLASSES[i]}/f1": f1[i] for i in range(4)})
    metrics.update({
        "macro/precision": prec_m,
        "macro/recall": rec_m,
        "macro/f1": f1_m,
        "macro/accuracy": acc_m,
    })
    return metrics


metrics = metrics_from_preds(y_true, probas)

print("\n=== Evaluation Metrics ===\n")
class_names = sorted(set(k.split("/")[0] for k in metrics if "/" in k and not k.startswith("macro")))

# Header
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 50)

# Per-class metrics
for cls in class_names:
    p = metrics.get(f"{cls}/precision", 0)
    r = metrics.get(f"{cls}/recall", 0)
    f = metrics.get(f"{cls}/f1", 0)
    print(f"{cls:<15} {p:10.3f} {r:10.3f} {f:10.3f}")

print("\n--- Macro Averages ---")
print(f"{'Macro Precision':<20}: {metrics['macro/precision']:.3f}")
print(f"{'Macro Recall':<20}: {metrics['macro/recall']:.3f}")
print(f"{'Macro F1-Score':<20}: {metrics['macro/f1']:.3f}")
print(f"{'Macro Accuracy':<20}: {metrics['macro/accuracy']:.3f}")