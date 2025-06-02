import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from torch.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import json
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import av
import wandb

CLASSES = ["baby_visible", "ventilation", "stimulation", "suction"]
NUM_LABELS = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSformer(nn.Module):
    
    def __init__(self,  
                 base_model_id: str = "facebook/timesformer-base-finetuned-ssv2", 
                 device: str = "cuda", 
                 num_classes: int = 4,
                 num_frames: int = 8):
        """
        Initialize the TimeSformer model.
        """
        self.model_name = "TimeSformer"
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = TimesformerConfig.from_pretrained(base_model_id)
        self.processor = AutoImageProcessor.from_pretrained(base_model_id)
        self.backbone = TimesformerForVideoClassification.from_pretrained(base_model_id)
        self.backbone.classifier = nn.Identity()
        # self.backbone.gradient_checkpointing_enable() # Enable gradient checkpointing - save GPU memory
        
        if num_frames != 8:  # 8 = default number of frames for SSV2 pretraining
            self.interpolate_pos_encoding(num_frames=num_frames)
        
        # self.backbone = torch.compile(self.backbone) # speed up training
        # if torch.cuda.device_count() > 1:
        #     self.backbone = torch.nn.DataParallel(self.backbone) # Use DataParallel if multiple GPUs are available
                
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes, bias=True)
        self.backbone.to(self.device)
        self.classifier.to(self.device)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

        
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

    
    def train_epoch(self, dataloader, optimizer, loss_fct, max_grad_norm=1.0):
        """
        Train the model for one epoch.
        """
        self.backbone.train()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        scaler  = GradScaler()
        
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with autocast(device_type='cuda'):
                outputs = self.forward(pixel_values, labels, loss_fct)
                loss = outputs["loss"]
                logits = outputs["logits"]
            # outputs = self.forward(pixel_values, labels, loss_fct)
            # loss = outputs["loss"]
            # logits = outputs["logits"]
            
            labels_list.append(labels.cpu())
            logits_list.append(logits.cpu())    
                       
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
    
    def eval_epoch(self, dataloader, loss_fct):
        """
        Evaluate the model for one epoch.
        """
        self.backbone.eval()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.forward(pixel_values, labels, loss_fct)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                labels_list.append(labels.cpu())
                logits_list.append(logits.cpu())
                
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
        
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
    
    def set_freezing_condition(self, freezing_condition: str = None):
        """
        Set the freezing condition for the model.
        """
        if freezing_condition == "all":
            for param in self.backbone.parameters():
                param.requires_grad = False
            return False
        elif freezing_condition == "none":
            for param in self.backbone.parameters():
                param.requires_grad = True
            return False
        elif freezing_condition == "partial":
            for param in self.backbone.parameters():
                param.requires_grad = False
            return True
        else:
            return False
        
    def get_layer_groups(self):
        """Return an *ordered list* of torch.nn.Module objects / ModuleLists."""
        return [
            self.classifier,                                          # 0
            nn.ModuleList([self.backbone.timesformer.layernorm,
                        self.backbone.timesformer.encoder.layer[-1]]),  # 1
            nn.ModuleList(self.backbone.timesformer.encoder.layer[8:11]),  # 2
            nn.ModuleList(self.backbone.timesformer.encoder.layer[4:8]),   # 3
            nn.ModuleList(self.backbone.timesformer.encoder.layer[0:4]),   # 4
            self.backbone.timesformer.embeddings                        # 5
        ]
    
    def unfreeze_schedule(self, epoch, epochs):
        """
        Gradually UNfreeze groups as training progresses.

        At 20 % of total epochs, group-1 is enabled,
        at 40 % group-2, …, until every group is trainable.
        """
        # thresholds must be **sorted** from high-to-low or low-to-high
        schedule = [(0.2, 1), (0.4, 2), (0.6, 3), (0.8, 4), (1.0, 5)]
        groups = self.get_layer_groups()

        progress = epoch / epochs
        newly_unfrozen = []

        for thresh, g_idx in schedule:
            if progress >= thresh:         # we have passed this milestone
                for m in groups[g_idx].modules():   # works for Module and ModuleList
                    for p in m.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            newly_unfrozen.append(p)
        return newly_unfrozen
    
    def manage_unfreezing(self, epoch, epochs, optimizer, scheduler, logger):
        """
        Manage the unfreezing schedule and parameter addition.
        """
        new_params = self.unfreeze_schedule(epoch, epochs)
        if new_params:
            logger.debug(f"Unfreezing condition met at epoch {epoch}")

            # Filter new parameters to avoid duplicates
            new_params = self.filter_new_params(new_params, optimizer)

            # Only add new parameters if they are not already present
            if new_params:
                logger.debug(f"Adding {len(new_params)} new parameters to optimizer")
                optimizer.add_param_group({"params": new_params})

                # Update the scheduler's optimizer reference, if applicable
                if scheduler is not None and hasattr(scheduler, 'optimizer'):
                    scheduler.optimizer = optimizer
                    logger.debug("Scheduler optimizer reference updated")
        
        # Log the current number of trainable parameters
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.debug(f"Number of trainable parameters: {num_trainable_params}")
 
        


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
        # Pre-compute index of image placeholder token once
        self.video_token = self.processor.tokenizer.video_token_id

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
        if len(frames) != self.num_frames:     # Pad if video too short
            frames.extend(frames[-1:] * (self.num_frames - len(frames)))
        return np.stack(frames)            # (T, H, W, 3)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = self._read_frames(rec["video"])  # (T, H, W, 3)

        processed = self.processor(frames, return_tensors="pt")

        pixel_values = processed["pixel_values"]  # (F, C, H, W)

        label_vec = torch.tensor([rec["labels"][c] for c in CLASSES], dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "labels": label_vec
        }

def compute_pos_weights(loader: DataLoader, device: str = "cpu") -> torch.Tensor:
    """Compute positive class weights (neg/pos) over an entire DataLoader."""
    total = torch.zeros(NUM_LABELS, device=device)
    pos = torch.zeros(NUM_LABELS, device=device)
    for batch in loader:
        labels = batch["labels"].to(device)
        total += labels.shape[0]
        pos += labels.sum(dim=0)
    neg = total - pos
    pos_weight = neg / pos.clamp(min=1)
    return pos_weight


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

    metrics = {f"{CLASSES[i]}/precision": prec[i] for i in range(NUM_LABELS)}
    metrics.update({f"{CLASSES[i]}/recall": rec[i] for i in range(NUM_LABELS)})
    metrics.update({f"{CLASSES[i]}/f1": f1[i] for i in range(NUM_LABELS)})
    metrics.update({
        "macro/precision": prec_m,
        "macro/recall": rec_m,
        "macro/f1": f1_m,
        "macro/accuracy": acc_m,
    })
    return metrics

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in the dataset.
    """
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    pixel_values = torch.cat(pixel_values, dim=0)
    labels = torch.stack(labels, dim=0)

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def main(args):
    
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = TimeSformer(
        base_model_id=args.model_id,
        device=args.device,
        num_classes=NUM_LABELS,
        num_frames=args.num_frames
    )
    model.to(device)
    
    train_ds = VideoJsonlDataset(
        jsonl_path=args.train_jsonl,
        processor=processor,
        num_frames=args.num_frames
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn = collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    ) 
    
    val_ds = VideoJsonlDataset(
        jsonl_path=args.val_jsonl,
        processor=processor,
        num_frames=args.num_frames
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn = collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    pos_weight = compute_pos_weights(train_loader, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.1, patience=1, verbose=True
    )

    best_val_loss = float("inf")
    no_improvement = 0
    
    