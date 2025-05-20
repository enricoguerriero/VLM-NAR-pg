import torch.nn as nn
import logging
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch import Tensor
from torch.optim import lr_scheduler
import wandb
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer, lr_scheduler
import copy

class BaseModel(nn.Module):
    """
    An abstract base class for video models.
    Provides a common interface for training, inference, and last-layer modifications.
    """
    def __init__(self, device = "cuda", model_name: str = "baseModel"):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.pos_weights = None
        self.processor = None
        self.classifier = None
        self.num_classes = None
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def process_input(self, frame_list = None, prompt = None, system_message = None):
        """
        To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_input().")
    
    def feature_extraction(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Extract features from the model.
        :param pixel_values: Input pixel values (images).
        :param input_ids: Input token IDs (text).
        :param attention_mask: Attention mask for the input.
        :return: Extracted features.
        """
        raise NotImplementedError("Subclasses must implement feature_extraction().")
    
    def save_features(self, dataloader: DataLoader, output_dir: str):
        """
        Save features to the output directory.
        """
        self.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting features"):

                labels = batch.pop("labels")
                input = self.get_input(batch)
                
                features = self.feature_extraction(**input)
                data = {
                    "features": features.cpu(),
                    "labels": labels.cpu()
                }
                
                torch.save(data, os.path.join(output_dir, f"features_{step}.pt"))
    
    def set_weights(self, weights):
        """
        Set the model weights from a given path.
        """
        self.pos_weights = weights
        
    def save_model(self):
        """
        Save the model to a file.
        """
        model_path = os.path.join("models/saved", self.model_name)
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, "model.pth"))   
    
    def save_checkpoint(self, epoch, optimizer, scheduler):
        """
        Save the model checkpoint.
        """
        checkpoint_path = os.path.join("models/saved", self.model_name, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_path, f"checkpoint_{epoch}.pth"))     
        
    def define_optimizer(self, 
                         optimizer_name: str, 
                         learning_rate: float,
                         momentum: float = None,
                         weight_decay: float = None):
        """
        Defines the optimizer for the model.
        By now you can choose between Adam and SGD.
        """
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                        lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not available")
        return optimizer
    
    def define_criterion(self, 
                         criterion_name: str, 
                         pos_weight: torch.Tensor = None):
        """
        Defines the criterion for the model.
        By now you can choose between BCE and CrossEntropy.
        """
        if criterion_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion_name == "wbce":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            raise ValueError(f"Criterion {criterion_name} not available")
        return criterion
    
    def define_scheduler(self, 
                         scheduler_name: str, 
                         optimizer: torch.optim.Optimizer = None,
                         epochs: int = None, 
                         patience: int = None,
                         step_size: int = 5, 
                         gamma: float = 0.1,
                         eta_min: float = 0,
                         factor: float = 0.1,
                         mode: str = "min",
                         cooldown: int = 0,
                         min_lr: float = 1e-6):
        """
        Defines the scheduler for the model.
        By now you can choose between StepLR and CosineAnnealingLR.
        """
        if scheduler_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=step_size, 
                                                        gamma=gamma)
        elif scheduler_name == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                   T_max=epochs, 
                                                                   eta_min=eta_min)
        elif scheduler_name == "reduceonplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode=mode, 
                                                                   factor=factor, 
                                                                   patience=patience,
                                                                   cooldown=cooldown,
                                                                   min_lr=min_lr)
        else:
            raise ValueError(f"Scheduler {scheduler_name} not available")
        return scheduler
    
    def collate_fn_tokens(self, batch):
        """
        Collate function for DataLoader.
        """
        pixel_values_videos = torch.cat([item["pixel_values_videos"] for item in batch], dim=0)
        input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        return {"pixel_values_videos": pixel_values_videos,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}

            
    def export_tokens_binary(self, 
                              video_folder: str, 
                              annotation_folder: str, 
                              output_folder: str,
                              clip_length: int,
                              overlapping: float,
                              frame_per_second: int,
                              prompts: list,
                              system_message: str = "You are a helpful assistant.",
                              logger: logging.Logger = None):
        """
        Export processed tokens in a binary setup: for each clip, save one file per prompt in `prompts` list, 
        with a single binary label (0 or 1) corresponding to that prompt/class.
        """
        if logger:
            logger.info(f"Exporting binary tokens for model {self.model_name}")
            logger.info(f"Video folder: {video_folder}")
            logger.info(f"Annotation folder: {annotation_folder}")
            logger.info(f"Output folder: {output_folder}")
            logger.info(f"Clip length: {clip_length}")
            logger.info(f"Overlapping: {overlapping}")
            logger.info(f"Frame per second: {frame_per_second}")
            logger.info(f"Number of prompts: {len(prompts)}")
        
        # sort video and annotation files
        video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
        annotation_files = sorted(glob.glob(os.path.join(annotation_folder, "*.txt")))
        if logger:
            logger.info(f"Found {len(video_files)} video files and {len(annotation_files)} annotation files.")
        
        folder_name = f'{clip_length}sec_{frame_per_second}fps'
        output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        if logger:
            logger.info(f"Created output folder: {output_folder}")
            logger.info(f"Exporting binary tokens to {output_folder}")
        
        for i, video_file in enumerate(video_files):
            annotation_file = annotation_files[i]
            if logger:
                logger.info("-" * 20)
                logger.info(f"Processing video {i + 1}/{len(video_files)}: {video_file}")
            annotation = self.read_annotations(annotation_file)
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                if logger:
                    logger.error(f"Error opening video file {video_file}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(int(fps / frame_per_second), 1)
            frame_per_clip = int(clip_length * frame_per_second)
            overlapping_frames = int(overlapping * frame_per_clip)
            
            frame_index = 0
            frames_list = []
            clip_index = 0
            first_frame_time = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index % frame_interval == 0:
                    current_time_ms = (frame_index / fps) * 1000
                    if first_frame_time is None:
                        first_frame_time = current_time_ms
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame)
                    
                    if len(frames_list) == frame_per_clip:
                        clip_start = first_frame_time
                        clip_length_ms = clip_length * 1000
                        # overall label vector for this clip
                        labels = self.label_clip(clip_start, clip_length_ms, annotation)
                        
                        # save one file per prompt/class
                        for idx, prompt in enumerate(prompts):
                            # binary label for this prompt
                            binary_label = int(labels[idx])
                            tokens = self.process_input(frames_list, prompt, system_message)
                            file_name = (
                                f"video_{i}_clip_{clip_index}_class_{idx}_label_{binary_label}.pt"
                            )
                            torch.save(tokens, os.path.join(output_folder, file_name))
                            if logger:
                                logger.info(f"Saved clip {clip_index} for prompt {idx} with label {binary_label}")
                        
                        # prepare for next clip
                        slide = frame_per_clip - overlapping_frames
                        first_frame_time += slide * (1000 / frame_per_second)
                        frames_list = frames_list[slide:]
                        clip_index += 1
                frame_index += 1

            cap.release()
            if logger:
                logger.info(f"Finished processing {video_file}")
                logger.info(f"Exported {clip_index * len(prompts)} clips (binary) from {video_file}")
                logger.info("-" * 20)
    
    def read_annotations(self, file_path):
        """
        Reads an annotation .txt file and returns a list of tuples:
        (label: str, start: int, end: int, length: int)

        Assumes each line has a variable-length label followed by three integer fields.
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Find first numeric token; tokens before that form the label
                idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
                if idx is None or idx + 2 >= len(parts):
                    continue  # skip malformed lines
                label = ' '.join(parts[:idx])
                ann_start = int(parts[idx])
                ann_end = int(parts[idx + 1])
                ann_length = int(parts[idx + 2])
                annotations.append((label, ann_start, ann_end, ann_length))
        return annotations
            
    def label_clip(self, clip_start, clip_length, annotations):
        """
        Label the clip based on the annotations.
        """
        # Initialize labels and overlap accumulators
        labels = [0, 0, 0, 0]
        overlap = [0, 0, 0, 0]

        clip_end = clip_start + clip_length

        # Mapping of annotation labels to indices
        category_map = {
            'Baby visible': 0,
            'CPAP': 1, 'PPV': 1,
            'Stimulation trunk': 2, 'Stimulation back/nates': 2,
            'Suction': 3
        }

        for ann in annotations:
            ann_label, ann_start, ann_end, _ = ann
            if ann_label not in category_map:
                continue
            # Compute overlap duration
            start_overlap = max(clip_start, ann_start)
            end_overlap = min(clip_end, ann_end)
            dur = end_overlap - start_overlap
            if dur > 0:
                idx = category_map[ann_label]
                overlap[idx] += dur

        # Determine labels based on >50% coverage
        threshold = clip_length / 2
        for i in range(len(labels)):
            if overlap[i] > threshold:
                labels[i] = 1

        return labels
    
    def calibrate_thresholds(self,
                             logits: torch.Tensor,
                             labels: torch.Tensor,
                             num_grid: int = 101):
        """
        Calibrate per-class sigmoid thresholds by maximizing F1 on a held-out set.

        Args:
            logits: Tensor of shape (N, C) raw model outputs
            labels: Tensor of shape (N, C) binary ground-truth labels
            num_grid: Number of threshold candidates to sweep between 0 and 1
        """
        # move to cpu and numpy
        probs = logits.sigmoid().detach().cpu().numpy()
        truths = labels.cpu().numpy().astype(int)
        N, C = truths.shape

        best_thresholds = []
        for c in range(C):
            best_f1, best_t = -1.0, 0.5
            for t in np.linspace(0.0, 1.0, num_grid):
                preds_c = (probs[:, c] >= t).astype(int)
                f1_c = f1_score(truths[:, c], preds_c, zero_division=0)
                if f1_c > best_f1:
                    best_f1, best_t = f1_c, t
            best_thresholds.append(best_t)
        # store as tensor
        self.calibrated_thresholds = torch.tensor(best_thresholds, device=self.device)

        return self.calibrated_thresholds
    

    
    def filter_new_params(self, new_params, optimizer):
        """
        Filter out parameters that are already in the optimizer.
        """
        existing_params = {id(p) for group in optimizer.param_groups for p in group['params']}
        return [p for p in new_params if id(p) not in existing_params]
    
    def manage_unfreezing(self, epoch, epochs, optimizer, scheduler):
        """
        Manage the unfreezing schedule and parameter addition.
        """
        new_params = self.unfreeze_schedule(epoch, epochs)
        if new_params:

            # Filter new parameters to avoid duplicates
            new_params = self.filter_new_params(new_params, optimizer)

            # Only add new parameters if they are not already present
            if new_params:
                optimizer.add_param_group({"params": new_params})

                # Update the scheduler's optimizer reference, if applicable
                if scheduler is not None and hasattr(scheduler, 'optimizer'):
                    scheduler.optimizer = optimizer 
        
        
        
    # ----------------- Training and Testing ---------------- #
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, max_grad_norm=1.0):
        """
        Train the model for one epoch.
        """
        self.train()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        
        scaler  = GradScaler()
        
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            optimizer.zero_grad()
            labels = batch.pop("labels").to(self.device)
            
            
            with autocast(device_type='cuda'):
                outputs = self.forward(**batch, labels = labels, loss_fct = criterion)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
            labels_list.append(labels.cpu())
            logits_list.append(logits.cpu())    
            
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
    
    def eval_epoch(self, dataloader: DataLoader, criterion: nn.Module):
        """
        Evaluate the model for one epoch.
        """
        self.eval()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
                labels = batch.pop("labels").to(self.device)
                outputs = self.forward(**batch, labels = labels, loss_fct = criterion)
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
        
    
    def test_from_tokens(self,
                   test_dataloader: DataLoader,
                   criterion_name: str = None,
                   pos_weight: torch.Tensor = None,
                   threshold: float | torch.Tensor = 0.5,
                   wandb_run=None):
        """
        Test the model.
        """
        criterion = self.define_criterion(criterion_name = criterion_name,
                                          pos_weight = pos_weight)
        test_loss, test_logits, test_labels = self.eval_epoch(test_dataloader, criterion)
        
        test_metrics = self.metric_computation(test_logits, test_labels, threshold)
        
        if wandb_run is not None:
            self.log_test_wandb(wandb_run = wandb_run, 
                                test_loss = test_loss, 
                                test_metrics = test_metrics)
        
        return {"test_loss": test_loss,
                "test_metrics": test_metrics}
        


    # ----- Training classifier ------ #
    
    def train_classifier_epoch(self, dataloader, optimizer, loss_fct, max_grad_norm=1.0):
        """
        Train the classifier for one epoch.
        """
        self.classifier.train()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        scaler  = GradScaler()
        
        for batch in tqdm(dataloader, desc="Training Classifier", unit="batch"):
            optimizer.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            # print(f"labels: {labels}", flush=True)
            # print(f"inputs: {inputs}", flush=True)
            with autocast(device_type='cuda'):
                outputs = self.forward_classifier(**inputs, labels = labels, loss_fct=loss_fct)
            loss = outputs["loss"]
            logits = outputs["logits"]
            # print(f"loss: {loss}", flush=True)
            # print(f"logits: {logits}", flush=True)
            labels_list.append(labels.cpu())
            logits_list.append(logits.cpu())
            
            if loss is not None:
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
    
    def eval_classifier_epoch(self, dataloader, loss_fct):
        """
        Evaluate the classifier for one epoch.
        """
        self.classifier.eval()
        total_loss = 0.0
        total_samples = 0
        labels_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Classifier", unit="batch"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = inputs.pop("labels")
                
                outputs = self.forward_classifier(**inputs, labels = labels, loss_fct=loss_fct)
                loss = outputs["loss"]
                logits = outputs["logits"]
                labels_list.append(labels)
                logits_list.append(logits)
                
                if loss is not None:
                    total_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)
        
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss, logits_tensor, labels_tensor
        
    def train_classifier(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 5,
        optimizer: Optimizer = None,
        criterion: nn.Module = None,
        threshold: float | torch.Tensor = 0.5,
        scheduler: lr_scheduler = None,
        patience: int = 3,
        show_progress: bool = True,
        wandb_run=None
    ):
        self.classifier.to(self.device)
        best_val_loss = float('inf')
        no_improve = 0

        epo_iter = range(1, epochs + 1)
        if show_progress:
            epo_iter = tqdm(epo_iter, desc="Epochs", unit="epoch")

        for epoch in epo_iter:
            train_loss, logits, labels = self.train_classifier_epoch(train_dataloader, optimizer, criterion)

            log_msg = f"[{epoch:02d}/{epochs}] train-loss: {train_loss:.4f}"
            # print(f"logits: {logits}, labels: {labels}", flush=True)
            train_metrics = self.metric_computation(logits, labels, threshold)
            log_msg += f" | train-f1: {train_metrics['f1_macro']:.4f}"

            if val_dataloader is not None:
                val_loss, val_logits, val_labels = self.eval_classifier_epoch(val_dataloader, criterion)
                
                log_msg += f" | val-loss: {val_loss:.4f}"
                # print(f"val_logits: {val_logits}, val_labels: {val_labels}", flush=True)
                val_metrics = self.metric_computation(val_logits, val_labels, threshold)
                log_msg += f" | val-f1: {val_metrics["f1_macro"]:.4f}"

            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            if show_progress:
                epo_iter.set_postfix_str(log_msg)
            if wandb_run is not None:
                self.log_wandb(wandb_run = wandb_run, 
                               epoch = epoch, 
                               train_loss = train_loss, 
                               train_metrics = train_metrics, 
                               val_loss = val_loss if val_dataloader is not None else None,
                               val_metrics = val_metrics if val_dataloader is not None else None)
            self.save_checkpoint(epoch = epoch,
                                 optimizer = optimizer,
                                 scheduler = scheduler)
            
            if val_dataloader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.classifier.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} with patience {patience}.")
                    break

        self.classifier.load_state_dict(best_model_wts)
        self.save_model()

        results = {"train_loss": train_loss,
                   "train_metrics": train_metrics,
                   "val_loss": val_loss if val_dataloader is not None else None,
                   "val_metrics": val_metrics if val_dataloader is not None else None}

        return results
    
    def test_classifier(
        self,
        test_dataloader: DataLoader,
        threshold: float | torch.Tensor = 0.5,
        wandb_run=None
    ):
        self.classifier.to(self.device)
        test_loss, logits, labels = self.eval_classifier_epoch(test_dataloader, None)
        
        test_metrics = self.metric_computation(logits, labels, threshold)        
        
        if wandb_run is not None:
            self.log_test_wandb(wandb_run = wandb_run, 
                           test_loss = test_loss, 
                           test_metrics = test_metrics)

        return {"test_loss": test_loss, "test_metrics": test_metrics}
