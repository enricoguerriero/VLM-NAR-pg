import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

class ClipDataset(Dataset):
    """
    A Dataset that builds clips on-the-fly from videos and annotations.

    Each clip is composed of `clip_length` seconds of frames (sampled at `frame_per_second`),
    with `overlapping` fraction overlap between consecutive clips.
    """
    def __init__(self,
                 video_folder: str,
                 annotation_folder: str,
                 clip_length: int,
                 overlapping: float,
                 frame_per_second: int,
                 transform=None):
        self.video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
        self.annotation_files = sorted(glob.glob(os.path.join(annotation_folder, "*.txt")))
        assert len(self.video_files) == len(self.annotation_files), \
            "Number of videos and annotation files must match"

        self.clip_length = clip_length
        self.overlapping = overlapping
        self.frame_per_second = frame_per_second
        self.transform = transform

        # Prepare all clip entries: list of dicts
        self.clips = []
        for vid_path, ann_path in zip(self.video_files, self.annotation_files):
            # read annotations
            annotations = self._read_annotations(ann_path)
            # open video to get metadata
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # compute sampling parameters
            frame_interval = max(int(fps / self.frame_per_second), 1)
            frame_per_clip = int(self.clip_length * self.frame_per_second)
            overlapping_frames = int(self.overlapping * frame_per_clip)
            slide = frame_per_clip - overlapping_frames

            # how many sampled frames available
            num_samples = total_frames // frame_interval
            if num_samples < frame_per_clip:
                continue

            # number of clips
            num_clips = 1 + (num_samples - frame_per_clip) // slide

            for k in range(num_clips):
                start_sample = k * slide
                self.clips.append({
                    'video_path': vid_path,
                    'annotations': annotations,
                    'start_sample': start_sample,
                    'frame_interval': frame_interval,
                    'frame_per_clip': frame_per_clip,
                    'fps': fps
                })

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        vid_path = clip_info['video_path']
        annotations = clip_info['annotations']
        start_sample = clip_info['start_sample']
        frame_interval = clip_info['frame_interval']
        frame_per_clip = clip_info['frame_per_clip']
        fps = clip_info['fps']

        # collect frames
        cap = cv2.VideoCapture(vid_path)
        frames = []
        for j in range(frame_per_clip):
            sample_idx = start_sample + j
            frame_no = sample_idx * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            else:
                # convert to tensor
                frame = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
            frames.append(frame)
        cap.release()

        # stack into tensor [T, C, H, W]
        clip_tensor = torch.stack(frames)

        # compute clip start time in ms
        clip_start_frame = start_sample * frame_interval
        clip_start_ms = (clip_start_frame / fps) * 1000
        clip_length_ms = self.clip_length * 1000
        label = self._label_clip(clip_start_ms, clip_length_ms, annotations)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {'frames': clip_tensor, 'labels': label_tensor}

    def _read_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
                if idx is None or idx + 2 >= len(parts):
                    continue
                label = ' '.join(parts[:idx])
                ann_start = int(parts[idx])
                ann_end = int(parts[idx+1])
                ann_length = int(parts[idx+2])
                annotations.append((label, ann_start, ann_end, ann_length))
        return annotations

    def _label_clip(self, clip_start, clip_length, annotations):
        labels = [0,0,0,0]
        overlap = [0,0,0,0]
        clip_end = clip_start + clip_length
        category_map = {
            'Baby visible': 0,
            'CPAP': 1, 'PPV': 1,
            'Stimulation trunk': 2, 'Stimulation back/nates': 2,
            'Suction': 3
        }
        for ann_label, ann_start, ann_end, _ in annotations:
            if ann_label not in category_map:
                continue
            start_ov = max(clip_start, ann_start)
            end_ov = min(clip_end, ann_end)
            dur = end_ov - start_ov
            if dur > 0:
                overlap[category_map[ann_label]] += dur
        thresh = clip_length / 2
        for i in range(len(labels)):
            if overlap[i] > thresh:
                labels[i] = 1
        return labels
    
    def compute_pos_weight(self):
        """
        Compute per-class positive weights for BCEWithLogitsLoss:
           pos_weight[c] = (# negative examples of class c) / (# positive examples of class c)
        Clips with zero positives in a class get weight=1.0 for that class to avoid division by zero.
        """
        # accumulate counts
        pos_counts = torch.zeros(4, dtype=torch.float32)
        n_samples = len(self)
        for idx in range(n_samples):
            # get label tensor: shape [4], 0/1
            lbl = self[idx]['labels'].float()
            pos_counts += lbl

        # negatives = total clips minus positives
        neg_counts = n_samples - pos_counts

        # avoid division by zero: if a class never appears positive, weight=1.0
        pos_counts = torch.where(pos_counts == 0, torch.ones_like(pos_counts), pos_counts)

        pos_weight = neg_counts / pos_counts
        self.pos_weight = pos_weight
    
    def compute_prior(self):
        """
        Compute the prior probability P(y=1) for each of the 4 classes:
           prior[c] = (# positive examples of class c) / (total # clips)
        """
        pos_counts = torch.zeros(4, dtype=torch.float32)
        total = len(self)
        for i in range(total):
            pos_counts += self[i]['labels'].float()
        # divide by total number of clips to get P(y=1)
        prior = pos_counts / total
        self.prior_probability = prior