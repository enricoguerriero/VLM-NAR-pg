#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, random, math, os, av, numpy as np, torch
from typing import List, Dict
from datasets import load_dataset, Dataset
from transformers import (VideoLlavaProcessor,
                          VideoLlavaForConditionalGeneration,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model
from torch.utils.data.dataloader import default_collate
import argparse

# -------------------------
# CONFIG (edit here)
# -------------------------
MODEL_ID          = "LanguageBind/Video-LLaVA-7B-hf"
MAX_NEW_TOKENS    = 64
NUM_FRAMES        = 8        # Video-LLaVA was trained with 8 frames
PROMPT_TEMPLATE   = (
    "USER: <video>\n"
    "You are an expert neonatal instructor.\n"
    "Possible events:\n"
    "A. Baby mannequin visible on the table.\n"
    "B. A caregiver holds a ventilation (PPV/CPAP) mask on the baby's face.\n"
    "C. A caregiver rubs the baby's trunk/back (stimulation).\n"
    "D. A suction tube is inserted in the baby's mouth or nose.\n\n"
    "Rules:\n"
    "1. B, C and D can only happen if A happens.\n"
    "2. B and D are mutually exclusive.\n\n"
    "TASK → Respond ONLY with the JSON dictionary:\n"
    "{\"baby_visible\":<yes/no>,\"ventilation\":<yes/no>,"
    "\"stimulation\":<yes/no>,\"suction\":<yes/no>} "
    "ASSISTANT:"
)

# -------------------------
# Helpers
# -------------------------
def sample_frames(container: av.container.input.InputContainer,
                  num_frames: int = 8) -> np.ndarray:
    """Uniformly sample `num_frames` RGB24 frames and return (T,H,W,3)."""
    total = container.streams.video[0].frames
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames  = []
    container.seek(0)
    for i, f in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(f.to_ndarray(format="rgb24"))
    assert len(frames) == num_frames, "could not decode enough frames"
    return np.stack(frames)


def labels_to_json(lbl: Dict[str,int]) -> str:
    yesno = lambda b: "yes" if b else "no"
    return json.dumps({
        "baby_visible": yesno(lbl["baby_visible"]),
        "ventilation": yesno(lbl["ventilation"]),
        "stimulation": yesno(lbl["stimulation"]),
        "suction":     yesno(lbl["suction"]),
    })

# -------------------------
# Dataset → HF Dataset
# -------------------------
def load_split(path: str) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    return ds

# -------------------------
# Custom collator
# -------------------------
class VideoSFTCollator:
    def __init__(self, processor: VideoLlavaProcessor, num_frames: int):
        self.processor = processor
        self.num_frames = num_frames
        self.pad_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"   # recommended

    def __call__(self, batch):
        # read video & build prompt/answer
        videos, texts, answers = [], [], []
        for item in batch:
            clip_path = item["video"]
            container = av.open(clip_path)
            frames = sample_frames(container, self.num_frames)
            videos.append(frames)
            prompt = PROMPT_TEMPLATE
            texts.append(prompt)
            answers.append(labels_to_json(item["labels"]))

        # tokenize
        proc_out = self.processor(text=texts,
                                  videos=videos,
                                  return_tensors="pt",
                                  padding=True)
        # Append answer after prompt and build training labels
        input_ids, attention_mask, labels = [], [], []
        for i in range(len(texts)):
            full = texts[i] + " " + answers[i]
            enc  = self.processor(text=full, videos=[videos[i]],
                                  return_tensors="pt")
            inp, att = enc["input_ids"], enc["attention_mask"]
            lab = inp.clone()
            prompt_len = len(self.processor.tokenizer(texts[i]).input_ids)
            lab[0, :prompt_len] = -100
            input_ids.append(inp[0]); attention_mask.append(att[0]); labels.append(lab[0])

        return {
            "input_ids":       torch.stack(input_ids),
            "attention_mask":  torch.stack(attention_mask),
            "labels":          torch.stack(labels),
            "videos":          torch.stack([torch.from_numpy(v) for v in videos]),
        }

# -------------------------
# Training entry-point
# -------------------------
def main(train_json: str,
         valid_json: str,
         output_dir: str,
         batch_per_gpu: int,
         grad_accum: int,
         lr: float,
         epochs: int,
         lora_r: int,
         lora_alpha: int,
         lora_dropout: float):

    processor = VideoLlavaProcessor.from_pretrained(MODEL_ID)
    model      = VideoLlavaForConditionalGeneration.from_pretrained(
                    MODEL_ID, torch_dtype=torch.float16, device_map="auto")

    # tie weights and apply LoRA
    model.tie_weights()
    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                      target_modules=["q_proj", "v_proj"],
                      lora_dropout=lora_dropout,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    # Load datasets
    train_ds = load_split(train_json)
    valid_ds = load_split(valid_json)

    collator = VideoSFTCollator(processor, NUM_FRAMES)

    args = TrainingArguments(
        output_dir                   = output_dir,
        per_device_train_batch_size  = batch_per_gpu,
        per_device_eval_batch_size   = 1,
        gradient_accumulation_steps  = grad_accum,
        num_train_epochs             = epochs,
        learning_rate                = lr,
        fp16                         = True,
        logging_steps                = 20,
        save_strategy                = "epoch",
        eval_strategy                = "epoch",
        report_to                    = "none",
        remove_unused_columns        = False, 
    )

    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_ds,
                      eval_dataset=valid_ds,
                      data_collator=collator)

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoLLaVA on neonatal resuscitation events.")
    parser.add_argument("--train_json", type=str, default="data/clips/train.jsonl")
    parser.add_argument("--valid_json", type=str, default="data/clips/validation.jsonl")
    parser.add_argument("--output_dir", type=str, default="runs/videollava_neoresus")
    parser.add_argument("--batch_per_gpu", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    main(args.train_json, args.valid_json, args.output_dir,
         args.batch_per_gpu, args.grad_accum, args.lr,
         args.epochs, args.lora_r, args.lora_alpha, args.lora_dropout)
