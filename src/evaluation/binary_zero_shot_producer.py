#!/usr/bin/env python
import json
import os
import time
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

from src.utils import (
    load_model,
    setup_logging,
    set_global_seed,
    load_config
)
from src.data import BinaryTokenDataset, VLMVideoDataset

def main():
    parser = ArgumentParser("Producer: generate captions stream (no padding)")
    parser.add_argument("--model_name",  type=str, required=True)
    parser.add_argument("--output_file", type=str,
                        default="stream.ndjson",
                        help="Where to append caption entries")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="(Optional) pause between samples")
    args = parser.parse_args()

    logger = setup_logging(args.model_name, "producer")
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))
    set_global_seed()

    config = load_config(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.model_name, None).to(device)
    model.eval()

    # load dataset
    ds = BinaryTokenDataset(
        data_dir=os.path.join(config["token_folder"], args.model_name, "binary"),
        num_classes=4
    )
    LABELS_CSV = "data/clips/test/labels.csv"
    ds = VLMVideoDataset(
        csv_file       = LABELS_CSV,
        processor      = model.processor,
        prompts        = ["USER: Describe the clip; focus on who is present in the clip: <video>. ASSISTANT:",
                          "USER: Describe the clip; focus on eventual respiration equipment and how is it eventually used: <video>. ASSISTANT:",
                          "USER: Describe the clip; if the baby / doll is being stimulated, describe also that movement: <video>. ASSISTANT:",
                          "USER: Describe the clip; if a suction tube is present, describe that and how it is used. Note that the suction tube is different from a ventilation mask: <video>. ASSISTANT:"],   
        system_message = "This is a simulation of a medical resuscitation context.",                             
        frames         = 16,
        frame_sample   = "uniform",
    )
    label2idx = {name: i for i, name in enumerate(ds.label_cols)} # for label mapping

    # prepare output
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    open(args.output_file, "a").close()

    logger.info(f"Streaming captions to {args.output_file}")
    clip_counter = 0
    with open(args.output_file, "a") as outf:
        for sample in tqdm(ds, desc="Generating samples"):

            inputs = {
                "input_ids":        sample["input_ids"].unsqueeze(0).to(device),
                "attention_mask":   sample["attention_mask"].unsqueeze(0).to(device),
                "pixel_values_videos": sample["pixel_values_videos"].unsqueeze(0).to(device),
            }

            with torch.no_grad():
                captions = model.generate_answer(
                    inputs=inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            class_idx = label2idx[sample["label_name"]]
            
            # single output per sample
            entry = {
                "clip_idx":  clip_counter,
                "class_idx": class_idx,
                "label":     sample["label"].tolist(),
                "caption":   captions[0]
            }
            outf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            outf.flush()
            os.fsync(outf.fileno())
            clip_counter += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    logger.info("Producer finished streaming all captions.")

if __name__ == "__main__":
    main()
