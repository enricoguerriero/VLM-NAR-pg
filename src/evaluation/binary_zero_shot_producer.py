#!/usr/bin/env python
import json
import os
import time
from argparse import ArgumentParser

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

from src.utils import (
    load_model,
    setup_logging,
    set_global_seed,
    load_config
)
from src.data import BinaryTokenDataset

def collate_fn(batch):
    """
    Pads text and stacks video tensors for a VLM-style model.
    Assumes keys: input_ids, attention_mask, pixel_values_videos, class_idx, label
    """
    from torch.nn.utils.rnn import pad_sequence

    seqs = [b["input_ids"].squeeze(0) for b in batch]
    masks = [b["attention_mask"].squeeze(0) for b in batch]

    input_ids = pad_sequence(seqs, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(masks, batch_first=True, padding_value=0)

    pixel_values_videos = torch.stack([b["pixel_values_videos"].permute(1, 0, 2, 3) for b in batch], dim=0)
    class_idx = torch.stack([b["class_idx"] for b in batch], dim=0)
    label     = torch.stack([b["label"]     for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values_videos,
        "class_idx": class_idx,
        "label": label,
    }

def main():
    parser = ArgumentParser("Producer: generate captions stream")
    parser.add_argument("--model_name",  type=str, required=True)
    parser.add_argument("--output_file", type=str,
                        default="stream.ndjson",
                        help="Where to append caption entries")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--sleep",       type=float, default=0.0,
                        help="(Optional) pause between batches")
    args = parser.parse_args()

    logger = setup_logging(args.model_name, "producer")
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))
    set_global_seed()

    config = load_config(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load your model    
    model = load_model(args.model_name, None).to(device)
    model.eval()

    # dataset + dataloader with padded batches
    ds = BinaryTokenDataset(
        data_dir=os.path.join(config["token_folder"], args.model_name, "binary"),
        num_classes=4
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # make sure output path exists
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    open(args.output_file, "a").close()  # touch

    logger.info(f"Streaming captions to {args.output_file}")
    clip_counter = 0
    with open(args.output_file, "a") as outf:
        for batch in tqdm(loader, desc="Generating batches"):
            # move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
            with torch.no_grad():
                captions = model.generate_answer(
                    inputs=inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            # append each result as NDJSON
            for i, caption in enumerate(captions):
                entry = {
                    "clip_idx":  clip_counter,
                    "class_idx": int(batch["class_idx"][i].item()),
                    "label":     batch["label"][i].tolist(),
                    "caption":   caption
                }
                line = json.dumps(entry, ensure_ascii=False)
                outf.write(line + "\n")
                outf.flush()
                os.fsync(outf.fileno())
                clip_counter += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    logger.info("Producer finished streaming all captions.")

if __name__ == "__main__":
    main()
