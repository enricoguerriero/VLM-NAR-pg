#!/usr/bin/env python
import json
import os
import time
from argparse import ArgumentParser

import torch
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
    # adjust this if your clips have different fields
    collated = {}
    for k in batch[0]:
        collated[k] = torch.stack([b[k] for b in batch], dim=0)
    return collated

def main():
    parser = ArgumentParser("Producer: generate captions stream")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="stream.ndjson",
                        help="Where to append caption entries")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.0,
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

    # dataset + dataloader
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

    # ensure output file exists
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    open(args.output_file, "a").close()

    logger.info(f"Streaming captions to {args.output_file}")
    clip_counter = 0
    with open(args.output_file, "a") as outf:
        for batch in tqdm(loader, desc="Generating batches"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                captions = model.generate_answer(inputs=inputs,
                                                 max_new_tokens=128,
                                                 do_sample=False)

            # write each entry as one line JSON
            for i, caption in enumerate(captions):
                entry = {
                    "clip_idx":   clip_counter,
                    "class_idx":  int(batch["class_idx"][i].item()),
                    "label":      batch["label"][i].tolist(),
                    "caption":    caption
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
