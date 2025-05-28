#!/usr/bin/env python
import json
import time
from argparse import ArgumentParser
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from dotenv import load_dotenv
from huggingface_hub import login
from src.utils import setup_logging
import re

PROMPTS = [
    """
    You are a judge. You are given the description of a small clip from a VLM model. 
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, is there a baby / mannequin / doll visible in the clip?
    Note that, if the caption refers to a doll, it means that the mannequin is visible.
    Start your answer with "yes" or "no".
    """,
    """
    You are a judge. You are given the description of a small clip from a VLM model. 
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, are the health workers holding a ventilation mask over the baby's (or mannequin's) face? 
    The mask is supposed to cover the mouth and nose of the baby.
    Note that it has to be a ventilation mask, not just a tube.
    Start your answer with "yes" or "no".
    """,
    """
    You are a judge. You are given the description of a small clip from a VLM model.
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, are the health workers applying stimulation to the baby's (or mannequin's) back, buttocks (nates), or trunk, using up-and-down movements?
    The stimulation is supposed to be applied to the back, buttocks (nates), or trunk.
    Note that the stimulation can occur simultaneously with ventilation or suction.
    Start your answer with "yes" or "no".
    """,
    """
    You are a judge. You are given the description of a small clip from a VLM model.
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, are the health workers inserting a small suction tube into the baby's (or mannequin's) mouth or nose?
    The suction tube is supposed to be inserted into the mouth or nose of the baby.
    Note that it has to be a tube, not a mask.
    Start your answer with "yes" or "no".
    """            
]

LABELS = ["Baby visible", "Ventilation", "Stimulation", "Suction"]

def main():
    parser = ArgumentParser("Consumer: judge captions stream")
    parser.add_argument("--input_file",  type=str, default="stream.ndjson")
    parser.add_argument("--output_file", type=str, default="predictions.ndjson")
    parser.add_argument("--poll_interval", type=float, default=1.0,
                        help="Seconds to wait when no new lines")
    args = parser.parse_args()

    logger = setup_logging("consumer", "binary_zero_shot")
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))

    # load judge model once
    JUDGE_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
    judge_model.eval()
    judge_pipe = TextGenerationPipeline(
        model=judge_model,
        tokenizer=judge_tokenizer,
        task="text-generation",
        pad_token_id=judge_tokenizer.eos_token_id,
        # device_map="auto"
    )

    # open files
    infile  = open(args.input_file, "r", encoding="utf-8")
    outfile = open(args.output_file, "a", encoding="utf-8")
    logger.info(f"Watching {args.input_file} → writing to {args.output_file}")

    # seek to end if we only want new items
    infile.seek(0, 2)
    pbar = tqdm(desc="Processed", unit="clip")

    while True:
        line = infile.readline()
        if not line:
            time.sleep(args.poll_interval)
            continue

        entry = json.loads(line)
        prompt = PROMPTS[entry["class_idx"]].replace("{answer}", entry["caption"])
        # run judge
        out = judge_pipe(prompt,
                         max_new_tokens=32,
                         do_sample=True,
                         batch_size=1)[0]["generated_text"].strip().lower()
        # extract yes/no
        match = re.search(r"\b(yes|no)\b", out)
        pred = match.group(1) == "yes" if match else False
        

        result = {
            **entry,
            "answer": out,
            "pred":   bool(pred)
        }
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        outfile.flush()
        pbar.update(1)

    # never reached, but for completeness:
    # infile.close()
    # outfile.close()

if __name__ == "__main__":
    main()
