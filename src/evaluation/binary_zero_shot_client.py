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
    # Baby visible (real baby vs. mannequin)
    '''You are a precise evaluator. Respond with yes or no.
    If you are uncertain, answer no.

    Definition of “yes”:
      A real baby *or* a mannequin baby is visible in the clip.
    Definition of “no”:
      No baby and no mannequin is visible.

    Example 1  
    Description: "A mannequin / a doll representing a baby is present on the table."  
    Answer: yes

    Example 2  
    Description: "There is a medical setting around a resuscitation table; no real baby or mannequin is present."  
    Answer: no

    Now evaluate:  
    Description: "{answer}"  
    Question: Is a subject visible?
    ''',

    # Ventilation (mask usage)
    """
    You are a precise evaluator. Answer only “yes” or “no”.

    Example 1:
      Description: “A mask covers the baby's mouth and nose to assist breathing / The baby is connected with a ventilator.”
      Answer: yes

    Example 2:
      Description: “A tube is inserted but no mask is used.”
      Answer: no

    Now evaluate:
    Description: '{answer}'
    Question: Is a ventilation mask held over the baby's (or mannequin's, or doll's) mouth and nose? Or is the baby (or mannequin, or doll) connected to a ventilator? (A mask, not a tube.)
    """,

    # Stimulation (up-and-down movements)
    """
    You are a precise evaluator. Answer only “yes” or “no”.

    Example 1:
      Description: “A health worker applies rhythmic up-and-down pressure on the baby's back.”
      Answer: yes

    Example 2:
      Description: “No stimulation: only mask ventilation is performed.”
      Answer: no

    Now evaluate:
    Description: '{answer}'
    Question: Are health workers applying up-and-down stimulation to the baby's (or mannequin's, or doll's) back, buttocks, or trunk?
    """,

    # Suction (small tube insertion)
    """
    You are a precise evaluator. Answer only “yes” or “no”.

    Example 1:
      Description: “A slender suction tube is inserted into the baby's mouth or nose.”
      Answer: yes

    Example 2:
      Description: “No tube insertion; only a mask is used.”
      Answer: no

    Now evaluate:
    Description: '{answer}'
    Question: Is a small suction tube inserted into the baby's (or mannequin's, or doll's) mouth or nose? (Not a mask.)
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
        return_full_text=False,
        clean_up_tokenization_spaces=True,
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
