#!/usr/bin/env python
"""
few_shot_evaluation.py

Single-script to test original judge prompts on sampled clips.
"""
import os
import random
import json
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from src.utils import set_global_seed, load_model
from src.data import BinaryTokenDataset

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_CAPTION = "VideoLLaVA"
MODEL_JUDGE   = "mistralai/Mistral-7B-Instruct-v0.1"
NUM_SAMPLES   = 40
SEED          = 42

# Original prompts for each class
PROMPTS = [
    """
    You are a judge. You are given the description of a small clip from a VLM model.
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, is there a baby / mannequin / doll visible in the clip?
    Note that, if the caption refers to a doll, it means that the mannequin is visible.
    Start your answer with \"yes\" or \"no\".
    """,
    """
    You are a judge. You are given the description of a small clip from a VLM model.
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, are the health workers holding a ventilation mask over the baby's (or mannequin's) face?
    The mask is supposed to cover the mouth and nose of the baby.
    Note that it has to be a ventilation mask, not just a tube.
    Start your answer with \"yes\" or \"no\".
    """,
    """
    You are a judge. You are given the description of a small clip from a VLM model.
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, are the health workers applying stimulation to the baby's (or mannequin's) back, buttocks (nates), or trunk, using up-and-down movements?
    The stimulation is supposed to be applied to the back, buttocks (nates), or trunk.
    Note that the stimulation can occur simultaneously with ventilation or suction.
    Start your answer with \"yes\" or \"no\".
    """,
    """
    You are a judge. You are given the description of a small clip from a VLM model.
    The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
    The description given from the model is the following:
    '{answer}'
    According to the description, are the health workers inserting a small suction tube into the baby's (or mannequin's) mouth or nose?
    The suction tube is supposed to be inserted into the mouth or nose of the baby.
    Note that it has to be a tube, not a mask.
    Start your answer with \"yes\" or \"no\".
    """
]

def main():
    # env + reproducibility
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))
    set_global_seed(SEED)

    # Load caption model (producer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_model = load_model(MODEL_CAPTION, None).to(device)
    caption_model.eval()

    # Load judge pipeline (client)
    judge_tokenizer = AutoTokenizer.from_pretrained(MODEL_JUDGE)
    judge_model = AutoModelForCausalLM.from_pretrained(
        MODEL_JUDGE,
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
        device_map="auto"
    )

    # Load dataset and sample indices
    ds = BinaryTokenDataset(
        data_dir=os.path.join("data/tokens", MODEL_CAPTION, "binary"),
        num_classes=4
    )
    all_idxs = list(range(len(ds)))
    sampled = random.Random(SEED).sample(all_idxs, k=min(NUM_SAMPLES, len(ds)))

    results = []
    for idx in sampled:
        sample = ds[idx]
        # Generate caption
        inputs = {
            "input_ids": sample["input_ids"].to(device),
            "attention_mask": sample["attention_mask"].to(device),
            "pixel_values_videos": sample["pixel_values_videos"].to(device)
        }
        with torch.no_grad():
            caption = caption_model.generate_answer(
                inputs=inputs,
                max_new_tokens=128,
                do_sample=False
            )[0]

        # Select and fill prompt
        class_idx = int(sample["class_idx"].item())
        prompt = PROMPTS[class_idx].replace("{answer}", caption.replace("\n", " "))

        # Run judge
        out = judge_pipe(
            prompt,
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=judge_tokenizer.eos_token_id,
            pad_token_id=judge_tokenizer.eos_token_id,
            stop=["\n"]
        )[0]["generated_text"].strip().lower().split()[0]
        pred = (out == "yes")

        result = {
            "clip_idx": idx,
            "class_idx": class_idx,
            "caption": caption,
            "judge_raw": out,
            "prediction": pred
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)

    # Save to JSON
    with open("few_shot_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluated {len(results)} clips. Saved to few_shot_results.json")

if __name__ == "__main__":
    main()
