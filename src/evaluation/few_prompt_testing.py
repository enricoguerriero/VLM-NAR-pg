#!/usr/bin/env python
"""
few_shot_evaluation_vlm.py

Single-script to test original judge prompts on sampled clips,
using our VLMBinaryClipDataset (one binary example per clip/class).
"""
import os
import random
import json
from dotenv import load_dotenv
from huggingface_hub import login

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from src.utils import set_global_seed, load_model
from src.data import VLMBinaryClipDataset   # <-- our new dataset

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_CAPTION = "VideoLLaVA"
MODEL_JUDGE   = "mistralai/Mistral-7B-Instruct-v0.1"
NUM_SAMPLES   = 40
SEED          = 42

# Original prompts for each class
PROMPTS = [
    # Baby visible (real baby vs. mannequin)
    """
    You are a precise evaluator. Answer only “yes” or “no”.

    Example 1:
      Description: “A real newborn lying on a table with no mannequin visible.”
      Answer: yes

    Example 2:
      Description: “A lifelike training mannequin is on the table; no real baby is present.”
      Answer: no

    Now evaluate:
    Description: '{answer}'
    Question: Is a real baby visible in this clip? (Mannequin does not count.)
    """,

    # Ventilation (mask usage)
    """
    You are a precise evaluator. Answer only “yes” or “no”.

    Example 1:
      Description: “A mask covers the baby's mouth and nose to assist breathing.”
      Answer: yes

    Example 2:
      Description: “A tube is inserted but no mask is used.”
      Answer: no

    Now evaluate:
    Description: '{answer}'
    Question: Is a ventilation mask held over the baby's (or mannequin's) mouth and nose? (A mask, not a tube.)
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
    Question: Are health workers applying up-and-down stimulation to the baby's (or mannequin's) back, buttocks, or trunk?
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
    Question: Is a small suction tube inserted into the baby's (or mannequin's) mouth or nose? (Not a mask.)
    """
]

SYSTEM_MESSAGE = "You are assisting in a newborn resuscitation simulation. The video is recorded from above a resuscitation table. A mannequin representing a newborn baby may or may not be present. Based on the visual evidence, respond to the following questions. Be explicit and unambiguous."

CAPTION_PROMPTS = [
    "Describe the scene in the clip and give it a caption. You should see the table. On the table, there may be either a real baby / mannequin or nothing.",
    "Describe the scene and give it a caption. Is the baby or mannequin visible on the table? If yes, is a health worker holding a large ventilation mask over the mannequin's face, covering both mouth and nose? This action supports breathing and is distinct from tube insertion. Be explicit. If there is not a baby / mannequin visible, no treatment is being performed.",
    "Describe the scene and give it a caption. Is the baby or mannequin visible on the table? If yes, is a health worker performing up-and-down stimulation on the mannequin's back, buttocks, or trunk? These are small, quick movements. Be clear and specific. If there is not a baby / mannequin visible, no treatment is being performed.",
    "Describe the scene and give it a caption. Is the baby or mannequin visible on the table? If yes, is a health worker inserting a small tube into the mouth or nose of the mannequin to provide suction? This cannot occur at the same time as mask ventilation. Be explicit. If there is not a baby / mannequin visible, no treatment is being performed."
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
    )

    processor = caption_model.processor

    # Build the VLMBinaryClipDataset
    ds = VLMBinaryClipDataset(
        video_folder=os.path.join("data", "videos"),
        annotation_folder=os.path.join("data", "annotations"),
        clip_length=5,
        overlapping=0.5,
        frame_per_second=2,
        num_classes=4,
        system_message=SYSTEM_MESSAGE,   
        prompts=CAPTION_PROMPTS,
        processor=processor,
        transform=None
    )

    # Sample some indices
    all_idxs = list(range(len(ds)))
    sampled = random.Random(SEED).sample(all_idxs, k=min(NUM_SAMPLES, len(ds)))

    results = []
    for idx in sampled:
        sample = ds[idx]

        true_label = sample["label"].item()
        class_idx  = sample["class_idx"].item()

        # Prepare inputs for caption model exactly as before
        inputs = {
            "input_ids":        sample["input_ids"].unsqueeze(0).to(device),
            "attention_mask":   sample["attention_mask"].unsqueeze(0).to(device),
            "pixel_values_videos": sample["pixel_values_videos"].unsqueeze(0).to(device)
        }

        # Generate caption
        with torch.no_grad():
            caption = caption_model.generate_answer(
                inputs=inputs,
                max_new_tokens=128,
                do_sample=False
            )

        print(f"[{idx}] Caption: {repr(caption)}")

        # Fill in the judge prompt
        prompt = PROMPTS[class_idx].replace("{answer}", caption.replace("\n", " "))

        # Run judge
        out_raw = judge_pipe(
            prompt,
            max_new_tokens=15,
            do_sample=False,
        )[0]["generated_text"].strip().lower()
        pred = out_raw.startswith("yes")

        result = {
            "clip_idx":   idx,
            "class_idx":  class_idx,
            "caption":    caption,
            "true_label": true_label,
            "judge_raw":  out_raw,
            "prediction": pred
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        results.append(result)

    # Save to JSON
    with open("few_shot_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluated {len(results)} examples. Saved to few_shot_results.json")

if __name__ == "__main__":
    main()
