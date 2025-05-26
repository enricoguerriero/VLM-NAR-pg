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
MODEL_CAPTION = "LLaVANeXT"
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
        # device_map="auto"
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
        
        true_label = sample["label"].tolist()
        
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
            )
        
        print(f"[{idx}] Caption: {repr(caption)}")

        # Select and fill prompt
        class_idx = int(sample["class_idx"].item())
        prompt = PROMPTS[class_idx].replace("{answer}", caption.replace("\n", " "))

        # Run judge
        out_raw = judge_pipe(
            prompt,
            max_new_tokens=15,
            do_sample=False,
            # eos_token_id=judge_tokenizer.eos_token_id,
            # pad_token_id=judge_tokenizer.eos_token_id,
            # stop=["\n"]
        )[0]["generated_text"].strip().lower()
        pred = out_raw.startswith("yes")

        result = {
            "clip_idx": idx,
            "class_idx": class_idx,
            "caption": caption,
            "true_label": true_label,
            "judge_raw": out_raw,
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
