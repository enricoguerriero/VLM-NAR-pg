#!/usr/bin/env python
"""
few_shot_evaluation_vlm.py
Evaluate clips with the VLMVideoDataset (4-prompt version).
"""
import os, random, json
from pathlib import Path
import re

from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from src.utils import set_global_seed, load_model
from src.data import VLMVideoDataset        

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_CAPTION = "LLaVANeXT"                                    # producer VLM
MODEL_JUDGE   = "mistralai/Mistral-7B-Instruct-v0.1"           # text judge
NUM_SAMPLES   = 40
SEED          = 42
DEVICE = "cuda" # for slurm
CAPTION_DEVICE = torch.device("cuda") 
JUDGE_DEVICE   = torch.device("cuda")

PROMPTS = [
		# Baby visible (real baby vs. mannequin)
		'''You are a precise evaluator. Respond with **just** yes or no.
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
		Question: Is a baby (real **or** mannequin, or doll) visible?
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

# Path to the labels file generated by build_resus_dataset.py
LABELS_CSV = Path("data/clips/test/labels.csv")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
		# env + reproducibility
		load_dotenv()
		login(os.getenv("HUGGINGFACE_TOKEN"))
		# set_global_seed(SEED)

		device = torch.device(DEVICE)

		# ── Load caption model (producer) ───────────────────────────────────────
		caption_model = load_model(MODEL_CAPTION, checkpoint=None)
		# caption_model.to(device).eval() # slurm 
		caption_model.to(CAPTION_DEVICE).eval() 
		caption_processor = caption_model.processor

		# ── Build dataset (each row = one (clip, prompt) pair) ─────────────────
		ds = VLMVideoDataset(
				csv_file       = LABELS_CSV,
				processor      = caption_processor,
				prompts        = ["USER: Describe the clip; focus on who is present in the clip: <video>. ASSISTANT:",
													"USER: Describe the clip; focus on eventual respiration equipment and how is it eventually used: <video>. ASSISTANT:",
													"USER: Describe the clip; if the baby / doll is being stimulated, describe also that movement: <video>. ASSISTANT:",
													"USER: Describe the clip; if a suction tube is present, describe that and how it is used. Note that the suction tube is different from a ventilation mask: <video>. ASSISTANT:"],   
				system_message = "This is a simulation of a medical resuscitation context.",                             
				frames         = 8,
				frame_sample   = "uniform",
		)
		# print some dataset info
		print(f"Loaded {len(ds)} samples from {LABELS_CSV} with {len(ds.label_cols)} label columns.")
		print(f"Dataset length: {len(ds)}")
		print(f"Dataset clips with baby visible: {ds.df[ds.label_cols[0]].sum()}")  # Baby visible
		print(f"Dataset clips with ventilation: {ds.df[ds.label_cols[1]].sum()}")  # Ventilation
		print(f"Dataset clips with stimulation: {ds.df[ds.label_cols[2]].sum()}")  # Stimulation
		print(f"Dataset clips with suction: {ds.df[ds.label_cols[3]].sum()}")      # Suction
		
		# ── Load judge (text-only) ──────────────────────────────────────────────
		judge_tok  = AutoTokenizer.from_pretrained(MODEL_JUDGE)
		judge_lm   = AutoModelForCausalLM.from_pretrained(
				MODEL_JUDGE,
				device_map="cuda",
				torch_dtype=torch.float16,
		)
		judge_lm.eval()
		judge_pipe = TextGenerationPipeline(
				model = judge_lm,
				tokenizer = judge_tok,
				task = "text-generation",
				pad_token_id = judge_tok.eos_token_id,
				return_full_text = False,
				clean_up_tokenization_spaces = True,
		)

		# ── Sampling ────────────────────────────────────────────────────────────
		subset = ds.balanced_sample(10)

		# Map label-name → index so we can reuse PROMPTS list
		label2idx = {name: i for i, name in enumerate(ds.label_cols)}

		results = []
		for local_idx, sample in enumerate(subset):
			ds_idx = subset.indices[local_idx]

			# — ground-truth scalar (0./1.) —
			true_label  = float(sample["label"].item())
			label_name  = sample["label_name"]
			class_idx   = label2idx[label_name]

			# — caption model inputs —
			cap_inputs = {
					"input_ids": sample["input_ids"].unsqueeze(0).to(device),
					"attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
			}
			# detect video / image key
			vid_key = "pixel_values_videos" if "pixel_values_videos" in sample else "pixel_values"
			cap_inputs[vid_key] = sample[vid_key].unsqueeze(0).to(device)

			# — generate caption —
			with torch.no_grad():
					caption = caption_model.generate_answer(
							inputs = cap_inputs,
							max_new_tokens = 128,
							do_sample = False,
					).strip()

			# — build judge prompt —
			judge_prompt = PROMPTS[class_idx].replace("{answer}", caption.replace("\n", " "))

			# — run judge LLM —
			judge_raw = judge_pipe(
					judge_prompt,
					max_new_tokens = 15,
					do_sample = False,
			)[0]["generated_text"].strip().lower()
			match = re.search(r"\b(yes|no)\b", judge_raw)
			pred = match.group(1) == "yes" if match else False

			# — log result —
			rec = dict(
					dataset_idx = ds_idx,
					file        = sample["file"],
					class_idx   = class_idx,
					class_name  = label_name,
					caption     = caption,
					true_label  = true_label,
					judge_raw   = judge_raw,
					prediction  = pred,
			)
			print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)
			results.append(rec)

		# ── Save JSON ───────────────────────────────────────────────────────────
		with open("few_shot_results_vlm.json", "w", encoding="utf-8") as f:
				json.dump(results, f, ensure_ascii=False, indent=2)

		print(f"✓ Evaluated {len(results)} samples → few_shot_results_vlm.json")


if __name__ == "__main__":
		main()
