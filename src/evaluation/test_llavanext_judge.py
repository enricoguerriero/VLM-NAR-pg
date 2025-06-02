import argparse
import json
from typing import Dict, List, Set

import av                      
import numpy as np             
import torch                   
from tqdm.auto import tqdm     
from sklearn.metrics import accuracy_score, precision_recall_fscore_support   
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import re

MODEL_JUDGE   = "mistralai/Mistral-7B-Instruct-v0.1"

PROMPTS = [
		# Baby visible (real baby vs. mannequin)
		'''You are a precise evaluator. Respond with **just** yes or no.
        You are given a caption provided by a captioning model of a video, in a newborn resuscitation simulation context.
        You need to determine if a baby is visible in the video based on the caption.

		Definition of “yes”:
			A real baby *or* a mannequin baby is visible in the clip.
		Definition of “no”:
			No baby and no mannequin is visible.

		Example 1  
		Description: "A mannequin / a doll representing a baby is present on the table."  
		Answer: yes

		Example 2  
		Description: "There is a medical setting around a resuscitation table; no baby or mannequin is present."  
		Answer: no

		Now evaluate:  
		Description: "{answer}"  
		Question: Is a baby (real **or** mannequin, or doll) visible?
		''',

		# Ventilation (mask usage)
		"""
		You are a precise evaluator. Answer only “yes” or “no”.
        You are given a caption provided by a captioning model of a video, in a newborn resuscitation simulation context.
        You need to determine if a ventilation mask is being used in the video based on the caption.

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
        You are given a caption provided by a captioning model of a video, in a newborn resuscitation simulation context.
        You need to determine if health workers are applying up-and-down stimulation to the baby's back, buttocks, or trunk based on the caption.

		Example 1:
			Description: “A health worker applies rhythmic up-and-down pressure on the baby's back / performing CPR.”
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
        You are given a caption provided by a captioning model of a video, in a newborn resuscitation simulation context.
        You need to determine if a small suction tube is inserted into the baby's mouth or nose based on the caption.

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


# ---------- Video helpers -------------------------------------------------- #

def _read_video_pyav(filepath: str, num_frames: int = 8) -> np.ndarray:
    """Decode *num_frames* RGB frames, uniformly sampled across the clip."""
    container = av.open(filepath)
    total = container.streams.video[0].frames
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != num_frames:     # Pad if video too short
        frames.extend(frames[-1:] * (num_frames - len(frames)))
    return np.stack(frames)            # (T, H, W, 3)


# ---------- Prompt helpers ------------------------------------------------- #

def _natural_name(label: str) -> str:
    """Convert snake_case to capitalised natural language ("baby_visible" -> "Baby visible")."""
    words = label.split("_")
    return " ".join([words[0].capitalize()] + words[1:])


def build_prompt(label_names: List[str]) -> str:
    """Return the single instruction prompt."""
    natural = [_natural_name(lbl) for lbl in label_names]
    joined = ", ".join(natural)
    return (
         f"You will be shown a short video clip of a newborn resuscitation simulation.\n"
        f"Decide which of the following actions are happening: {joined}.\n\n"
        "Rules:\n"
        "- No actions can happen if the baby is not visible (baby is considered visible if the doll is visible).\n"
        "- Stimulation can happen with either ventilation or suction.\n"
        "- Ventilation and suction cannot happen at the same time.\n\n"
        "Reply with a comma-separated list of actions (exact names). If none appear, reply `none`."
    )


def build_conversation(prompt: str) -> List[Dict]:
    """Wrap user prompt + video into LLaVA chat format."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        }
    ]


# ---------- Prediction per clip ------------------------------------------- #
@torch.inference_mode()
def caption_video(
    model,
    processor,
    video: np.ndarray,
    prompt: str,
    label_names: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
) -> Dict[str, int]:
    
    conv = build_conversation(prompt)
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt_text, videos=video, return_tensors="pt").to(device)
    
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,            # greedy decoding for determinism
        temperature=0.4,
    )
    decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(f"Response: {decoded}")
    
    reply = decoded.split("ASSISTANT:")[-1].strip()
    return reply
     
def predict_from_caption(
    caption: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    label_names: List[str]
):
    pass


# ---------- Main ----------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Zero‑shot multi‑label evaluation with a single list prompt and minimal output."
    )
    parser.add_argument("--jsonl", required=True, help="Path to dataset jsonl.")
    parser.add_argument("--model", default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="HF model id or local path.")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled per clip.")
    parser.add_argument("--half", action="store_true", help="Load model in fp16 (saves GPU memory).")
    args = parser.parse_args()

    # Load dataset
    with open(args.jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    label_names = list(samples[0]["labels"].keys())
    prompt = '''
    You are in a simulation of a newborn resuscitation scenario.
    You will be shown a short video clip of a newborn resuscitation simulation.
    You need to caption the video, describing exactly:
    - Who is present in the video
    - What actions are being performed
    - What objects are being used
    Be explicit and precise in your description, focus on medical details.'''
    
    # Model / processor
    dtype = torch.float16 if args.half else torch.float32
    device = torch.device("cuda:1")
    judge_device = torch.device("cuda:5")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto"
    )
    processor = LlavaNextVideoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model.eval()
    
    judge = AutoModelForCausalLM.from_pretrained(MODEL_JUDGE,
				torch_dtype=torch.float16,).to(judge_device)
    judge_processor = AutoTokenizer.from_pretrained(MODEL_JUDGE)
    judge_pipe = TextGenerationPipeline(
            model = judge,
            tokenizer = judge_processor,
            task = "text-generation",
            device=judge_device.index,
            pad_token_id = judge_processor.eos_token_id,
            return_full_text = False,
            clean_up_tokenization_spaces = True,
    )

    # Prediction loop -------------------------------------------------------
    y_true = {lbl: [] for lbl in label_names}
    y_pred = {lbl: [] for lbl in label_names}
    captions = []
    results = []

    for sample in tqdm(samples, desc="Evaluating", unit="clip"):
        print(f"\n[VIDEO] {sample['video']}", flush=True)
        video = _read_video_pyav(sample["video"], args.num_frames)

        caption = caption_video(model, processor, video, prompt, label_names, device)
        captions.append(caption)
        print(f"Caption: {caption}", flush=True)
        
        true_label = sample["labels"]
        
        for class_idx, label in enumerate(label_names):
            judge_prompt = PROMPTS[class_idx].replace("{answer}", caption.replace("\n", " ")).to(judge_device)
            judge_raw = judge_pipe(
                    judge_prompt,
                    max_new_tokens = 15,
                    do_sample = False,
            )[0]["generated_text"].strip().lower()
            match = re.search(r"\b(yes|no)\b", judge_raw)
            pred = match.group(1) == "yes" if match else False
            y_pred[label].append(pred)
            y_true[label].append(true_label[label])
            
            rec = dict(
					file        = sample["video"],
					class_idx   = class_idx,
					class_name  = label,
					caption     = caption,
					true_label  = true_label[label],
					judge_raw   = judge_raw,
					prediction  = pred,
			)
            print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)
            results.append(rec)
        

    # Metrics ---------------------------------------------------------------
    print("\n===== Per‑class metrics =====", flush=True)
    macro_prec, macro_rec, macro_f1 = [], [], []
    for lbl in label_names:
        acc = accuracy_score(y_true[lbl], y_pred[lbl])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[lbl], y_pred[lbl], average="binary", zero_division=0
        )
        macro_prec.append(prec)
        macro_rec.append(rec)
        macro_f1.append(f1)
        print(f"{lbl:15s}  acc={acc:5.3f}  prec={prec:5.3f}  rec={rec:5.3f}  f1={f1:5.3f}", flush=True)

    print("\n===== Macro‑average =====", flush=True)
    print(
        f"acc={np.mean([accuracy_score(y_true[l], y_pred[l]) for l in label_names]):5.3f}  "
        f"prec={np.mean(macro_prec):5.3f}  rec={np.mean(macro_rec):5.3f}  f1={np.mean(macro_f1):5.3f}", flush=True
    )


if __name__ == "__main__":
    main()
