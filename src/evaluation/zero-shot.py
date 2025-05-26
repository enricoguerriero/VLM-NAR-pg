from argparse import ArgumentParser
from src.utils import (
    load_model,
    setup_logging,
    setup_wandb,
    set_global_seed,
    load_config,
    compute_metrics,
    log_test_wandb
)
from src.data import TokenDataset
import os
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline
)
import torch
import json
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

def main():
    
    parser = ArgumentParser(description="Test NAR model with 0-shot prompting (per-label questions).")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to test.")
    args = parser.parse_args()

    logger = setup_logging(args.model_name, "binary_zero_shot")
    logger.info("Starting 0-shot per-label test script.")
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))
    
    set_global_seed()
    config = load_config(args.model_name)
    config["test_type"] = "0-shot-per-label"
    wandb_run = setup_wandb(args.model_name, 
                            config)
    
    logger.info("-" * 20)
    logger.info("loading model...")
    model = load_model(args.model_name, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loaded model on {device}")
    logger.info("-" * 20)
    
    test_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                args.model_name,
                                "test",
                                ),
        num_classes = 4
    )
    logger.info("Test dataset loaded successfully.")
    
    JUDGE_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME)
    judge_model = AutoModelForCausalLM.from_pretrained(JUDGE_NAME, device_map="auto")
    judge_model.eval()
    judge_pipe = TextGenerationPipeline(
        model = judge_model,
        tokenizer = judge_tokenizer,
        task = "text-generation",
        pad_token_id = judge_tokenizer.eos_token_id
    )
    logger.info("Judge model loaded successfully.")
    logger.info("-" * 20)
    
    LABELS = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    
    PROMPT = """[INST]
        You are given an image caption. For each of these labels: {labels}, output a JSON object with keys as label names and boolean values indicating presence.
        Respond with exactly the JSON object and no other text.
        Caption: "{caption}"
        [/INST]"""
        
    all_entries = []
    prompts = []
    
    for idx, clip in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Preparing prompts"):
        class_idx = clip["class_idx"]
        label = clip["label"]

        # generate caption
        caption = model.generate_answer(inputs=clip, max_new_tokens=128, do_sample=False)

        # build judge prompt
        prompt = PROMPTS[class_idx].replace("{answer}", caption)
        prompts.append(prompt)

        all_entries.append({
            "clip_idx": idx,
            "class_idx": int(class_idx.item()),
            "label": label.tolist(),
            "caption": caption
        })
        
    checkpoint_path = os.path.join("data/captions", f"{args.model_name}_zero_shot_checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w") as ckpt_f:
        json.dump({
            "entries": all_entries,
            "prompts": prompts
        }, ckpt_f, indent=2)
    logger.info(f"Checkpoint saved with {len(all_entries)} entries to {checkpoint_path}")
    
    logger.info(f"Running judge pipeline on {len(prompts)} prompts in batches...")
    outputs = judge_pipe(
        prompts,
        max_new_tokens=32,
        do_sample=True,
        batch_size=16,       # adjust to fit your GPU memory
        eos_token_id=judge_tokenizer.eos_token_id
    )

    
    TP = [0]*4; FP = [0]*4; TN = [0]*4; FN = [0]*4
    all_preds = []
    total = len(all_entries)

    for entry, out in zip(all_entries, outputs):
        raw = out["generated_text"]
        answer = raw.split('[/INST]')[-1].strip().splitlines()[-1].lower()
        pred = answer.startswith("yes")

        c = entry["class_idx"]
        truth = entry["label"]

        if pred and truth:      TP[c] += 1
        elif pred and not truth: FP[c] += 1
        elif not pred and truth: FN[c] += 1
        else:                    TN[c] += 1

        all_preds.append({
            **entry,
            "pred": pred,
            "answer": answer
        })
    
    
    N = total
    C = len(LABELS)
    logits = torch.zeros(N, C)
    labels = torch.zeros(N, C)

    for p in all_preds:
        i, c_idx = p["clip_idx"], p["class_idx"]
        logits[i, c_idx] = float(p["pred"])
        labels[i, c_idx] = float(p["label"])

    metrics = compute_metrics(logits=logits, labels=labels, threshold=0.5)
    logger.info(f"Metrics computed successfully. F1 macro: {metrics['f1_macro']:.4f}")
    
    log_test_wandb(
        wandb_run,
        0,
        metrics
    )
    wandb_run.finish()
    logger.info("Metrics logged to wandb successfully.")
    logger.info("-" * 20)
    logger.info(f"Evaluation of {args.model_name} completed successfully, bye bye!")
    logger.info("-" * 20)
    
if __name__ == "__main__":
    main()