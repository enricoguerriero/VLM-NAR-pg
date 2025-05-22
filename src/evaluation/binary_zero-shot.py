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
from src.data import BinaryTokenDataset
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
    
    test_dataset = BinaryTokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                args.model_name,
                                "binary",
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
    
    def judge_answer(answer, class_idx):
        
        prompts = [
            f"""
            You are a judge. You are given the description of a small clip from a VLM model. 
            The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
            The description given from the model is the following:
            '{answer}'
            According to the description, is there a baby / mannequin / doll visible in the clip?
            Note that, if the caption refers to a doll, it means that the mannequin is visible.
            Start your answer with "yes" or "no".
            """,
            f"""
            You are a judge. You are given the description of a small clip from a VLM model. 
            The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
            The description given from the model is the following:
            '{answer}'
            According to the description, are the health workers holding a ventilation mask over the baby's (or mannequin's) face? 
            The mask is supposed to cover the mouth and nose of the baby.
            Note that it has to be a ventilation mask, not just a tube.
            Start your answer with "yes" or "no".
            """,
            f"""
            You are a judge. You are given the description of a small clip from a VLM model.
            The clip is about the simulation of a newborn resuscitation. In the simulation, a mannequin is adopted to look like a baby.
            The description given from the model is the following:
            '{answer}'
            According to the description, are the health workers applying stimulation to the baby's (or mannequin's) back, buttocks (nates), or trunk, using up-and-down movements?
            The stimulation is supposed to be applied to the back, buttocks (nates), or trunk.
            Note that the stimulation can occur simultaneously with ventilation or suction.
            Start your answer with "yes" or "no".
            """,
            f"""
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
        output = judge_pipe(prompts[class_idx],
                            max_length=50, 
                            do_sample=False,
                            eos_token_id=judge_tokenizer.eos_token_id)
        out = output[0]["generated_text"]
        answer = out.split('[/INST]')[-1].strip()
        answer = answer.strip().splitlines()[-1].strip().lower()
        return answer.startswith("yes"), answer
    
    # Initialize counters
    TP = [0]*4; FP = [0]*4; TN = [0]*4; FN = [0]*4
    total = 0
    all_preds = []
    
    # ----------------------------
    # Main testing loop
    # ----------------------------
    for i, clip in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Testing clips"):
        
        class_idx = clip["class_idx"]
        label = clip["label"]
        
        # 1) Generate the caption/answer from your model
        caption = model.generate_answer(inputs=clip, max_new_tokens=128, do_sample=False)
        
        # 2) Ask the judge model for each label
        pred, answer = judge_answer(caption, class_idx)
        
        # 3) Update the counters
        if pred and label:
            TP[class_idx] += 1
        elif pred and not label:
            FP[class_idx] += 1
        elif not pred and label:
            FN[class_idx] += 1
        else:
            TN[class_idx] += 1
            
        # 4) Update the all preds list
        all_preds.append({
            "clip": i,
            "pred": pred,
            "label": label,
            "answer": answer,
            "caption": caption
        })
        total += 1
        
        logger.debug("-" * 20)
        logger.debug(f"Clip {i}: {clip}")
        logger.debug(f"Class: {LABELS[class_idx]}")
        logger.debug(f"Real label: {label}")
        logger.debug(f"Caption: {caption}")
        logger.debug(f"Answer: {answer}")
        logger.debug(f"Predicted label: {pred}")
        logger.debug("-" * 20)
    # ----------------------------
    
    
    # number of samples and classes
    N = total
    C = len(LABELS)

    logits = torch.zeros(N, C)
    labels = torch.zeros(N, C)

    for entry in all_preds:
        i        = entry["clip"]        # sample index
        c_idx    = entry["class_idx"]   # which of the 4 classes
        pred     = entry["pred"]        # bool – did judge say yes?
        truth    = entry["label"]       # bool – the gold label
        
        logits[i, c_idx] = float(pred)
        labels[i, c_idx] = float(truth)

    metrics = compute_metrics(
        logits = logits,
        labels = labels,
        threshold = 0.5
    )
    logger.info("Metrics computed successfully.")
    logger.info(f"f1 macro: {metrics['f1_macro']}")
    
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