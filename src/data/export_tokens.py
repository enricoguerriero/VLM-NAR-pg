from src.data import ClipDataset
from src.utils import (
    load_model,
    setup_logging,
    set_global_seed,
    load_config
)
from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch

def main():
    
    parser = ArgumentParser(description="Export tokens from a dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to use for tokenization.",
    )
    
    args = parser.parse_args()
    model_name = args.model_name
    
    set_global_seed()
    logger = setup_logging(model_name, "export_tokens")
    config = load_config(model_name)
    
    logger.info("-" * 20)
    logger.info(f"Starting token export for {model_name}")
    logger.info("-" * 20)
    
    logger.info("Loading model...")
    model = load_model(model_name, None)
    logger.info("Model loaded successfully.")
    
    logger.info("Loading dataset...")
    train_dataset = ClipDataset(
        video_folder = os.path.join(config["video_dir"],
                                    "train"),
        annotation_folder = os.path.join(config["annotation_dir"],
                                          "train"),
        clip_length = config["clip_length"],
        overlapping = config["overlapping"],
        frame_per_second = config["frame_per_second"]
    )
    logger.info("Train dataset loaded successfully.")
    validation_dataset = ClipDataset(
        video_folder = os.path.join(config["video_dir"],
                                    "validation"),
        annotation_folder = os.path.join(config["annotation_dir"],
                                          "validation"),
        clip_length = config["clip_length"],
        overlapping = config["overlapping"],
        frame_per_second = config["frame_per_second"]
    )
    logger.info("Validation dataset loaded successfully.")
    test_dataset = ClipDataset(
        video_folder = os.path.join(config["video_dir"],
                                    "test"),
        annotation_folder = os.path.join(config["annotation_dir"],
                                          "test"),
        clip_length = config["clip_length"],
        overlapping = config["overlapping"],
        frame_per_second = config["frame_per_second"]
    )
    logger.info("Test dataset loaded successfully.")
    
    logger.info("Exporting tokens...")
    os.makedirs(os.path.join(config["token_dir"],
                             model_name,
                             "train"), exist_ok=True)
    for i, clip in tqdm(enumerate(train_dataset), desc="Exporting train tokens"):
        frames = clip["frames"]
        labels = clip["labels"]
        
        np_frames = [f.permute(1,2,0).cpu().numpy() for f in frames]
        tokens = model.process_input(np_frames, 
                                     config["prompt"], 
                                     config["system_message"])
    
        file_name = os.path.join(config["token_dir"],
                           model_name,
                           "train",
                           f"clip_{i}.pt")
        torch.save({
            "tokens": tokens,
            "labels": labels
        }, file_name)
    logger.info("Train tokens exported successfully.")
    
    os.makedirs(os.path.join(config["token_dir"],
                                model_name,
                                "validation"), exist_ok=True)
    for i, clip in tqdm(enumerate(validation_dataset), desc="Exporting validation tokens"):
        frames = clip["frames"]
        labels = clip["labels"]
        
        np_frames = [f.permute(1,2,0).cpu().numpy() for f in frames]
        tokens = model.process_input(np_frames, 
                                     config["prompt"], 
                                     config["system_message"])
    
        file_name = os.path.join(config["token_dir"],
                           model_name,
                           "validation",
                           f"clip_{i}.pt")
        torch.save({
            "tokens": tokens,
            "labels": labels
        }, file_name)
    logger.info("Validation tokens exported successfully.")
    
    os.makedirs(os.path.join(config["token_dir"],
                                model_name,
                                "test"), exist_ok=True)
    for i, clip in tqdm(enumerate(test_dataset), desc="Exporting test tokens"):
        frames = clip["frames"]
        labels = clip["labels"]
        
        np_frames = [f.permute(1,2,0).cpu().numpy() for f in frames]
        tokens = model.process_input(np_frames, 
                                     config["prompt"], 
                                     config["system_message"])
    
        file_name = os.path.join(config["token_dir"],
                           model_name,
                           "test",
                           f"clip_{i}.pt")
        torch.save({
            "tokens": tokens,
            "labels": labels
        }, file_name)
    logger.info("Test tokens exported successfully.")
    
    
    
    
if __name__ == "__main__":
    main()