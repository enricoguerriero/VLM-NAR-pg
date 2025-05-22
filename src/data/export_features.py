from src.data import TokenDataset
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
    
    parser = ArgumentParser(description="Export features from a dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to use for feature extraction.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint file to load the model weights from. If not provided, the model will be loaded from the default path.",
    )
    
    args = parser.parse_args()
    model_name = args.model_name
    checkpoint = args.checkpoint
    
    set_global_seed()
    logger = setup_logging(model_name, "export_features")
    config = load_config(model_name)
    
    logger.info("-" * 20)
    logger.info(f"Starting feature export for {model_name}")
    logger.info("-" * 20)
    
    logger.info("Loading model...")
    model = load_model(model_name, checkpoint)
    logger.info("Model loaded successfully.")
    
    logger.info("Loading dataset...")
    train_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                model_name,
                                "train")
    )
    logger.info("Train dataset loaded successfully.")
    validation_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                model_name,
                                "validation")
    )
    logger.info("Validation dataset loaded successfully.")
    test_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                model_name,
                                "test")
    )
    logger.info("Test dataset loaded successfully.")
    
    logger.info("Creating data loaders...")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    logger.info("Train data loader created successfully.")
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    logger.info("Validation data loader created successfully.")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    logger.info("Test data loader created successfully.")
    
    logger.info("Exporting features...")
    model.save_features(
        train_dataloader,
        os.path.join(config["feature_folder"],
                     model_name,
                     "train")
    )
    logger.info("Train features exported successfully.")
    model.save_features(
        validation_dataloader,
        os.path.join(config["feature_folder"],
                     model_name,
                     "validation")
    )
    logger.info("Validation features exported successfully.")   
    model.save_features(
        test_dataloader,
        os.path.join(config["feature_folder"],
                     model_name,
                     "test")
    )
    logger.info("Test features exported successfully. Bye bye!")
    
if __name__ == "__main__":
    main()