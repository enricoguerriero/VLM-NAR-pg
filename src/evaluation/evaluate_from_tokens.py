from argparse import ArgumentParser
from src.utils import (
    load_model,
    setup_logging,
    setup_wandb,
    set_global_seed,
    load_config,
    compute_metrics,
    log_test_wandb,
    collate_fn_tokens
)
from src.data import TokenDataset
import os
from torch.utils.data import DataLoader


def main():
    
    parser = ArgumentParser(description="Evaluate a model from tokenized data.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint.",
    )
    
    args = parser.parse_args()
    model_name = args.model_name
    checkpoint = args.checkpoint
    
    set_global_seed()
    logger = setup_logging(model_name, "evaluate_from_tokens")
    config = load_config(model_name)
    wandb_run = setup_wandb(model_name, config)
    
    logger.info("-" * 20)
    logger.info(f"Starting training for {model_name} from tokens")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("-" * 20)
    
    logger.info("Loading model...")
    model = load_model(model_name, checkpoint)
    logger.info("Model loaded successfully.")
    
    logger.info("Loading datasets...")
    test_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                model_name,
                                "test")
    )
    logger.info("Test dataset loaded successfully.")
    
    logger.info("Creating dataloaders...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["tokens_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn_tokens
    )
    logger.info("Dataloaders created successfully.")
    logger.info("-" * 20)
    
    logger.info(f"Starting evaluation of {model_name}...")
    
    criterion = model.define_criterion(
        criterion_name = config["criterion"],
        pos_weight=test_dataset.pos_weight
    )
    
    test_loss, test_logits, test_labels = model.eval_epoch(
        test_dataloader,
        criterion
    )
    logger.info("Evaluation completed successfully.")
    
    test_metrics = compute_metrics(
        test_logits,
        test_labels,
        config["threshold"]
    )
    logger.info("Metrics computed successfully.")
    logger.info(f"Test Loss: {test_loss}, test f1: {test_metrics['f1_macro']}")
    
    log_test_wandb(
        wandb_run,
        test_loss,
        test_metrics
    )
    logger.info("Metrics logged to wandb successfully.")
    wandb_run.finish()
    logger.info("-" * 20)
    logger.info(f"Evaluation of {model_name} completed successfully, bye bye!")
    logger.info("-" * 20)
    
if __name__ == "__main__":
    main()