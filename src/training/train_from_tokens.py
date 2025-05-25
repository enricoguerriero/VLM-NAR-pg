from argparse import ArgumentParser
from src.utils import (
    load_model,
    setup_logging,
    setup_wandb,
    set_global_seed,
    load_config,
    compute_metrics,
    log_wandb
)
from src.data import TokenDataset
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import copy

def main():
    
    parser = ArgumentParser(description="Train a model from tokenized data.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to train.",
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
    logger = setup_logging(model_name, "train_from_tokens")
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
    train_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                model_name,
                                "train")
    )
    logger.info("Train dataset loaded successfully.")
    val_dataset = TokenDataset(
        data_dir = os.path.join(config["token_folder"],
                                model_name,
                                "validation")
    )
    logger.info("Validation dataset loaded successfully.")
    
    logger.info("Loading dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["token_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    logger.info("Train dataloader loaded successfully.")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["token_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    logger.info("Validation dataloader loaded successfully.")
    logger.info("-" * 20)
    
    logger.info(f"Starting training the model {model_name} for {config["epochs"]} epochs...")
    
    best_val_loss = float("inf")
    no_improvement_count = 0
    
    optimizer = model.define_optimizer(
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"]
    )
    logger.info(f"Optimizer {config['optimizer']} defined.")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Weight decay: {config['weight_decay']}")
    logger.info(f"Momentum: {config['momentum']}")
    
    criterion = model.define_criterion(
        criterion_name=config["criterion"],
        pos_weight=train_dataset.pos_weight
    )
    logger.info(f"Criterion {config['criterion']} defined.")
    logger.info(f"Pos weight: {train_dataset.pos_weight}")
    
    scheduler = model.define_scheduler(
        scheduler_name = config["scheduler"],
        optimizer = optimizer,
        epochs = config["epochs"],
        patience = config["scheduler_patience"],
        step_size = config["step_size"],
        gamma = config["gamma"],
        eta_min = config["eta_min"],
        factor = config["factor"],
        mode = config["mode"],
        cooldown = config["cooldown"],
        min_lr = config["min_lr"]
    )
    logger.info(f"Scheduler {config['scheduler']} defined.")
    logger.info(f"Scheduler patience: {config['scheduler_patience']}, ")
    logger.info(f"Step size: {config['step_size']}, ")
    logger.info(f"Gamma: {config['gamma']}, ")
    logger.info(f"Eta min: {config['eta_min']}, ")
    logger.info(f"Factor: {config['factor']}, ")
    logger.info(f"Mode: {config['mode']}, ")
    logger.info(f"Cooldown: {config['cooldown']}, ")
    logger.info(f"Min lr: {config['min_lr']}")
    logger.info("-" * 20)
    
    unfreezing = model.set_freezing_condition(config["freezing_condition"])
    logger.info(f"Unfreezing condition: {unfreezing}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("-" * 20)
    
    prior_probability = train_dataset.prior_probability.clamp_min(1e-6).clamp_max(1-1e-6)
    bias = torch.log(prior_probability / (1 - prior_probability))
    logger.info(f"Prior probability: {prior_probability}")
    logger.info(f"Bias: {bias}")
    logger.info("-" * 20)
    
    threshold = config["threshold"]
    if isinstance(threshold, list):
        threshold = torch.tensor(threshold, dtype=torch.float32)
    
    try:
        model.classifier.bias.data.copy_(bias) # 1 layer classifier
    except AttributeError:
        model.classifier[-1].bias.data.copy_(bias) # multi layer classifier
        
    epochs_iter = tqdm(range(1, config["epochs"] + 1), desc="Epochs", unit="epoch")
    
    for epoch in epochs_iter:
        
        logger.debug(f"Epoch {epoch}/{config['epochs']}")
        
        if unfreezing:
            model.manage_unfreezing(epoch,
                                    config["epochs"],
                                    optimizer,
                                    scheduler
            )
            logger.debug(f"Unfreezing model at epoch {epoch}")
            logger.debug(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        logger.debug("Training...")
        train_loss, train_logits, train_labels = model.train_epoch(
            train_dataloader,
            optimizer,
            criterion)
        logger.debug(f"Train loss: {train_loss:.4f}")
        log_msg = f"Train loss: {train_loss:.4f}"
        
        train_metrics = compute_metrics(
            logits = train_logits,
            labels = train_labels,
            threshold = threshold
        )
        logger.debug(f"Train metrics: {train_metrics}")
        log_msg += f", Train f1: {train_metrics['f1_macro']:.4f}"
        
        logger.debug("Validating...")
        val_loss, val_logits, val_labels = model.eval_epoch(
            val_dataloader,
            criterion)
        logger.debug(f"Validation loss: {val_loss:.4f}")
        log_msg += f", Validation loss: {val_loss:.4f}"
        
        val_metrics = compute_metrics(
            logits = val_logits,
            labels = val_labels,
            threshold = threshold
        )
        logger.debug(f"Validation metrics: {val_metrics}")
        log_msg += f", Validation f1: {val_metrics['f1_macro']:.4f}"
        
        try:
            scheduler.step(val_metrics["f1_macro"]) # for ReduceLROnPlateau
        except AttributeError:
            scheduler.step()
        logger.debug(f"Scheduler step: {scheduler.get_last_lr()}")
        
        epochs_iter.set_postfix_str(log_msg)
        
        log_wandb(
            wandb_run,
            epoch,
            train_loss,
            val_loss,
            train_metrics,
            val_metrics
        )
        logger.debug("Logged to wandb.")
        
        model.save_checkpoint(
            epoch,
            optimizer,
            scheduler
        )
        logger.debug("Checkpoint saved.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improvement_count = 0
            logger.debug("Best model weights updated.")
        else:
            no_improvement_count += 1
            logger.debug(f"No improvement count: {no_improvement_count}")
            if no_improvement_count >= config["patience"]:
                logger.info(f"Early stopping at epoch {epoch}, no improvement for {no_improvement_count} epochs.")
                break
    
    logger.info("Training completed.")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)
    model.save_model("_tokens")
    logger.info("Best model saved.")
    
    wandb_run.finish()
    logger.info("Wandb run finished, bye bye!")
    
if __name__ == "__main__":
    main()