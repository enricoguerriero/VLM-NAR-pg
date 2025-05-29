from src.data import VLMVideoMultiLabelDataset
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
from torch.utils.data import DataLoader
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from tqdm import tqdm
import copy

def main():
    parser = ArgumentParser(description="Train a model from video clips.")
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

    # ─── Setup ───────────────────────────────────────────────────────────────────
    set_global_seed()
    logger = setup_logging(model_name, "train_from_clips")
    config = load_config(model_name)
    wandb_run = setup_wandb(model_name, config)

    logger.info("─" * 40)
    logger.info(f"Starting training for {model_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("─" * 40)

    # ─── Model ───────────────────────────────────────────────────────────────────
    model = load_model(model_name, checkpoint)
    logger.info("Model loaded.")

    # ─── Datasets ────────────────────────────────────────────────────────────────
    prompt = model.prompt_definition(config["prompt"], config["system_message"])
    logger.info("Loading datasets...")
    train_ds = VLMVideoMultiLabelDataset(
        csv_file = config["train_csv"],
        processor = model.processor,
        prompt = prompt,
        # system_message = config["system_message"],
        frames = config["frames"],
        frame_sample = config["frame_sample"],
    )
    validation_ds = VLMVideoMultiLabelDataset(
        csv_file = config["validation_csv"],
        processor = model.processor,
        prompt = prompt,
        # system_message = config["system_message"],
        frames = config["frames"],
        frame_sample = config["frame_sample"],
    )
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(validation_ds)}")
    
    # ─── DataLoaders ─────────────────────────────────────────────────────────────
    collate_fn = model.build_video_colllate_fn()
    logger.info("Creating DataLoaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
    )
    logger.info(f"Train DataLoader created with {len(train_loader)} batches.")
    validation_loader = DataLoader(
        validation_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
    )
    logger.info(f"Validation DataLoader created with {len(validation_loader)} batches.")
    
    # ─── Training Loop ───────────────────────────────────────────────────────────
    logger.info("Starting training loop...")
    
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
        pos_weight= train_ds.pos_weight 
    )
    logger.info(f"Criterion {config['criterion']} defined.")
    logger.info(f"Pos weight: {train_ds.pos_weight}")
    scheduler = model.define_scheduler(
        scheduler_name=config["scheduler"],
        optimizer=optimizer,
        epochs=config["epochs"],
        patience=config["scheduler_patience"],
        step_size=config["step_size"],
        gamma=config["gamma"],
        eta_min=config["eta_min"],
        factor=config["factor"],
        mode=config["mode"],
        cooldown=config["cooldown"],
        min_lr=config["min_lr"]
    )
    logger.info(f"Scheduler {config['scheduler']} defined.")
    logger.info(f"Patience: {config['scheduler_patience']}")
    logger.info(f"Step size: {config['step_size']}")
    logger.info(f"Gamma: {config['gamma']}")
    logger.info(f"Eta min: {config['eta_min']}")
    logger.info(f"Factor: {config['factor']}")
    logger.info(f"Mode: {config['mode']}")
    logger.info(f"Cooldown: {config['cooldown']}")
    logger.info(f"Min LR: {config['min_lr']}")
    prior = train_ds.prior.clamp_min(1e-6).clamp_max(1-1e-6)
    logger.info(f"Prior: {prior}")
    bias = torch.log(prior / (1 - prior))
    logger.info(f"Bias: {bias}")
    try:
        model.classifier.bias.data.copy_(bias)
    except AttributeError:
        model.classifier[-1].bias.data.copy_(bias)
    
    unfreezing = model.set_freezing_condition(config["freezing_condition"])
    logger.info(f"Unfreezing condition set to: {unfreezing}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    best_val_loss = float("inf")
    best_wts = None
    no_impovement = 0
    epochs = config["epochs"]
    desc = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")
    
    threshold = config["threshold"]
    if isinstance(threshold, list):
        threshold = torch.tensor(threshold, dtype=torch.float32)
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Patience: {config['patience']}")
    
    for epoch in desc:
        
        logger.debug(f"Starting epoch {epoch}/{epochs}")
        model.train()
        
        if unfreezing:
            model.manage_unfreezing(
                epoch=epoch,
                epochs=epochs,
                optimizer=optimizer,
                scheduler=scheduler
            )
            logger.debug(f"Unfreezing model at epoch {epoch}")
            logger.debug(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        logger.debug("Training...")
        train_loss, train_logits, train_labels = model.train_epoch(
            train_loader, optimizer, criterion
        )
        logger.debug(f"Train loss: {train_loss:.4f}")
        desc.set_postfix({"train_loss": train_loss})
        logger.debug("Computing train metrics...")
        train_metrics = compute_metrics(
            logits=train_logits.detach().cpu(),
            labels=train_labels.detach().cpu(),
            threshold=threshold.detach().cpu()
        )
        logger.debug(f"Train metrics: {train_metrics}")
        desc.set_postfix({"train_f1": train_metrics["f1_macro"]})
        
        logger.debug("Validating...")
        val_loss, val_logits, val_labels = model.eval_epoch(
            validation_loader, criterion
        )
        logger.debug(f"Validation loss: {val_loss:.4f}")
        desc.set_postfix({"val_loss": val_loss})
        logger.debug("Computing validation metrics...")
        val_metrics = compute_metrics(
            logits=val_logits.detach().cpu(),
            labels=val_labels.detach().cpu(),
            threshold=threshold.detach().cpu()  
        )
        logger.debug(f"Validation metrics: {val_metrics}")
        desc.set_postfix({"val_f1": val_metrics["f1_macro"]})
        
        try:
            scheduler.step(val_metrics["f1_macro"])  # for ReduceLROnPlateau
        except AttributeError:
            scheduler.step()
        logger.debug(f"Scheduler step done. New learning rate: {optimizer.param_groups[0]['lr']}")
        
        log_wandb(
            wandb_run,
            epoch=epoch,
            train_loss=train_loss,
            train_metrics=train_metrics,
            val_loss=val_loss,
            val_metrics=val_metrics
        )
        logger.debug("Logged metrics to Weights & Biases.")
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug("Cleared CUDA cache.")
        
        model.save_checkpoint(
            epoch,
            optimizer,
            scheduler
        )
        logger.debug("Checkpoint saved.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            logger.debug(f"New best validation loss: {best_val_loss:.4f}. Saving best weights.")
            no_impovement = 0
        else:
            no_impovement += 1
            logger.debug(f"No improvement in validation loss. No improvement count: {no_impovement}")
            if no_impovement >= config["patience"]:
                logger.info(f"No improvement for {config['patience']} epochs. Stopping training.")
                break
    logger.info("Training completed.")
    if best_wts is not None:
        model.load_state_dict(best_wts)
        logger.info("Loaded best model weights.")
    else:
        logger.warning("No best weights found. Model may not have improved.")
    model.save_model("_from_clips")
    logger.info("Model saved after training.")
    
    wandb_run.finish()
    logger.info("Weights & Biases run finished, bye bye.")
    
if __name__ == "__main__":
    main()