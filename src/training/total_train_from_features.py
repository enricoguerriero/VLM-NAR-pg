# train_from_features_combined.py

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
from src.data import FeatureDataset
from torch.utils.data import DataLoader, ConcatDataset
import os
import torch
from tqdm import tqdm

def main():
    parser = ArgumentParser(description="Train a model from feature vector data (train+validation).")
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
    logger = setup_logging(model_name, "train_from_features_combined")
    config = load_config(model_name)
    wandb_run = setup_wandb(model_name, config)

    logger.info("─" * 40)
    logger.info(f"Starting COMBINED training for {model_name} from features")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("─" * 40)

    # ─── Model ───────────────────────────────────────────────────────────────────
    model = load_model(model_name, checkpoint)
    logger.info("Model loaded.")

    # ─── Datasets ────────────────────────────────────────────────────────────────
    train_ds = FeatureDataset(
        os.path.join(config["feature_folder"], model_name, "train")
    )
    val_ds = FeatureDataset(
        os.path.join(config["feature_folder"], model_name, "validation")
    )
    combined_ds = ConcatDataset([train_ds, val_ds])
    logger.info(
        f"Combined dataset size: {len(combined_ds)} "
        f"({len(train_ds)} train + {len(val_ds)} validation)"
    )

    combined_loader = DataLoader(
        combined_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # ─── Optimizer & Criterion ──────────────────────────────────────────────────
    optimizer = model.define_optimizer(
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"]
    )
    pos_weight, prior_probability = train_ds.weight_computation()
    criterion = model.define_criterion(
        criterion_name=config["criterion"],
        pos_weight=pos_weight.to(model.device)
    )

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

    # initialize classifier bias from prior
    bias = torch.log(prior_probability.clamp_min(1e-6).clamp_max(1 - 1e-6) /
                     (1 - prior_probability.clamp_min(1e-6)))
    try:
        model.classifier.bias.data.copy_(bias)
    except AttributeError:
        model.classifier[-1].bias.data.copy_(bias)

    # ─── Training Loop ──────────────────────────────────────────────────────────
    best_loss = float("inf")
    best_wts = None

    desc = tqdm(range(1, config["epochs"] + 1), desc="Epochs", unit="epoch")
    for epoch in desc:
        # train on combined data
        train_loss, train_logits, train_labels = model.train_classifier_epoch(
            combined_loader, optimizer, criterion
        )
        train_metrics = compute_metrics(
            logits=train_logits,
            labels=train_labels,
            threshold=config["threshold"]
        )

        # step scheduler on training loss (or switch to train_metrics if desired)
        try:
            scheduler.step(train_metrics['f1_macro'])
        except TypeError:
            scheduler.step()

        # log to wandb
        log_wandb(
            wandb_run,
            epoch,
            train_loss,
            None,           # no separate val_loss
            train_metrics,
            None            # no separate val_metrics
        )

        # update progress bar
        desc.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_f1":    f"{train_metrics['f1_macro']:.4f}"
        })

        # track best
        if train_loss < best_loss:
            best_loss = train_loss
            best_wts = {k: v.cpu() for k, v in model.state_dict().items()}

        # clear cache & checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        model.save_checkpoint(epoch, optimizer, scheduler)

    # ─── Finalize ───────────────────────────────────────────────────────────────
    logger.info("Training complete.")
    logger.info(f"Best combined training loss: {best_loss:.4f}")
    if best_wts is not None:
        model.load_state_dict(best_wts)
    model.save_model("_features_combined")
    wandb_run.finish()
    logger.info("Model saved. Bye!")

if __name__ == "__main__":
    main()
