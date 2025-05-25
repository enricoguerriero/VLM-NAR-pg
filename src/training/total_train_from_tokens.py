# train_from_tokens_combined.py

from argparse import ArgumentParser
from src.utils import (
    load_model,
    setup_logging,
    setup_wandb,
    set_global_seed,
    load_config,
    compute_metrics,
    log_wandb,
    collate_fn_tokens
)
from src.data import TokenDataset
from torch.utils.data import DataLoader, ConcatDataset
import os
import torch
from tqdm import tqdm

def main():
    parser = ArgumentParser(description="Train a model from tokenized data (train+validation).")
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
    logger = setup_logging(model_name, "train_from_tokens_combined")
    config = load_config(model_name)
    wandb_run = setup_wandb(model_name, config)

    logger.info("─" * 40)
    logger.info(f"Starting COMBINED training for {model_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("─" * 40)

    # ─── Model ───────────────────────────────────────────────────────────────────
    model = load_model(model_name, checkpoint)
    logger.info("Model loaded.")

    # ─── Datasets ────────────────────────────────────────────────────────────────
    train_ds = TokenDataset(
        data_dir=os.path.join(config["token_folder"], model_name, "train")
    )
    val_ds = TokenDataset(
        data_dir=os.path.join(config["token_folder"], model_name, "validation")
    )
    combined_ds = ConcatDataset([train_ds, val_ds])
    logger.info(f"Combined dataset size: {len(combined_ds)} samples "
                f"({len(train_ds)} train + {len(val_ds)} validation)")

    combined_loader = DataLoader(
        combined_ds,
        batch_size=config["token_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn_tokens
    )

    # ─── Optimizer & Criterion ──────────────────────────────────────────────────
    optimizer = model.define_optimizer(
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"]
    )
    criterion = model.define_criterion(
        criterion_name=config["criterion"],
        pos_weight=train_ds.pos_weight  # still using train prior
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
    prior = train_ds.prior_probability.clamp_min(1e-6).clamp_max(1 - 1e-6)
    bias = torch.log(prior / (1 - prior))
    try:
        model.classifier.bias.data.copy_(bias)
    except AttributeError:
        model.classifier[-1].bias.data.copy_(bias)

    # ─── Training Loop ──────────────────────────────────────────────────────────
    best_loss = float("inf")
    best_wts = None

    epochs = config["epochs"]
    desc = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")
    for epoch in desc:
        # train
        train_loss, train_logits, train_labels = model.train_epoch(
            combined_loader, optimizer, criterion
        )
        train_metrics = compute_metrics(
            logits=train_logits,
            labels=train_labels,
            threshold=config["threshold"]
        )

        # scheduler step on training loss (or use a metric if preferred)
        try:
            scheduler.step(train_loss)
        except TypeError:
            scheduler.step()

        # log
        log_wandb(
            wandb_run,
            epoch,
            train_loss,
            None,              # no separate validation loss
            train_metrics,
            None               # no separate validation metrics
        )

        # progress bar
        desc.set_postfix({
            "loss": f"{train_loss:.4f}",
            "f1": f"{train_metrics['f1_macro']:.4f}"
        })

        # save best
        if train_loss < best_loss:
            best_loss = train_loss
            best_wts = {k: v.cpu() for k, v in model.state_dict().items()}

        # clear cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # checkpoint each epoch
        model.save_checkpoint(epoch, optimizer, scheduler)

    # ─── Finalize ───────────────────────────────────────────────────────────────
    logger.info("Training complete.")
    logger.info(f"Best training loss: {best_loss:.4f}")
    if best_wts is not None:
        model.load_state_dict(best_wts)
    model.save_model("_combined")
    wandb_run.finish()
    logger.info("Model saved. Bye!")

if __name__ == "__main__":
    main()
