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
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import copy
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import torch.nn as nn
import wandb
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_NAME = "Newborn_Activity_Recognition_optimization"

def objective(trial: optuna.Trial) -> float:
    
    set_global_seed()
    
    logger.info(f"Trial number: {trial.number}")
    model_copy = copy.deepcopy(model).to(DEVICE)
    for param in model_copy.backbone.parameters():
        param.requires_grad = False
    
    optimizer_name = trial.suggest_categorical('optimizer_name', ['adam', 'sgd', 'adamw'])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 1, 10)
    
    momentum = None
    if optimizer_name == 'sgd':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
    threshold = torch.tensor([
        trial.suggest_float('threshold_1', 0.3, 0.7),
        trial.suggest_float('threshold_2', 0.3, 0.7),
        trial.suggest_float('threshold_3', 0.3, 0.7),
        trial.suggest_float('threshold_4', 0.3, 0.7)
    ]).to(DEVICE)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    patience = 0 # already optimizing epochs
    
    num_classes = 4 # hardcoded but for all models the following line works
    # num_classes = model.classifier[-1].out_features
    backbone_hidden = model_copy.backbone.config.text_config.hidden_size
    model_copy.classifier = nn.Sequential(
        nn.Linear(backbone_hidden*2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_classes)
    ).to(DEVICE)
    pos_weight, prior_prob = train_dataset.weight_computation()
    pos_weight = pos_weight.to(DEVICE)
    prior_prob = prior_prob.to(DEVICE)
    model_copy.classifier[-1].bias.data.copy_(torch.log(prior_prob / (1 - prior_prob)))

    criterion = model_copy.define_criterion(
        criterion_name='wbce',
        pos_weight=pos_weight
    )
    optimizer = model_copy.define_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum
    )
    scheduler_name = trial.suggest_categorical('scheduler_name', ['steplr', 'cosineannealinglr', 'reduceonplateau'])
    step_size = None
    gamma = None
    eta_min = None
    scheduler_patience = None
    factor = None
    cooldown = None
    min_lr = None
    if scheduler_name == 'steplr':
        step_size = trial.suggest_int('step_size', 1, 3)
        gamma = trial.suggest_float('gamma', 0.1, 0.9) 
    elif scheduler_name == 'cosineannealinglr':
        eta_min = trial.suggest_float('eta_min', 0.0, 1e-3)
    elif scheduler_name == 'reduceonplateau':
        factor = trial.suggest_float('factor', 0.1, 0.9)
        scheduler_patience = trial.suggest_int('patience', 1, 3)
        cooldown = trial.suggest_int('cooldown', 0, 5)
        min_lr = trial.suggest_float('min_lr', 1e-6, 1e-2, log=True)

    scheduler = model_copy.define_scheduler(
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        epochs=epochs,
        patience=scheduler_patience,
        step_size=step_size,
        gamma=gamma,
        eta_min=eta_min,
        factor=factor,
        cooldown=cooldown,
        min_lr=min_lr
    )
    
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    epochs_iter = tqdm(range(epochs), desc="Training epochs", unit="epoch")
    
    for epoch in epochs_iter:
        
        logger.debug(f"Epoch {epoch + 1}/{epochs}")
        
        logger.debug("Training...")
        train_loss, train_logits, train_labels = model_copy.train_classifier_epoch(
            train_loader,
            optimizer,
            criterion
        )
        log_msg = f"Train loss: {train_loss:.4f}"
        logger.debug(f"Train loss: {train_loss:.4f}")
        
        train_metrics = compute_metrics(train_logits, train_labels, threshold)
        logger.debug(f"Train metrics: {train_metrics}")
        log_msg += f", Train f1: {train_metrics['f1_macro']:.4f}"
        
        logger.debug("Validating...")
        val_loss, val_logits, val_labels = model_copy.eval_classifier_epoch(
            val_loader,
            criterion
        )
        logger.debug(f"Validation loss: {val_loss:.4f}")
        log_msg += f", Val loss: {val_loss:.4f}"
        
        val_metrics = compute_metrics(val_logits, val_labels, threshold)
        logger.debug(f"Validation metrics: {val_metrics}")
        log_msg += f", Val f1: {val_metrics['f1_macro']:.4f}"

        try:
            scheduler.step(val_loss) # ReduceLROnPlateau
        except AttributeError:
            scheduler.step()
        epochs_iter.set_postfix_str(log_msg)
        # log_wandb(
        #     wandb_run,
        #     epoch + 1,
        #     train_loss,
        #     val_loss,
        #     train_metrics,
        #     val_metrics
        # )
        logger.debug("Logged to wandb.")
        
        # 1. Report intermediate objective
        trial.report(val_metrics["f1_macro"], step=epoch)

        # 2. Handle pruning
        if trial.should_prune():
            # gracefully stop this trial
            raise optuna.exceptions.TrialPruned()
        
    return val_metrics['f1_macro']

if __name__ == "__main__":
    
    parser = ArgumentParser(description="Hyperparameter optimization for model training.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to optimize.",
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
    
    logger = setup_logging(model_name, "hyperparameter_optimization")
    config = load_config(model_name)
    wandb_run = setup_wandb(model_name, config)
    
    logger.info("-" * 20)
    logger.info(f"Starting hyperparameter optimization for {model_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info("-" * 20)
    
    # Data
    logger.info("Loading datasets...")
    train_dataset = FeatureDataset(
        data_dir=os.path.join(config["feature_folder"], model_name, "train")
    )
    logger.info("Train dataset loaded successfully.")
    val_dataset = FeatureDataset(
        data_dir=os.path.join(config["feature_folder"], model_name, "validation")
    )
    logger.info("Validation dataset loaded successfully.")
    
    # Load model
    model = load_model(model_name, checkpoint)
    
    STUDY_NAME = f"newborn_activity_recognition_{model_name}"
    db_dir = os.path.join("data", "db", model_name)
    os.makedirs(db_dir, exist_ok=True)
    DB_PATH = f"sqlite:///{os.path.join(db_dir, 'optuna_newborn.db')}"
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )
    
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="val_f1_macro",
        wandb_kwargs={"project": PROJECT_NAME, "reinit": True}
    )
    
    # Optimize the objective function
    study.optimize(objective, n_trials=100, callbacks=[wandb_callback])
    
    wandb_run.log({
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params
    })
    
    # Visualize results
    wandb_run.log({
        "optimization_history": plot_optimization_history(study),
        "param_importances": plot_param_importances(study),
        "parallel_coordinate": plot_parallel_coordinate(study),
        "slice_plot": plot_slice(study),
        "contour_plot": plot_contour(study)
    })
    
    logger.info("Hyperparameter optimization completed.")