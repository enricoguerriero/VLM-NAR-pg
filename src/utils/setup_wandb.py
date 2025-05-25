import wandb
import time

def setup_wandb(model_name: str, config: dict):
    """
    Set up Weights & Biases for logging.
    """
    run = wandb.init(
        project = "Newborn_Activity_Recognition",
        name = f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}',
        config = config,
        resume = "allow")
    
    return run


def log_wandb(
    wandb_run,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: dict,
    val_metrics: dict
):
    """
    Log metrics to Weights & Biases using wandb.plot.confusion_matrix.
    """
    class_names = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    final_log = {}
    
    # helper to flatten labels tensor → numpy
    def _get_flat(arr, i):
        # arr might be torch.Tensor or numpy; ensure numpy and shape (N, C)
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        return arr.reshape(-1, len(class_names))[:, i].tolist()
    
    # TRAIN
    for i, class_name in enumerate(class_names):
        y_true = _get_flat(train_metrics["labels"], i)
        y_pred = train_metrics["preds"][:, i].tolist()
        
        cm = wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=["Negative", "Positive"]
        )
        final_log[f"train/{class_name}/confusion_matrix"] = cm
        final_log[f"train/{class_name}/accuracy"]  = train_metrics["accuracy"][i]
        final_log[f"train/{class_name}/precision"] = train_metrics["precision"][i]
        final_log[f"train/{class_name}/recall"]    = train_metrics["recall"][i]
        final_log[f"train/{class_name}/f1"]        = train_metrics["f1"][i]
    
    # VALIDATION
    if val_metrics is not None:
        for i, class_name in enumerate(class_names):
            y_true = _get_flat(val_metrics["labels"], i)
            y_pred = val_metrics["preds"][:, i].tolist()
            
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=["Negative", "Positive"]
            )
            final_log[f"val/{class_name}/confusion_matrix"] = cm
            final_log[f"val/{class_name}/accuracy"]  = val_metrics["accuracy"][i]
            final_log[f"val/{class_name}/precision"] = val_metrics["precision"][i]
            final_log[f"val/{class_name}/recall"]    = val_metrics["recall"][i]
            final_log[f"val/{class_name}/f1"]        = val_metrics["f1"][i]
    
    # epoch losses & macros
    final_log["epoch"]               = epoch
    final_log["train/loss"]          = train_loss
    if val_loss is not None:
        final_log["val/loss"]        = val_loss
    
    final_log["train/accuracy_macro"]  = train_metrics["accuracy_macro"]
    final_log["train/precision_macro"] = train_metrics["precision_macro"]
    final_log["train/recall_macro"]    = train_metrics["recall_macro"]
    final_log["train/f1_macro"]        = train_metrics["f1_macro"]
    if val_metrics is not None:
        final_log["val/accuracy_macro"]  = val_metrics["accuracy_macro"]
        final_log["val/precision_macro"] = val_metrics["precision_macro"]
        final_log["val/recall_macro"]    = val_metrics["recall_macro"]
        final_log["val/f1_macro"]        = val_metrics["f1_macro"]
    
    wandb_run.log(final_log, step=epoch)


def log_test_wandb(
    wandb_run,
    test_metrics: dict,
    test_loss: float = None
):
    """
    Log test metrics to Weights & Biases using wandb.plot.confusion_matrix.
    """
    class_names = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    final_log = {}
    
    def _get_flat(arr, i):
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        return arr.reshape(-1, len(class_names))[:, i].tolist()
    
    for i, class_name in enumerate(class_names):
        y_true = _get_flat(test_metrics["labels"], i)
        y_pred = test_metrics["preds"][:, i].tolist()
        
        cm = wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=["Negative", "Positive"]
        )
        final_log[f"test/{class_name}/confusion_matrix"] = cm
        final_log[f"test/{class_name}/accuracy"]  = test_metrics["accuracy"][i]
        final_log[f"test/{class_name}/precision"] = test_metrics["precision"][i]
        final_log[f"test/{class_name}/recall"]    = test_metrics["recall"][i]
        final_log[f"test/{class_name}/f1"]        = test_metrics["f1"][i]
    
    final_log["test/accuracy_macro"]  = test_metrics["accuracy_macro"]
    final_log["test/precision_macro"] = test_metrics["precision_macro"]
    final_log["test/recall_macro"]    = test_metrics["recall_macro"]
    final_log["test/f1_macro"]        = test_metrics["f1_macro"]
    if test_loss is not None:
        final_log["test/loss"]        = test_loss
    
    wandb_run.log(final_log)