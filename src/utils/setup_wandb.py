import wandb
import time
import numpy as np
import matplotlib.pyplot as plt
import io

def plot_confusion_matrix(matrix, class_name):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {class_name}")

    # Show values inside the boxes
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(buf, caption=f"Confusion matrix for {class_name}")

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
    Log metrics to Weights & Biases.
    """
    class_names = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    final_log = {}
    
    for i, class_name in enumerate(class_names):
        matrix = np.array([
            [train_metrics["TN"][i], train_metrics["FP"][i]],
            [train_metrics["FN"][i], train_metrics["TP"][i]]
        ])
        final_log[f"train/{class_name}/Confusion_matrix"] =  plot_confusion_matrix(matrix, class_name)
        final_log[f"train/{class_name}/accuracy"] = train_metrics["accuracy"][i]
        final_log[f"train/{class_name}/precision"] = train_metrics["precision"][i]
        final_log[f"train/{class_name}/recall"] = train_metrics["recall"][i]
        final_log[f"train/{class_name}/f1"] = train_metrics["f1"][i]
        
        matrix = np.array([
            [val_metrics["TN"][i], val_metrics["FP"][i]],
            [val_metrics["FN"][i], val_metrics["TP"][i]]
        ])
        final_log[f"val/{class_name}/Confusion_matrix"] =  plot_confusion_matrix(matrix, class_name)
        final_log[f"val/{class_name}/accuracy"] = val_metrics["accuracy"][i]
        final_log[f"val/{class_name}/precision"] = val_metrics["precision"][i]
        final_log[f"val/{class_name}/recall"] = val_metrics["recall"][i]
        final_log[f"val/{class_name}/f1"] = val_metrics["f1"][i]
    final_log["epoch"] = epoch
    final_log["train/loss"] = train_loss
    if val_loss is not None:
        final_log["val/loss"] = val_loss
    final_log["train/accuracy_macro"] = train_metrics["accuracy_macro"]
    final_log["train/precision_macro"] = train_metrics["precision_macro"]
    final_log["train/recall_macro"] = train_metrics["recall_macro"]
    final_log["train/f1_macro"] = train_metrics["f1_macro"]
    if val_metrics is not None:
        final_log["val/accuracy_macro"] = val_metrics["accuracy_macro"]
        final_log["val/precision_macro"] = val_metrics["precision_macro"]
        final_log["val/recall_macro"] = val_metrics["recall_macro"]
        final_log["val/f1_macro"] = val_metrics["f1_macro"]
    wandb_run.log(final_log, step=epoch)

def log_test_wandb(
    wandb_run,
    test_metrics: dict,
    test_loss: float = None
):
    """
    Log test metrics to Weights & Biases.
    """
    class_names = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    final_log = {}
    for i, class_name in enumerate(class_names):
        matrix = np.array([
            [test_metrics["TN"][i], test_metrics["FP"][i]],
            [test_metrics["FN"][i], test_metrics["TP"][i]]
        ])
        final_log[f"test/{class_name}/Confusion_matrix"] = wandb.Image(matrix, caption=f"Confusion matrix for {class_name}")
        final_log[f"test/{class_name}/accuracy"] = test_metrics["accuracy"][i]
        final_log[f"test/{class_name}/precision"] = test_metrics["precision"][i]
        final_log[f"test/{class_name}/recall"] = test_metrics["recall"][i]
        final_log[f"test/{class_name}/f1"] = test_metrics["f1"][i]
    final_log["test/accuracy_macro"] = test_metrics["accuracy_macro"]
    final_log["test/precision_macro"] = test_metrics["precision_macro"]
    final_log["test/recall_macro"] = test_metrics["recall_macro"]
    final_log["test/f1_macro"] = test_metrics["f1_macro"]
    if test_loss is not None:
        final_log["test/loss"] = test_loss
    wandb_run.log(final_log)