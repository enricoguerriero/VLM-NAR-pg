import torch
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float | torch.Tensor = 0.5):
    
    """
    Compute TP/FP/FN/TN and derived metrics for a multi-label task.
    """
    if threshold is None:
        thr = 0.5
    else:
        thr = threshold

    probs = torch.sigmoid(logits)
    
    if not isinstance(thr, torch.Tensor):
        thr = torch.tensor(thr, dtype=probs.dtype, device=probs.device)
    else:
        thr = thr.to(device=probs.device, dtype=probs.dtype)

    if thr.ndim == 0:
        thr = thr.expand_as(probs)
    elif thr.shape != probs.shape:
        thr = thr.expand(probs.shape)

    preds = (probs >= thr).int()
    truths = labels.int()
    
    if truths.dim() == 3:
        n, m, C = truths.shape
        preds = preds.view(n * m, C)
        truths = truths.view(n * m, C)

    preds = preds.cpu().numpy()
    truths = truths.cpu().numpy()
    
    TP = ((preds == 1) & (truths == 1)).sum(axis=0)
    FP = ((preds == 1) & (truths == 0)).sum(axis=0)
    FN = ((preds == 0) & (truths == 1)).sum(axis=0)
    TN = ((preds == 0) & (truths == 0)).sum(axis=0)
    
    total = TP + FP + FN + TN
    acc_per_class = (TP + TN) / total.clip(min=1)
    
    # precision/recall/f1 via sklearn
    p, r, f1, _ = precision_recall_fscore_support(
        truths,
        preds,
        average=None,
        zero_division=0
    )

    metrics = {
        "logits": logits,
        "labels": labels,
        "probs": probs,
        "preds": preds,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy":      acc_per_class,
        "precision":     p,
        "recall":        r,
        "f1":            f1,
        "accuracy_macro":  acc_per_class.mean().item(),
        "precision_macro": p.mean().item(),
        "recall_macro":    r.mean().item(),
        "f1_macro":        f1.mean().item()
    }
    return metrics
