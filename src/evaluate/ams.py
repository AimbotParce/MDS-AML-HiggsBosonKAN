import torch


def ams_score(y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor, br: float = 10.0) -> float:
    """
    Compute the Approximate Median Significance (AMS) metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels, where 1 or 's' indicates signal, 0 or 'b' indicates background.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (same encoding as y_true).
    weights : array-like of shape (n_samples,)
        Event weights for each observation.
    br : float, default=10.0
        Regularization term (background regularization constant).

    Returns
    -------
    ams : float
        The AMS metric value.
    """
    tp = torch.sum(weights[(y_true == 1) & (y_pred == 1)])
    fn = torch.sum(weights[(y_true == 1) & (y_pred == 0)])
    fp = torch.sum(weights[(y_true == 0) & (y_pred == 1)])
    tn = torch.sum(weights[(y_true == 0) & (y_pred == 0)])

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if fpr + br <= 0:
        return 0.0
    rad = 2 * ((tpr + fpr + br) * torch.log(1.0 + tpr / (fpr + br)) - tpr)
    return torch.sqrt(rad).item() if rad > 0 else 0.0
