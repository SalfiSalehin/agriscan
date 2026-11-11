"""Small numpy metrics (no sklearn): confusion matrix, per-class precision/recall/F1."""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros_like(tp), where=both > 0)
    return precision, recall, f1, support.astype(int)


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(int(cm.sum()), 1))


def macro_f1(cm: np.ndarray) -> float:
    """Mean F1 over classes that have at least one true sample."""
    _, _, f1, support = per_class_prf(cm)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0
