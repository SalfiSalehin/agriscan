"""Torch-free arithmetic for training on a mix of lab photos and real-world photos.

PlantVillage (lab) has ~13,000 training images, PlantDoc's training photos only a few hundred.
Plain shuffling would show the model a field photo about once in 40 draws. Instead every epoch
draws as many images as the PlantVillage training split has, but a fixed share of the draws
(`share`) comes from the field photos, which are therefore repeated (with random augmentation).
"""

import numpy as np


def sample_weights(n_lab: int, n_field: int, share: float) -> np.ndarray:
    """Per-image sampling weights for a [lab..., field...] list so that, in expectation,
    a share `share` of all draws is a field photo."""
    if not 0.0 <= share < 1.0:
        raise ValueError("share must be in [0, 1)")
    if n_lab <= 0:
        raise ValueError("need at least one lab image")
    if share > 0 and n_field <= 0:
        raise ValueError("share > 0 but there are no field images")
    weights = np.full(n_lab + n_field, 0.0, dtype=np.float64)
    weights[:n_lab] = (1.0 - share) / n_lab
    if n_field:
        weights[n_lab:] = share / n_field
    return weights


def expected_class_share(lab_labels, field_labels, num_classes: int, share: float) -> np.ndarray:
    """Expected fraction of each class among the sampled images."""
    lab = np.bincount(np.asarray(lab_labels, dtype=int), minlength=num_classes).astype(float)
    field = np.bincount(np.asarray(field_labels, dtype=int), minlength=num_classes).astype(float)
    p = (1.0 - share) * lab / max(lab.sum(), 1.0)
    if share > 0 and field.sum() > 0:
        p = p + share * field / field.sum()
    return p


def effective_class_weights(
    lab_labels, field_labels, num_classes: int, share: float, power: float = 0.5
) -> np.ndarray:
    """Inverse-frequency weights ** power, computed on the mix that is actually sampled
    (the same formula as ml.data.class_weights, which this reduces to when share = 0)."""
    p = expected_class_share(lab_labels, field_labels, num_classes, share)
    if (p <= 0).any():
        raise ValueError("a class has no training images")
    return ((1.0 / (num_classes * p)) ** power).astype(np.float32)


def balanced_loss(lab_loss: float, field_loss: float | None, field_weight: float = 0.5) -> float:
    """Model-selection loss: the lab validation loss alone, or a weighted mean with the field
    validation loss. Without the field term a model that forgets the real world would still
    win on the (much easier) lab validation set."""
    if field_loss is None:
        return float(lab_loss)
    return float((1.0 - field_weight) * lab_loss + field_weight * field_loss)
