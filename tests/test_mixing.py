import numpy as np
import pytest

from ml import mixing as mx


def test_sample_weights_give_the_requested_field_share():
    w = mx.sample_weights(1000, 100, 0.25)
    assert w.sum() == pytest.approx(1.0)
    assert w[1000:].sum() == pytest.approx(0.25)
    assert w[0] < w[-1]  # a field photo is drawn far more often than a lab photo


def test_sample_weights_validation():
    assert mx.sample_weights(10, 0, 0.0).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mx.sample_weights(10, 0, 0.25)
    with pytest.raises(ValueError):
        mx.sample_weights(10, 5, 1.0)
    with pytest.raises(ValueError):
        mx.sample_weights(0, 5, 0.2)


def test_sampling_matches_expectation():
    rng = np.random.default_rng(0)
    w = mx.sample_weights(900, 100, 0.3)
    draws = rng.choice(1000, size=50_000, p=w)
    assert (draws >= 900).mean() == pytest.approx(0.3, abs=0.01)


def test_effective_class_weights_reduce_to_plain_weights_without_field():
    lab = [0] * 90 + [1] * 10
    got = mx.effective_class_weights(lab, [], 2, 0.0, 0.5)
    counts = np.array([90.0, 10.0])
    assert got == pytest.approx((counts.sum() / (2 * counts)) ** 0.5)  # ml.data.class_weights


def test_effective_class_weights_use_the_mix():
    lab = [0] * 90 + [1] * 10
    field = [1] * 10
    # half of all draws are field photos, all of them class 1 -> classes are balanced in the mix
    got = mx.effective_class_weights(lab, field, 2, 0.5, 1.0)
    p = mx.expected_class_share(lab, field, 2, 0.5)
    assert p == pytest.approx([0.45, 0.55])
    assert got[0] == pytest.approx(1 / (2 * 0.45))
    with pytest.raises(ValueError):
        mx.effective_class_weights([0, 0], [], 2, 0.0)


def test_balanced_loss():
    assert mx.balanced_loss(0.2, None) == 0.2
    assert mx.balanced_loss(0.2, 1.0) == pytest.approx(0.6)
    assert mx.balanced_loss(0.2, 1.0, field_weight=0.25) == pytest.approx(0.4)
