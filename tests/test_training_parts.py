import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.augment import RandomBackground
from ml.data import ListDataset, ManifestDataset, class_weights
from ml.models import build_model, set_backbone_trainable, set_train_mode, split_params
from ml.transforms import build_eval_transform, build_train_transform


def test_class_weights_favour_rare_classes():
    labels = [0] * 90 + [1] * 10
    full = class_weights(labels, 2, power=1.0)
    assert abs(full[1] / full[0] - 9) < 1e-4
    softer = class_weights(labels, 2, power=0.5)
    assert abs(softer[1] / softer[0] - 3) < 1e-4


def test_class_weights_reject_empty_class():
    with pytest.raises(ValueError):
        class_weights([0, 0, 1], 3)


def _one_image_dataset(tmp_path, **kwargs):
    (tmp_path / "a").mkdir()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(tmp_path / "a" / "0.jpg")
    csv = tmp_path / "m.csv"
    pd.DataFrame({"path": ["a/0.jpg"], "class_id": ["a"]}).to_csv(csv, index=False)
    return ManifestDataset(csv, tmp_path, {"a": 0}, transform=build_eval_transform(32), **kwargs)


def test_manifest_dataset_reads_images(tmp_path):
    dataset = _one_image_dataset(tmp_path)
    image, label = dataset[0]
    assert len(dataset) == 1
    assert image.shape == (3, 32, 32)
    assert label == 0


def test_manifest_dataset_uses_masks_for_background(tmp_path):
    masks = tmp_path / "masks"
    (masks / "a").mkdir(parents=True)
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[8:22, 10:30] = 255
    Image.fromarray(mask).save(masks / "a" / "0.png")
    dataset = _one_image_dataset(tmp_path, background=RandomBackground(p=1.0), mask_root=masks)
    assert dataset.mask_coverage() == 1.0
    image, _ = dataset[0]
    assert image.shape == (3, 32, 32)
    no_masks = ManifestDataset(tmp_path / "m.csv", tmp_path, {"a": 0}, mask_root=tmp_path / "none")
    assert no_masks.mask_coverage() == 0.0


def test_transform_output_shapes():
    image = Image.new("RGB", (256, 256), (120, 120, 120))
    train = build_train_transform(224, erase_prob=0.5)(image)
    evaluated = build_eval_transform(224)(image)
    assert train.shape == evaluated.shape == (3, 224, 224)


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "resnet18"])
def test_forward_and_freezing(arch):
    model = build_model(arch, 19, pretrained=False)
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 19)
    set_backbone_trainable(model, arch, False)
    backbone, head = split_params(model, arch)
    assert all(not p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in head)
    set_train_mode(model, arch, backbone_frozen=True)
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms
    assert all(not m.training for m in norms)
    set_train_mode(model, arch, backbone_frozen=False)
    assert all(m.training for m in norms)


def test_list_dataset_and_mixed_sampling(tmp_path):
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

    from ml.mixing import sample_weights

    for name, colour in (("lab", (200, 0, 0)), ("field", (0, 0, 200))):
        Image.new("RGB", (40, 30), colour).save(tmp_path / f"{name}.jpg")
    transform = build_eval_transform(32)
    lab = ListDataset([tmp_path / "lab.jpg"] * 90, [0] * 90, transform)
    field = ListDataset([tmp_path / "field.jpg"] * 10, [1] * 10, transform)
    image, label = field[0]
    assert image.shape == (3, 32, 32) and label == 1
    weights = sample_weights(len(lab), len(field), 0.3)
    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights), num_samples=len(lab), replacement=True, generator=generator
    )
    loader = DataLoader(ConcatDataset([lab, field]), batch_size=30, sampler=sampler, drop_last=True)
    labels = torch.cat([y for _, y in loader])
    assert len(labels) == 90
    assert 0.15 < labels.float().mean().item() < 0.45  # about 30% field photos
