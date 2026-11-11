"""Model builders and freezing helpers for the two architectures we compare."""

from torch import nn
from torchvision import models

ARCHS = ("mobilenet_v3_small", "resnet18")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r}, choose from {ARCHS}")
    return model


def _head(model: nn.Module, arch: str) -> nn.Module:
    return model.classifier if arch == "mobilenet_v3_small" else model.fc


def split_params(model: nn.Module, arch: str):
    """(backbone parameters, head parameters). The head is the classifier / fc layer."""
    head = list(_head(model, arch).parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    backbone, _ = split_params(model, arch)
    for p in backbone:
        p.requires_grad = trainable


def set_train_mode(model: nn.Module, arch: str, backbone_frozen: bool) -> None:
    """model.train(), but keep BatchNorm statistics fixed while the backbone is frozen."""
    model.train()
    if backbone_frozen:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
