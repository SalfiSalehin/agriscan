"""Image transforms shared by training and evaluation.

`build_eval_transform` is the reference preprocessing. The web backend (Phase 4) must
reproduce it exactly: resize shorter side to 256, centre-crop 224, ImageNet normalisation.
The background replacement happens earlier, inside the dataset (it needs the leaf mask).
"""

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_RESIZE_RATIO = 256 / 224


def build_train_transform(image_size: int = 224, erase_prob: float = 0.25):
    fill = tuple(int(m * 255) for m in IMAGENET_MEAN)
    steps = [
        T.RandomRotation(30, interpolation=T.InterpolationMode.BILINEAR, fill=fill),
        T.RandomResizedCrop(image_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # hue stays small on purpose: yellowing/browning is part of the disease signal
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if erase_prob > 0:
        steps.append(T.RandomErasing(p=erase_prob, scale=(0.02, 0.15), value="random"))
    return T.Compose(steps)


def build_eval_transform(image_size: int = 224):
    return T.Compose(
        [
            T.Resize(round(image_size * EVAL_RESIZE_RATIO)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
