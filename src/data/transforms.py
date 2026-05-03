from torchvision import transforms

# ImageNet stats (for pretrained models expecting 3-channel input)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Reasonable stats for FER-2013 grayscale images
_GRAY_MEAN = (0.5,)
_GRAY_STD = (0.5,)


def get_train_transforms(image_size: int = 48, channels: int = 1):
    """Augmentation pipeline for training.

    Args:
        image_size: Target spatial resolution (48 for native, 224 for pretrained).
        channels:   1 for grayscale-only models, 3 to replicate into RGB for
                    pretrained ImageNet models.
    """
    pipeline = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandAugment(num_ops=2, magnitude=9),
    ]

    if channels == 3:
        pipeline.append(transforms.Grayscale(num_output_channels=3))
        pipeline.append(transforms.ToTensor())
        pipeline.append(transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD))
    else:
        pipeline.append(transforms.ToTensor())
        pipeline.append(transforms.Normalize(_GRAY_MEAN, _GRAY_STD))

    return transforms.Compose(pipeline)


def get_eval_transforms(image_size: int = 48, channels: int = 1):
    """Deterministic pipeline for validation and test (no augmentation).

    Args:
        image_size: Target spatial resolution.
        channels:   1 for grayscale, 3 for ImageNet-pretrained models.
    """
    pipeline = [
        transforms.Resize((image_size, image_size)),
    ]

    if channels == 3:
        pipeline.append(transforms.Grayscale(num_output_channels=3))
        pipeline.append(transforms.ToTensor())
        pipeline.append(transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD))
    else:
        pipeline.append(transforms.ToTensor())
        pipeline.append(transforms.Normalize(_GRAY_MEAN, _GRAY_STD))

    return transforms.Compose(pipeline)
