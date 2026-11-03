"""
src/data/augmentations.py
Albumentations augmentation pipelines for training and validation.
Forensic-safe: does NOT apply heavy geometric distortions that would destroy forgery residuals.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Training augmentations - forensic safe (no heavy blur, no extreme JPEG)."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.ImageCompression(quality_range=(70, 95), p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Validation/test augmentations - just resize and normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})
