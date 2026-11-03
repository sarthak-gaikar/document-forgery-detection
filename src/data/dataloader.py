"""
src/data/dataloader.py
DataLoader factory for Document Forgery Detection (RealText-V2 & Synthetic).
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DocumentForgeryDataset, SyntheticForgeryDataset, RealTextDataset
from src.data.augmentations import get_train_transforms, get_val_transforms


def build_dataloaders(
    data_dir: str | Path,
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders for RealText-V2 or Synthetic datasets.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = DocumentForgeryDataset(
        data_dir=data_dir,
        split="train",
        transform=get_train_transforms(image_size),
    )
    val_ds = DocumentForgeryDataset(
        data_dir=data_dir,
        split="val",
        transform=get_val_transforms(image_size),
    )
    test_ds = DocumentForgeryDataset(
        data_dir=data_dir,
        split="test",
        transform=get_val_transforms(image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_ds) > batch_size else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
