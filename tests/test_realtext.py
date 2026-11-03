"""
tests/test_realtext.py
Unit and integration tests for RealText-V2 dataset pipeline and 3 training algorithms.
"""
from __future__ import annotations
import pytest
import torch
import shutil
from pathlib import Path

from src.data.dataset import DocumentForgeryDataset, RealTextDataset
from src.data.dataloader import build_dataloaders
from src.models.dual_branch import DualBranchForgeryNet
from src.models.unet import UNetResNet34
from src.models.deeplab import DeepLabV3PlusEfficientNet
from src.losses.combined import MultiTaskLoss, BCEDiceLoss, FocalDiceLoss
from scripts.download_realtext import create_mock_realtext_dataset


@pytest.fixture(scope="module")
def mock_realtext_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("realtext_v2_test")
    create_mock_realtext_dataset(tmp_dir, n_samples=20)
    return tmp_dir


def test_realtext_manifest_creation(mock_realtext_dir):
    manifest_p = mock_realtext_dir / "manifest.json"
    assert manifest_p.exists()


def test_realtext_dataset_loading(mock_realtext_dir):
    ds = DocumentForgeryDataset(data_dir=mock_realtext_dir, split="train")
    assert len(ds) > 0
    sample = ds[0]
    assert "image" in sample
    assert "mask" in sample
    assert "label" in sample
    assert sample["image"].shape[0] == 3
    assert sample["mask"].shape[0] == 1


def test_realtext_dataloaders(mock_realtext_dir):
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=mock_realtext_dir,
        image_size=256,
        batch_size=2,
    )
    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 3, 256, 256)
    assert batch["mask"].shape == (2, 1, 256, 256)
    assert batch["label"].shape == (2,)


def test_algorithm1_dual_branch_forward_backward(mock_realtext_dir):
    """Test Algorithm 1: Dual-Branch Network forward + backward pass."""
    model = DualBranchForgeryNet(pretrained_rgb=False)
    loss_fn = MultiTaskLoss(alpha_cls=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 256, 256)
    mask_target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    cls_target = torch.tensor([1.0, 0.0])

    out = model(x)
    loss, details = loss_fn(out["seg_logits"], out["cls_logits"], mask_target, cls_target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert "loss_seg" in details
    assert "loss_cls" in details


def test_algorithm2_unet_forward_backward(mock_realtext_dir):
    """Test Algorithm 2: U-Net with ResNet-34 forward + backward pass."""
    model = UNetResNet34(encoder_weights=None, in_channels=3, classes=1)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 256, 256)
    mask_target = torch.randint(0, 2, (2, 1, 256, 256)).float()

    out = model(x)
    loss = loss_fn(out["seg_logits"], mask_target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)


def test_algorithm3_deeplab_forward_backward(mock_realtext_dir):
    """Test Algorithm 3: DeepLabV3+ with EfficientNet-B3 forward + backward pass."""
    model = DeepLabV3PlusEfficientNet(encoder_weights=None, in_channels=3, classes=1)
    loss_fn = FocalDiceLoss(gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 256, 256)
    mask_target = torch.randint(0, 2, (2, 1, 256, 256)).float()

    out = model(x)
    loss = loss_fn(out["seg_logits"], mask_target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
