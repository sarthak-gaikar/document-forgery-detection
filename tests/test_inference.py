"""
tests/test_inference.py
Unit tests for the inference pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from src.inference.pipeline import (
    preprocess_image,
    create_heatmap_overlay,
    extract_forgery_regions,
    tile_inference,
)
from src.models.dual_branch import DualBranchForgeryNet


def test_preprocess_image():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    tensor = preprocess_image(img, image_size=256)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32


def test_create_heatmap_overlay():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[20:40, 20:40] = 0.9  # simulated tamper region
    overlay = create_heatmap_overlay(img, prob_map, threshold=0.5)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_extract_forgery_regions():
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[10:30, 10:30] = 0.95
    regions = extract_forgery_regions(prob_map, threshold=0.5, min_area=20)
    assert len(regions) >= 1
    r = regions[0]
    assert "bbox" in r
    assert "mean_score" in r
    assert r["mean_score"] > 0.8


def test_tile_inference_small_image():
    model = DualBranchForgeryNet(pretrained_rgb=False)
    model.eval()
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    prob = tile_inference(model, img, tile_size=256, overlap=32, device=torch.device("cpu"))
    assert prob.shape == (128, 128)
    assert prob.min() >= 0.0 and prob.max() <= 1.0

