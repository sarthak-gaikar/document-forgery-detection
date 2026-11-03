"""
src/models/dual_branch.py
Model D: Dual-Branch Forgery Detection Network (Proposed Architecture)

Architecture:
  Branch 1 (RGB Spatial): ResNet-34 encoder capturing appearance features
  Branch 2 (Frequency/Residual): SRM-inspired high-pass filter capturing noise/residual features
  Fusion: Gated multi-scale attention fusion of both branches
  Decoder: UNet++ style decoder producing pixel-level segmentation mask
  Classifier Head: Global average pooling + FC for document-level binary classification

This dual-branch design is motivated by the observation that image forgeries leave
both spatial (appearance) artifacts AND frequency/noise residuals that are invisible
to the eye but detectable by learned high-pass filters.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# SRM (Spatial Rich Model) Filter Layer
# ---------------------------------------------------------------------------

def get_srm_kernels() -> torch.Tensor:
    """
    Returns 3 SRM high-pass filter kernels as a (3, 1, 5, 5) tensor.
    These filters are fixed (not learned) and highlight noise/residual patterns
    left by image editing operations.
    """
    # Filter 1: Second-order derivative (edge/noise detector)
    f1 = np.array([
        [ 0,  0, 0,  0, 0],
        [ 0, -1, 2, -1, 0],
        [ 0,  2,-4,  2, 0],
        [ 0, -1, 2, -1, 0],
        [ 0,  0, 0,  0, 0],
    ], dtype=np.float32) / 4.0

    # Filter 2: Laplacian
    f2 = np.array([
        [-1, 2, -2, 2, -1],
        [ 2,-6,  8,-6,  2],
        [-2, 8,-12, 8, -2],
        [ 2,-6,  8,-6,  2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0

    # Filter 3: First-order difference (gradient)
    f3 = np.array([
        [ 0,  0,  0, 0, 0],
        [ 0,  0,  0, 0, 0],
        [ 0,  1, -2, 1, 0],
        [ 0,  0,  0, 0, 0],
        [ 0,  0,  0, 0, 0],
    ], dtype=np.float32) / 2.0

    kernels = np.stack([f1, f2, f3], axis=0)  # (3, 5, 5)
    kernels = kernels[:, np.newaxis, :, :]     # (3, 1, 5, 5)
    return torch.from_numpy(kernels)


class SRMLayer(nn.Module):
    """
    Fixed SRM high-pass filter applied to each color channel independently.
    Input: (B, 3, H, W) — RGB image
    Output: (B, 9, H, W) — 3 SRM filters × 3 channels
    """
    def __init__(self):
        super().__init__()
        kernels = get_srm_kernels()  # (3, 1, 5, 5)
        # Apply same 3 filters to each of 3 color channels → grouped conv
        # repeat kernels for R, G, B: (9, 1, 5, 5)
        kernels_rgb = kernels.repeat(3, 1, 1, 1)  # (9, 1, 5, 5)
        self.register_buffer("weight", kernels_rgb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process each channel separately then cat
        R = F.conv2d(x[:, 0:1, :, :], self.weight[:3], padding=2, groups=1)
        G = F.conv2d(x[:, 1:2, :, :], self.weight[3:6], padding=2, groups=1)
        B = F.conv2d(x[:, 2:3, :, :], self.weight[6:9], padding=2, groups=1)
        out = torch.cat([R, G, B], dim=1)  # (B, 9, H, W)
        return out


# ---------------------------------------------------------------------------
# Frequency Branch (processes SRM output)
# ---------------------------------------------------------------------------

class FrequencyBranch(nn.Module):
    """
    Learns from SRM noise residuals. Produces multi-scale feature maps.
    Input: (B, 9, H, W) — SRM filtered channels
    Output: list of feature maps at 4 scales: 1/2, 1/4, 1/8, 1/16 of input
    """
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )  # 1/2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )  # 1/4
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )  # 1/8
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )  # 1/16

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.layer0(x)
        f1 = self.layer1(x)   # 1/2
        f2 = self.layer2(f1)  # 1/4
        f3 = self.layer3(f2)  # 1/8
        f4 = self.layer4(f3)  # 1/16
        return [f1, f2, f3, f4]


# ---------------------------------------------------------------------------
# RGB Branch (ResNet-34 encoder)
# ---------------------------------------------------------------------------

class RGBBranch(nn.Module):
    """
    RGB spatial branch using ResNet-34 as encoder.
    Returns feature maps at 4 scales matching the frequency branch outputs.
    Falls back to a simple CNN if torchvision is not available.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            if pretrained:
                try:
                    weights = ResNet34_Weights.IMAGENET1K_V1
                    backbone = resnet34(weights=weights)
                except Exception:
                    backbone = resnet34(weights=None)
            else:
                backbone = resnet34(weights=None)
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 1/2, 64ch
            self.pool = backbone.maxpool
            self.layer1 = backbone.layer1  # 1/4, 64ch
            self.layer2 = backbone.layer2  # 1/8, 128ch
            self.layer3 = backbone.layer3  # 1/16, 256ch
            self.layer4 = backbone.layer4  # 1/32, 512ch
            self.use_resnet = True
        except Exception:
            self.use_resnet = False
            self._build_fallback()

    def _build_fallback(self):
        self.enc0 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.use_resnet:
            x = self.layer0(x)  # 1/2, 64
            x = self.pool(x)    # 1/4, 64
            f1 = self.layer1(x) # 1/4, 64
            f2 = self.layer2(f1)# 1/8, 128
            f3 = self.layer3(f2)# 1/16, 256
            f4 = self.layer4(f3)# 1/32, 512
        else:
            x = self.enc0(x)
            f1 = self.enc1(x)
            f2 = self.enc2(f1)
            f3 = self.enc3(f2)
            f4 = self.enc4(f3)
        return [f1, f2, f3, f4]


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Gated attention fusion of RGB and frequency features at a single scale.
    Learns a soft gate α ∈ [0,1] per spatial location to select from RGB or freq.
    Output: fused features with same channels as max(rgb_ch, freq_ch) projected to out_ch.
    """
    def __init__(self, rgb_ch: int, freq_ch: int, out_ch: int):
        super().__init__()
        combined_ch = rgb_ch + freq_ch
        self.gate = nn.Sequential(
            nn.Conv2d(combined_ch, combined_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(combined_ch // 2, 2, 1),   # 2 weights: rgb, freq
            nn.Softmax(dim=1),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(combined_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        # Align spatial sizes
        if rgb_feat.shape[2:] != freq_feat.shape[2:]:
            freq_feat = F.interpolate(freq_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)
        cat = torch.cat([rgb_feat, freq_feat], dim=1)
        weights = self.gate(cat)  # (B, 2, H, W)
        w_rgb = weights[:, 0:1]
        w_freq = weights[:, 1:2]
        gated = torch.cat([w_rgb * rgb_feat, w_freq * freq_feat], dim=1)
        return self.proj(gated)


# ---------------------------------------------------------------------------
# UNet++ style Decoder
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full Dual-Branch Network
# ---------------------------------------------------------------------------

class DualBranchForgeryNet(nn.Module):
    """
    Model D: Dual-Branch Forgery Detection Network.

    Input: (B, 3, H, W) — RGB document image
    Output: dict with:
        "seg_logits":  (B, 1, H, W) — pixel-level forgery mask logits
        "cls_logits":  (B, 1)       — document-level forgery classification logits
    """
    def __init__(self, pretrained_rgb: bool = True, classes: int = 1):
        super().__init__()
        self.srm = SRMLayer()
        self.rgb_branch = RGBBranch(pretrained=pretrained_rgb)
        self.freq_branch = FrequencyBranch()

        # Gated fusion at each of 4 scales
        # RGB channels:  [64, 128, 256, 512] (or [64,64,128,256,512] with pool)
        # Freq channels: [64, 128, 256, 512]
        rgb_chs  = [64, 128, 256, 512]
        freq_chs = [64, 128, 256, 512]
        fused_chs = [64, 128, 256, 256]

        self.fusions = nn.ModuleList([
            GatedFusion(rgb_chs[i], freq_chs[i], fused_chs[i])
            for i in range(4)
        ])

        # Decoder (4 upsampling stages)
        self.dec4 = DecoderBlock(fused_chs[3], fused_chs[2], 128)
        self.dec3 = DecoderBlock(128, fused_chs[1], 64)
        self.dec2 = DecoderBlock(64, fused_chs[0], 32)
        self.dec1 = DecoderBlock(32, 0, 16)

        # Segmentation head
        self.seg_head = nn.Conv2d(16, classes, 1)

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_chs[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Residual features from SRM
        srm_out = self.srm(x)  # (B, 9, H, W)

        # Branch features
        rgb_feats = self.rgb_branch(x)    # list of 4 tensors
        freq_feats = self.freq_branch(srm_out)  # list of 4 tensors

        # Gated fusion at each scale
        fused = [self.fusions[i](rgb_feats[i], freq_feats[i]) for i in range(4)]
        # fused: [f1(1/4), f2(1/8), f3(1/16), f4(1/32)]

        # Decoder
        d4 = self.dec4(fused[3], fused[2])   # 1/16
        d3 = self.dec3(d4, fused[1])          # 1/8
        d2 = self.dec2(d3, fused[0])          # 1/4
        d1 = self.dec1(d2)                    # 1/2

        # Upsample to original size
        seg_logits = self.seg_head(d1)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Classification
        cls_logits = self.cls_head(self.cls_pool(fused[3]))  # (B, 1)

        return {"seg_logits": seg_logits, "cls_logits": cls_logits}


def build_dual_branch(cfg) -> DualBranchForgeryNet:
    model_cfg = cfg.model
    weights = getattr(model_cfg, "rgb_encoder_weights", None)
    pretrained = (weights is not None and str(weights).lower() not in ("none", "null", "false"))
    return DualBranchForgeryNet(
        pretrained_rgb=pretrained,
        classes=getattr(model_cfg, "classes", 1),
    )
