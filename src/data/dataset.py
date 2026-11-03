"""
src/data/dataset.py
PyTorch Dataset for Document Forgery Detection (supports RealText-V2 and Synthetic datasets).
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class DocumentForgeryDataset(Dataset):
    """
    Universal Dataset for Document Forgery Detection.
    Works with both RealText-V2 and Synthetic dataset formats.

    Manifest format (manifest.json):
    {
      "n_samples": ...,
      "samples": [
        {
          "sample_id": "realtext_0001",
          "image_path": "train/image/part000/doc_001.jpg",   # relative or absolute
          "mask_path": "train/mask/part000/doc_001.png",     # optional if authentic
          "is_forged": true,
          "doc_type": "invoice",
          "tamper_type": "text_replacement",
          "split": "train"
        },
        ...
      ]
    }

    Also supports legacy layout ({sample_id}.png / {sample_id}_mask.png).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_cls_label: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.return_cls_label = return_cls_label

        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found at {manifest_path}. "
                "Download or index RealText-V2 using: python scripts/download_realtext.py"
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.samples = [s for s in manifest.get("samples", []) if s.get("split") == split]
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split='{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path_str: Optional[str], default_filename: str) -> Optional[Path]:
        if path_str:
            p = Path(path_str)
            if p.is_absolute() and p.exists():
                return p
            candidate = self.data_dir / p
            if candidate.exists():
                return candidate
        default_p = self.data_dir / default_filename
        if default_p.exists():
            return default_p
        return None

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]
        sample_id = meta.get("sample_id", f"sample_{idx}")

        # Resolve image path
        img_p = self._resolve_path(meta.get("image_path"), f"{sample_id}.png")
        if img_p is None:
            # Try .jpg
            img_p = self._resolve_path(meta.get("image_path"), f"{sample_id}.jpg")
        if img_p is None:
            raise FileNotFoundError(f"Could not find image for sample {sample_id} in {self.data_dir}")

        # Load image via OpenCV
        image = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if image is None:
            with Image.open(str(img_p)) as img:
                image = np.array(img.convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        # Resolve mask path
        is_forged = bool(meta.get("is_forged", False))
        mask_p = self._resolve_path(meta.get("mask_path"), f"{sample_id}_mask.png")

        if mask_p is not None and mask_p.exists():
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                with Image.open(str(mask_p)) as m_img:
                    mask = np.array(m_img.convert("L"))
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            # Binarize: > 127 -> 1, else 0
            mask = (mask > 127).astype(np.uint8)
        else:
            # Authentic sample or missing mask -> all-zero mask
            mask = np.zeros((H, W), dtype=np.uint8)

        # Double check forged flag if mask has positive pixels
        if mask.sum() > 0:
            is_forged = True

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]   # (C, H, W) float tensor
            mask = transformed["mask"]     # (H, W) uint8 tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        # Ensure mask is float (for BCE / Dice loss)
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        result = {
            "image": image,                                   # (3, H, W)
            "mask": mask,                                     # (1, H, W)
            "sample_id": sample_id,
            "doc_type": meta.get("doc_type", "document"),
            "tamper_type": meta.get("tamper_type") or ("authentic" if not is_forged else "tampered"),
        }

        if self.return_cls_label:
            result["label"] = torch.tensor(1.0 if is_forged else 0.0, dtype=torch.float32)

        return result

    def get_class_weights(self) -> Tuple[float, float]:
        """Returns (neg_weight, pos_weight) for class balancing."""
        n_forged = sum(1 for s in self.samples if s.get("is_forged", False))
        n_authentic = len(self.samples) - n_forged
        if n_forged == 0:
            return 1.0, 1.0
        pos_weight = n_authentic / max(1, n_forged)
        return 1.0, float(pos_weight)


# Backward compatibility aliases
SyntheticForgeryDataset = DocumentForgeryDataset
RealTextDataset = DocumentForgeryDataset
