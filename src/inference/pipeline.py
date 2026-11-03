"""
src/inference/pipeline.py
Inference pipeline for document forgery detection.

Supports:
- Aspect-ratio preserving letterbox preprocessing (no character squishing)
- High-resolution Hann-windowed tile inference (for scanned / HD pages)
- Table-line and sub-pixel noise suppression in post-processing
- Single image inference and PDF multi-page rendering
- Heatmap overlay generation and bounding box extraction
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from src.utils import get_logger, resolve_device

logger = get_logger("inference")


def load_model(checkpoint_path: str | Path, model_cfg, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load a model from checkpoint."""
    if device is None:
        device = resolve_device("auto")
    
    if hasattr(model_cfg, "model"):
        if hasattr(model_cfg.model, "rgb_encoder_weights"):
            model_cfg.model.rgb_encoder_weights = "none"
        if hasattr(model_cfg.model, "encoder_weights"):
            model_cfg.model.encoder_weights = "none"

    from scripts.train import build_model
    
    model = build_model(model_cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def render_pdf_page(pdf_path: str | Path, page_idx: int = 0, dpi: int = 150) -> np.ndarray:
    """Render a PDF page to a numpy RGB array using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()
        return img_array
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF inference. Install with: pip install PyMuPDF")


def letterbox_image(image: np.ndarray, target_size: int = 512, pad_val: int = 255) -> tuple[np.ndarray, dict]:
    """
    Resize image to fit within (target_size, target_size) while preserving aspect ratio.
    Pads remaining border with white (255) for document background consistency.
    Returns: (padded_image, pad_info_dict).
    """
    H, W = image.shape[:2]
    scale = target_size / max(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_size, target_size, 3), pad_val, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized

    pad_info = {
        "top": top,
        "left": left,
        "new_h": new_h,
        "new_w": new_w,
        "orig_h": H,
        "orig_w": W,
        "scale": scale,
    }
    return padded, pad_info


def unpad_probability_map(prob_map: np.ndarray, pad_info: dict) -> np.ndarray:
    """
    Extract active document region from a padded probability map and resize back
    to the original image shape (orig_h, orig_w).
    """
    top = pad_info["top"]
    left = pad_info["left"]
    h = pad_info["new_h"]
    w = pad_info["new_w"]
    valid_slice = prob_map[top : top + h, left : left + w]
    return cv2.resize(valid_slice, (pad_info["orig_w"], pad_info["orig_h"]), interpolation=cv2.INTER_LINEAR)


def preprocess_image(
    image: np.ndarray,
    image_size: int = 512,
    preserve_aspect: bool = False,
    return_pad_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Preprocess a numpy RGB image to a normalized tensor (1, 3, H, W).
    If preserve_aspect is True, applies white-background letterboxing.
    If return_pad_info is True, returns (tensor, pad_info).
    """
    if preserve_aspect:
        img_np, pad_info = letterbox_image(image, target_size=image_size)
    else:
        img_np = cv2.resize(image, (image_size, image_size))
        pad_info = {
            "top": 0,
            "left": 0,
            "new_h": image_size,
            "new_w": image_size,
            "orig_h": image.shape[0],
            "orig_w": image.shape[1],
            "scale": 1.0,
        }

    img = img_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    if return_pad_info:
        return tensor, pad_info
    return tensor


def tile_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Tile-based inference for high-resolution images.
    Uses Hann window to smoothly blend overlapping tile predictions.
    Returns: probability map of shape (H, W) with values in [0, 1].
    """
    if device is None:
        device = resolve_device("auto")

    H, W = image.shape[:2]

    # If image is smaller than or equal to tile_size, use letterbox padding
    if H <= tile_size and W <= tile_size:
        tensor, pad_info = preprocess_image(image, tile_size, preserve_aspect=True, return_pad_info=True)
        with torch.no_grad():
            out = model(tensor.to(device))
            logit = out.get("seg_logits", out.get("logits", None))
            if logit is None and "cls_logits" in out:
                score = torch.sigmoid(out["cls_logits"]).item()
                return np.full((H, W), score, dtype=np.float32)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            return unpad_probability_map(prob, pad_info)

    stride = max(1, tile_size - overlap)
    output = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    gauss_1d = np.hanning(tile_size).astype(np.float32)
    gauss_1d = np.clip(gauss_1d, 0.1, 1.0)
    gauss_2d = np.outer(gauss_1d, gauss_1d)

    y_starts = list(range(0, max(1, H - tile_size + 1), stride))
    x_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if y_starts and H > tile_size and (H - tile_size) not in y_starts:
        y_starts.append(H - tile_size)
    if x_starts and W > tile_size and (W - tile_size) not in x_starts:
        x_starts.append(W - tile_size)
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]

    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                y0_act = max(0, y1 - tile_size)
                x0_act = max(0, x1 - tile_size)

                tile = image[y0_act:y1, x0_act:x1]
                tensor = preprocess_image(tile, tile_size, preserve_aspect=False).to(device)

                out = model(tensor)
                logit = out.get("seg_logits", out.get("logits", None))
                if logit is not None:
                    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                    output[y0_act:y1, x0_act:x1] += prob * gauss_2d
                    weights[y0_act:y1, x0_act:x1] += gauss_2d

    weights = np.maximum(weights, 1e-6)
    return output / weights


def create_heatmap_overlay(
    image: np.ndarray,
    prob_map: np.ndarray,
    alpha: float = 0.45,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Create a colored heatmap overlay on the document image.
    Highlights tampered regions in red/yellow with clean contour outlines.
    """
    if prob_map.shape[:2] != image.shape[:2]:
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)

    # Clean binary mask with small morphological opening
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    return overlay


def extract_forgery_regions(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 60,
    filter_lines: bool = True,
) -> list[dict]:
    """
    Extract bounding boxes of detected tampered regions.
    Filters out spurious line artifacts (table borders, rules) and tiny noise dots.
    """
    binary = (prob_map > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        # Line artifact filter: exclude very thin, long horizontal or vertical lines
        if filter_lines:
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > 10.0 and min(w, h) <= 6:
                continue

        region_probs = prob_map[y : y + h, x : x + w]
        regions.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "width": int(w),
            "height": int(h),
            "area": int(area),
            "mean_score": float(region_probs.mean()),
            "max_score": float(region_probs.max()),
        })

    return sorted(regions, key=lambda r: r["max_score"] * r["area"], reverse=True)


class ForgeryDetector:
    """
    High-level inference detector with aspect-ratio preservation and auto-tiling.
    
    Usage:
        detector = ForgeryDetector(model)
        result = detector.detect("invoice.png")
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        tile_size: int = 512,
        overlap: int = 64,
        threshold: float = 0.5,
        use_tiling: Union[bool, str] = "auto",
        min_area: int = 60,
        auto_tile_threshold: int = 1200,
    ):
        self.model = model
        self.device = device or resolve_device("auto")
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold
        self.use_tiling = use_tiling
        self.min_area = min_area
        self.auto_tile_threshold = auto_tile_threshold
        self.model.to(self.device)
        self.model.eval()

    def detect(self, input_path: str | Path, output_dir: Optional[str | Path] = None) -> dict:
        input_path = Path(input_path)

        if input_path.suffix.lower() == ".pdf":
            image = render_pdf_page(input_path)
        else:
            img_bgr = cv2.imread(str(input_path))
            if img_bgr is not None:
                image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                with Image.open(input_path) as img:
                    image = np.array(img.convert("RGB"))

        H, W = image.shape[:2]
        max_dim = max(H, W)

        # Determine whether to use high-resolution tiling or letterboxed single-pass
        do_tile = (
            (self.use_tiling is True)
            or (self.use_tiling == "auto" and max_dim >= self.auto_tile_threshold)
        )

        cls_score = None
        if do_tile:
            inspection_mode = "tiled_high_res"
            prob_map = tile_inference(
                self.model, image,
                tile_size=self.tile_size,
                overlap=self.overlap,
                device=self.device,
            )
        else:
            inspection_mode = "letterbox_aspect_preserved"
            tensor, pad_info = preprocess_image(
                image,
                image_size=self.tile_size,
                preserve_aspect=True,
                return_pad_info=True,
            )
            with torch.no_grad():
                out = self.model(tensor.to(self.device))
                seg_logits = out.get("seg_logits", out.get("logits", None))
                cls_logits = out.get("cls_logits", None)

                if seg_logits is not None:
                    prob_512 = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
                    prob_map = unpad_probability_map(prob_512, pad_info)
                else:
                    prob_map = np.zeros((H, W), dtype=np.float32)

                if cls_logits is not None:
                    cls_score = float(torch.sigmoid(cls_logits).item())

        regions = extract_forgery_regions(
            prob_map,
            threshold=self.threshold,
            min_area=self.min_area,
            filter_lines=True,
        )

        has_regions = len(regions) > 0
        max_region_score = max((r["max_score"] for r in regions), default=0.0)
        max_seg = float(prob_map.max())

        # Robust decision logic: require valid filtered region or classification score
        if cls_score is not None:
            is_forged = bool((cls_score >= self.threshold) or (has_regions and max_region_score >= 0.70))
            confidence = float(max(cls_score, max_region_score if has_regions else 0.0))
        else:
            is_forged = bool(has_regions)
            if is_forged:
                confidence = float(max_region_score)
            else:
                # Authentic confidence is high when mean background prob is near zero
                mean_p = float(prob_map.mean())
                confidence = float(np.clip(1.0 - (mean_p * 10.0), 0.60, 0.99))

        overlay = create_heatmap_overlay(image, prob_map, threshold=self.threshold)

        result = {
            "input_path": str(input_path),
            "is_forged": is_forged,
            "confidence": confidence,
            "cls_score": cls_score,
            "max_seg_prob": max_seg,
            "inspection_mode": inspection_mode,
            "forgery_regions": regions,
            "prob_map": prob_map,
            "original_image": image,
            "heatmap_overlay": overlay,
        }

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem

            Image.fromarray(overlay).save(output_dir / f"{stem}_heatmap.png")
            binary_mask = (prob_map > self.threshold).astype(np.uint8) * 255
            Image.fromarray(binary_mask).save(output_dir / f"{stem}_mask.png")

            report = {
                "input_path": str(input_path),
                "is_forged": is_forged,
                "confidence": confidence,
                "inspection_mode": inspection_mode,
                "regions_count": len(regions),
                "forgery_regions": regions,
            }
            with open(output_dir / f"{stem}_report.json", "w") as f:
                json.dump(report, f, indent=2)

        return result
