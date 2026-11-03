"""
src/api/app.py
FastAPI backend for Document Forgery Detection.

Endpoints:
- GET /health: Health status and model info
- POST /detect: Upload an image or PDF, returns detection results, confidence, and bounding boxes
"""
from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from src.utils import project_root, load_config, resolve_device

app = FastAPI(
    title="Document Forgery Detection API",
    description="Deep Learning-Based Tampered Text Detection and Localization (Forensic Aid Only)",
    version="1.0.0",
)

ROOT = project_root()
detector = None


def get_detector():
    global detector
    if detector is None:
        cfg = load_config(ROOT / "configs/dual_branch.yaml")
        model_name = getattr(cfg.model, "name", "dual_branch_forgery_net")
        ckpt_path = ROOT / cfg.paths.checkpoint_dir / model_name / "best_model.pth"

        from src.models.dual_branch import build_dual_branch
        from src.inference.pipeline import ForgeryDetector

        model = build_dual_branch(cfg)
        device = resolve_device("auto")
        if ckpt_path.exists():
            import torch
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()

        detector = ForgeryDetector(model=model, device=device)
    return detector


@app.get("/")
def index():
    return {
        "service": "Document Forgery Detection API",
        "disclaimer": "Forensic assistance aid only - NOT a definitive legal verdict.",
        "endpoints": ["/health", "/detect", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(resolve_device("auto")),
        "model_loaded": detector is not None,
    }


@app.post("/detect")
async def detect_document(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Upload an image (PNG, JPG) or PDF to run tamper detection.
    """
    contents = await file.read()
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    if suffix not in (".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"):
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    temp_path = ROOT / "outputs" / "temp_uploads"
    temp_path.mkdir(parents=True, exist_ok=True)
    save_file = temp_path / filename
    with open(save_file, "wb") as f:
        f.write(contents)

    try:
        det = get_detector()
        det.threshold = threshold
        result = det.detect(save_file)

        # Convert heatmap to base64
        overlay_pil = Image.fromarray(result["heatmap_overlay"])
        buf = io.BytesIO()
        overlay_pil.save(buf, format="JPEG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "filename": filename,
            "is_forged": result["is_forged"],
            "confidence": round(result["confidence"], 4),
            "regions_count": len(result["forgery_regions"]),
            "forgery_regions": result["forgery_regions"],
            "heatmap_base64_jpeg": overlay_b64,
            "disclaimer": "Forensic assistance aid only - NOT a legal verdict.",
        }
    finally:
        if save_file.exists():
            try:
                save_file.unlink()
            except Exception:
                pass

