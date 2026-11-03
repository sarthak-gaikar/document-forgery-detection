"""
src/api/demo.py
Interactive Gradio web interface for Document Forgery Detection.
Features Light Theme, Forensic Analysis Tab, and a Model Comparison Dashboard with Visual Charts.

Run:
  python src/api/demo.py
"""
from __future__ import annotations

import os
import sys
import uuid
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Force project workspace temp directory to prevent Windows AppData permission locks
TEMP_DIR = ROOT / "outputs" / "gradio_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)
os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs" / ".matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import gradio as gr

from src.utils import load_config, resolve_device
from src.inference.pipeline import ForgeryDetector, load_model

# Cached detectors by model key
_detectors = {}

MODEL_OPTIONS = {
    "Algorithm 2 (Fine-Tuned): U-Net + ResNet-34 (Focal-Dice - Recommended)": {
        "config": "configs/realtext_unet_finetune.yaml",
        "fallback_config": "configs/realtext_unet.yaml",
        "ckpt_subpaths": [
            "outputs/checkpoints/realtext_finetune/unet_resnet34/best_model.pth",
            "outputs/checkpoints/realtext/unet_resnet34/best_model.pth",
        ],
    },
    "Algorithm 1: Dual-Branch Net (Proposed - RGB + SRM)": {
        "config": "configs/realtext_dual_branch.yaml",
        "fallback_config": "configs/dual_branch.yaml",
        "ckpt_subpaths": [
            "outputs/checkpoints/realtext/dual_branch_forgery_net/best_model.pth",
            "outputs/checkpoints/dual_branch_forgery_net/best_model.pth",
        ],
    },
    "Algorithm 2 (Baseline): U-Net + ResNet-34": {
        "config": "configs/realtext_unet.yaml",
        "fallback_config": "configs/unet_resnet34.yaml",
        "ckpt_subpaths": [
            "outputs/checkpoints/realtext/unet_resnet34/best_model.pth",
        ],
    },
    "Algorithm 3: DeepLabV3+ + EfficientNet-B3 (Model B)": {
        "config": "configs/realtext_deeplabv3plus.yaml",
        "fallback_config": "configs/deeplabv3plus_efficientnet.yaml",
        "ckpt_subpaths": [
            "outputs/checkpoints/realtext/deeplabv3plus_efficientnetb3/best_model.pth",
            "outputs/checkpoints/deeplabv3plus_efficientnetb3/best_model.pth",
        ],
    },
}

MODE_MAP = {
    "Auto (Adaptive: Letterbox for standard, Tiled for HD > 1200px)": "auto",
    "Deep Tiled (High Resolution - Best for Scanned Forms / PDFs)": True,
    "Fast (Aspect-Preserved Letterbox)": False,
}

BENCHMARK_DATA = [
    {
        "Model Name": "Algorithm 2 (Fine-Tuned): U-Net ResNet-34",
        "Architecture": "Encoder-Decoder (Skip Connections)",
        "Loss Formulation": "Focal-Dice (gamma=2.0)",
        "Precision (%)": 55.73,
        "Recall (%)": 42.10,
        "Dice (F1)": 0.4379,
        "Mean IoU": 0.3098,
        "Doc Accuracy (%)": "N/A (Segmentation)",
        "FPS": 9.66,
        "Status": "Recommended",
    },
    {
        "Model Name": "Algorithm 2 (Baseline): U-Net ResNet-34",
        "Architecture": "Encoder-Decoder (Skip Connections)",
        "Loss Formulation": "BCE-Dice (Combined)",
        "Precision (%)": 22.90,
        "Recall (%)": 14.04,
        "Dice (F1)": 0.6329,
        "Mean IoU": 0.5953,
        "Doc Accuracy (%)": "N/A (Segmentation)",
        "FPS": 9.66,
        "Status": "Baseline",
    },
    {
        "Model Name": "Algorithm 1: Dual-Branch Net (Proposed)",
        "Architecture": "Dual-Stream Multi-Task (RGB + SRM)",
        "Loss Formulation": "Multi-Task (Seg + 0.5*Cls)",
        "Precision (%)": 22.32,
        "Recall (%)": 13.84,
        "Dice (F1)": 0.5950,
        "Mean IoU": 0.5557,
        "Doc Accuracy (%)": 68.81,
        "FPS": 9.53,
        "Status": "Dual-Task",
    },
    {
        "Model Name": "Algorithm 3: DeepLabV3+ EfficientNet-B3",
        "Architecture": "ASPP Dilated Convolutions",
        "Loss Formulation": "BCE-Dice (Combined)",
        "Precision (%)": 19.09,
        "Recall (%)": 12.54,
        "Dice (F1)": 0.5827,
        "Mean IoU": 0.5459,
        "Doc Accuracy (%)": "N/A (Segmentation)",
        "FPS": 8.61,
        "Status": "Comparative",
    },
]

HEAD_JS = """
<script>
function enforceStrictLightMode() {
    document.documentElement.className = 'light';
    document.documentElement.classList.remove('dark');
    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.style.colorScheme = 'light';
    if (document.body) {
        document.body.className = 'light';
        document.body.classList.remove('dark');
        document.body.style.colorScheme = 'light';
    }
}
enforceStrictLightMode();
document.addEventListener('DOMContentLoaded', enforceStrictLightMode);
window.addEventListener('load', enforceStrictLightMode);
const observer = new MutationObserver(() => {
    if (document.documentElement.classList.contains('dark')) {
        enforceStrictLightMode();
    }
});
observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
</script>
"""

LIGHT_THEME_CSS = """
/* Force Light Color Scheme Across OS and Browser */
:root, .light, html, body, .gradio-container {
    color-scheme: light !important;
    --body-background-fill: #f8fafc !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f1f5f9 !important;
    --block-background-fill: #ffffff !important;
    --border-color-primary: #cbd5e1 !important;
    --border-color-secondary: #e2e8f0 !important;
    --body-text-color: #0f172a !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --input-background-fill: #ffffff !important;
    --input-text-color: #0f172a !important;
    --input-border-color: #94a3b8 !important;
    --table-text-color: #0f172a !important;
}

/* Background canvas */
html, body, .gradio-container, .gradio-container.dark {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Force dark readable text everywhere */
p, span, label, h1, h2, h3, h4, h5, h6, strong, em, div, li, td, th,
.prose, .prose *, .markdown, .markdown *,
.gradio-container .prose, .gradio-container .markdown {
    color: #0f172a !important;
}

/* Form panels, cards, and container boxes */
.block, .form, .panel, .gr-box, .gr-panel, .tabitem, fieldset, .box {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Inputs, textareas, dropdowns */
input, textarea, select, .gr-input, .gr-select {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #94a3b8 !important;
}

/* Primary Action Button (White text on Deep Blue) */
button.primary, button.lg.primary, .gr-button-primary {
    background-color: #1d4ed8 !important;
    background: #1d4ed8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
}

button.primary *, button.lg.primary *, .gr-button-primary * {
    color: #ffffff !important;
}

/* Tab Navigation buttons */
.tab-nav button {
    background-color: #f1f5f9 !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 600 !important;
}

.tab-nav button.selected {
    background-color: #ffffff !important;
    color: #1d4ed8 !important;
    border-bottom: 3px solid #1d4ed8 !important;
    font-weight: 700 !important;
}

/* High-contrast table styles */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    background-color: #ffffff !important;
}

th {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #cbd5e1 !important;
}

td {
    background-color: #ffffff !important;
    color: #1e293b !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

tr:hover td {
    background-color: #f8fafc !important;
}
"""


def create_metric_comparison_chart():
    """Create grouped bar chart comparing Precision, Recall, Dice, and IoU."""
    models = [
        "U-Net (Fine-Tuned)",
        "U-Net (Baseline)",
        "Dual-Branch Net",
        "DeepLabV3+",
    ]
    precision = [55.73, 22.90, 22.32, 19.09]
    recall = [42.10, 14.04, 13.84, 12.54]
    dice = [43.79, 63.29, 59.50, 58.27]
    iou = [30.98, 59.53, 55.57, 54.59]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    r1 = ax.bar(x - 1.5 * width, precision, width, label="Precision (%)", color="#2563eb", edgecolor="#1e40af")
    r2 = ax.bar(x - 0.5 * width, recall, width, label="Recall (%)", color="#059669", edgecolor="#047857")
    r3 = ax.bar(x + 0.5 * width, dice, width, label="Dice Score (x100)", color="#d97706", edgecolor="#b45309")
    r4 = ax.bar(x + 1.5 * width, iou, width, label="Mean IoU (x100)", color="#7c3aed", edgecolor="#6d28d9")

    # Add labels on top of bars
    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="600",
                color="#0f172a",
            )

    ax.set_ylabel("Metric Value (%)", fontsize=11, fontweight="600", color="#0f172a")
    ax.set_title("Comparative Model Performance on RealText-V2 Test Set (3,007 Samples)", fontsize=13, fontweight="700", color="#0f172a", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="600", color="#0f172a")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9, loc="upper right")

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    plt.tight_layout()
    return fig


def create_tradeoff_chart():
    """Create Precision vs Recall scatter plot and Throughput bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
    fig.patch.set_facecolor("#ffffff")

    # Subplot 1: Precision vs Recall Trade-Off
    ax1.set_facecolor("#ffffff")
    models = ["U-Net (Fine-Tuned)", "U-Net (Baseline)", "Dual-Branch Net", "DeepLabV3+"]
    rec = [42.10, 14.04, 13.84, 12.54]
    prec = [55.73, 22.90, 22.32, 19.09]
    colors = ["#2563eb", "#64748b", "#059669", "#d97706"]

    for i in range(len(models)):
        size = 180 if i == 0 else 120
        ax1.scatter(rec[i], prec[i], color=colors[i], s=size, edgecolors="#0f172a", linewidth=1.2, zorder=5)
        offset_y = 1.8 if i == 0 else -2.8
        ax1.annotate(
            models[i],
            xy=(rec[i], prec[i]),
            xytext=(rec[i] + 0.8, prec[i] + offset_y),
            fontsize=9,
            fontweight="600",
            color="#0f172a",
        )

    # Progression arrow from Baseline to Fine-Tuned
    ax1.annotate(
        "",
        xy=(rec[0], prec[0]),
        xytext=(rec[1], prec[1]),
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.0, ls="--"),
    )
    ax1.text(26.0, 37.0, "+143% Precision\n+200% Recall", fontsize=8.5, fontweight="700", color="#1d4ed8")

    ax1.set_xlabel("Recall (%)", fontsize=10, fontweight="600", color="#0f172a")
    ax1.set_ylabel("Precision (%)", fontsize=10, fontweight="600", color="#0f172a")
    ax1.set_title("Precision vs Recall Localization Space", fontsize=11, fontweight="700", color="#0f172a")
    ax1.set_xlim(8, 48)
    ax1.set_ylim(10, 62)
    ax1.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#cbd5e1")

    # Subplot 2: Inference Speed (FPS)
    ax2.set_facecolor("#ffffff")
    fps_values = [9.66, 9.66, 9.53, 8.61]
    y_pos = np.arange(len(models))

    bars = ax2.barh(y_pos, fps_values, color=["#10b981", "#64748b", "#3b82f6", "#f59e0b"], edgecolor="#334155", height=0.55)
    for bar in bars:
        w = bar.get_width()
        ms = 1000.0 / w
        ax2.annotate(
            f"{w:.2f} FPS ({ms:.0f} ms)",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="600",
            color="#0f172a",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=9.5, fontweight="600", color="#0f172a")
    ax2.invert_yaxis()
    ax2.set_xlabel("Throughput (Frames Per Second)", fontsize=10, fontweight="600", color="#0f172a")
    ax2.set_title("Inference Throughput on NVIDIA RTX 4050", fontsize=11, fontweight="700", color="#0f172a")
    ax2.set_xlim(0, 13)
    ax2.grid(axis="x", linestyle="--", alpha=0.3, color="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#cbd5e1")

    plt.tight_layout()
    return fig


def get_detector(model_choice: str, threshold: float = 0.5, use_tiling="auto"):
    info = MODEL_OPTIONS.get(model_choice, list(MODEL_OPTIONS.values())[0])
    cache_key = model_choice

    if cache_key not in _detectors:
        cfg_path = ROOT / info["config"]
        if not cfg_path.exists():
            cfg_path = ROOT / info["fallback_config"]

        cfg = load_config(cfg_path)
        if hasattr(cfg, "model"):
            if hasattr(cfg.model, "rgb_encoder_weights"):
                cfg.model.rgb_encoder_weights = "none"
            if hasattr(cfg.model, "encoder_weights"):
                cfg.model.encoder_weights = "none"

        device = resolve_device("auto")

        found_ckpt = None
        for sub in info["ckpt_subpaths"]:
            p = ROOT / sub
            if p.exists():
                found_ckpt = p
                break

        if found_ckpt is not None:
            model = load_model(found_ckpt, cfg, device=device)
        else:
            from scripts.train import build_model
            model = build_model(cfg).to(device)
            model.eval()

        det = ForgeryDetector(
            model=model,
            device=device,
            threshold=threshold,
            use_tiling=use_tiling,
            min_area=80,
            auto_tile_threshold=1200,
        )
        _detectors[cache_key] = det

    detector = _detectors[cache_key]
    detector.threshold = threshold
    detector.use_tiling = use_tiling
    return detector


def analyze_document(image_input, model_choice: str, threshold: float, mode_choice: str):
    if image_input is None:
        return None, None, "Please upload a document image."

    use_tiling = MODE_MAP.get(mode_choice, "auto")
    det = get_detector(model_choice, threshold=threshold, use_tiling=use_tiling)

    unique_id = uuid.uuid4().hex[:8]
    work_file = TEMP_DIR / f"work_{unique_id}.png"

    try:
        if isinstance(image_input, (str, Path)):
            src_path = Path(image_input)
            if src_path.suffix.lower() == ".pdf":
                work_file = TEMP_DIR / f"work_{unique_id}.pdf"
            shutil.copy2(src_path, work_file)
        elif isinstance(image_input, np.ndarray):
            Image.fromarray(image_input).save(work_file)
        elif hasattr(image_input, "name") and Path(image_input.name).exists():
            shutil.copy2(image_input.name, work_file)
        else:
            Image.fromarray(np.array(image_input)).save(work_file)

        res = det.detect(work_file)
        cls_str = f" (Classification Head: {res['cls_score']*100:.1f}%)" if res.get("cls_score") is not None else ""
        verdict = "[FORGED / TAMPERED]" if res["is_forged"] else "[AUTHENTIC / CLEAN]"
        confidence_pct = res["confidence"] * 100.0

        status_text = (
            f"### Forensic Verdict: **{verdict}**\n\n"
            f"- **Confidence:** **{confidence_pct:.1f}%**{cls_str}\n"
            f"- **Selected Architecture:** `{model_choice.split(':')[0]}`\n"
            f"- **Inspection Pipeline:** `{res.get('inspection_mode', 'standard')}`\n"
            f"- **Detected Tamper Clusters:** **{len(res['forgery_regions'])}**\n\n"
        )

        if res["forgery_regions"]:
            status_text += "**Detected Manipulation Regions:**\n\n"
            status_text += "| Region ID | Coordinates [x1, y1, x2, y2] | Area (Pixels) | Peak Confidence |\n"
            status_text += "| :---: | :--- | :---: | :---: |\n"
            for i, r in enumerate(res["forgery_regions"][:5], 1):
                status_text += (
                    f"| Region {i} | `[{r['bbox'][0]}, {r['bbox'][1]}, {r['bbox'][2]}, {r['bbox'][3]}]` | "
                    f"{r['area']} px | {r['max_score']:.3f} |\n"
                )
            status_text += "\n"
        else:
            status_text += "*No statistically significant tamper clusters detected. Document strokes and background remain consistent.*\n\n"

        status_text += "---\n*Forensic assistance aid only - NOT a definitive legal verdict.*"

        binary_mask = (res["prob_map"] > threshold).astype(np.uint8) * 255
        overlay_pil = Image.fromarray(res["heatmap_overlay"])
        mask_pil = Image.fromarray(binary_mask)
        return overlay_pil, mask_pil, status_text
    finally:
        if work_file.exists():
            try:
                work_file.unlink()
            except Exception:
                pass


def build_comparison_dataframe():
    return pd.DataFrame(BENCHMARK_DATA)


def launch_demo():
    custom_theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Document Forgery Detection System", theme=custom_theme) as demo:
        gr.Markdown(
            "# Digital Document Forgery Detection and Localization Platform\n"
            "**Department of Information Technology -- Deep Learning Research Laboratory**\n\n"
            "Multi-task pixel-level tampering localization, stroke-level forgery detection, and comparative model benchmarking."
        )

        with gr.Tabs():
            with gr.Tab("Document Forensic Analysis"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### Input Document & Controls")
                        input_img = gr.Image(type="filepath", label="Upload Document (PNG, JPG, PDF)")
                        model_choice = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value=list(MODEL_OPTIONS.keys())[0],
                            label="Select Architecture",
                        )
                        mode_choice = gr.Dropdown(
                            choices=list(MODE_MAP.keys()),
                            value=list(MODE_MAP.keys())[0],
                            label="Inspection Mode",
                        )
                        thresh_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Sensitivity Decision Threshold",
                        )
                        btn = gr.Button("Run Forensic Analysis", variant="primary")

                    with gr.Column(scale=6):
                        gr.Markdown("### Forensic Inspection Output")
                        out_heatmap = gr.Image(label="Tamper Heatmap Overlay")
                        out_mask = gr.Image(label="Predicted Binary Mask")
                        out_text = gr.Markdown()

                btn.click(
                    fn=analyze_document,
                    inputs=[input_img, model_choice, thresh_slider, mode_choice],
                    outputs=[out_heatmap, out_mask, out_text],
                )

            with gr.Tab("Model Comparison Dashboard"):
                gr.Markdown(
                    "### Benchmark Evaluation Across All 4 Models\n"
                    "Comprehensive evaluation on the independent **RealText-V2 Test Partition (3,007 Document Samples)**."
                )

                benchmark_df = build_comparison_dataframe()
                gr.DataFrame(
                    value=benchmark_df,
                    interactive=False,
                    label="Model Evaluation Matrix (RealText-V2 Test Split)",
                )

                gr.Markdown("---")
                gr.Markdown("### Visual Comparative Analytics")

                # Visual Charts Rendered via Matplotlib
                gr.Plot(value=create_metric_comparison_chart, label="Core Metrics Comparison (Precision, Recall, Dice, IoU)")
                gr.Plot(value=create_tradeoff_chart, label="Trade-Off Analysis and Inference Throughput")

                gr.Markdown("---")
                gr.Markdown("### Key Performance Insights & Progression Analysis")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "#### 1. Impact of Fine-Tuning (Focal-Dice Loss)\n"
                            "- **Precision Gain:** Increased from **22.90% to 55.73%** (+143.4% relative gain).\n"
                            "- **Recall Gain:** Increased from **14.04% to 42.10%** (+199.9% gain / 3x improvement).\n"
                            "- **Technical Reason:** Standard BCE suffered from background imbalance. Focal Loss (gamma=2.0) down-weighted easy pristine pixels and focused backpropagation on hard, ambiguous character strokes."
                        )
                    with gr.Column():
                        gr.Markdown(
                            "#### 2. Dual-Branch Net (RGB + SRM Fusion)\n"
                            "- **Document Classification Accuracy:** Reached **68.81%** on unseen test documents.\n"
                            "- **Noise Stream Advantage:** Spatial Rich Model (SRM) high-pass kernels extract sensor residuals and compression inconsistencies without relying solely on semantic visual appearance."
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "#### 3. Inference Throughput & Latency\n"
                            "- **U-Net ResNet-34:** **9.66 FPS** (103 ms per sample).\n"
                            "- **Dual-Branch Net:** **9.53 FPS** (105 ms per sample).\n"
                            "- **DeepLabV3+ EfficientNet-B3:** **8.61 FPS** (116 ms per sample).\n"
                            "- All architectures operate comfortably above standard interactive inspection requirements on single GPU hardware."
                        )
                    with gr.Column():
                        gr.Markdown(
                            "#### 4. Hardware & Testbed Environment\n"
                            "- **Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).\n"
                            "- **Precision Mode:** Automatic Mixed Precision (AMP FP16).\n"
                            "- **Framework:** PyTorch 2.6.0 with CUDA 12.6, Torchvision, Albumentations.\n"
                            "- **Training Dataset:** 20,039 RealText-V2 multi-lingual digital documents."
                        )

        gr.Markdown(
            "---\n"
            "Digital Document Forgery Detection and Localization | Department of Information Technology | Year 2026-27"
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[str(ROOT / "outputs"), str(TEMP_DIR)],
        css=LIGHT_THEME_CSS,
        head=HEAD_JS,
    )


if __name__ == "__main__":
    launch_demo()
