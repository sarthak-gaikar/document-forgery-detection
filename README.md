# Deep Learning-Based Tampered Text Detection and Localization

> **Forensic Assistance Tool — NOT a Legal Verdict**
> This system is an AI-assisted forensic *aid* only. Its outputs must never be
> used as conclusive legal evidence. Always involve a qualified human forensic
> expert for any legal proceedings.

---

## Project Overview

A deep-learning pipeline that accepts a digital document (PDF or image) and outputs:

1. **Classification**: Genuine / Forged + confidence score
2. **Localization**: Pixel-level tamper mask + heatmap overlay
3. **Forgery type** *(optional)*: text replacement, copy-move, splicing, insertion/removal

### Scope
| In scope | Out of scope (limitations) |
|---|---|
| Digital document images (rendered PDFs, screenshots, PNG/JPEG exports) | Camera photos of physical documents |
| Printed / typed English text | Handwritten text |
| Receipts, invoices, forms, certificates, marksheets | PDF structure / metadata forensics |
| Residual artifact-based detection | Perfect edits with zero residual artifacts |

---

## Repository Structure

`
forgery_detection/
├── configs/               # YAML experiment configs
│   ├── base.yaml          # Shared defaults
│   ├── unet_resnet34.yaml # Model A
│   ├── deeplabv3plus_efficientnet.yaml  # Model B
│   ├── segformer_b2.yaml  # Model C
│   ├── dual_branch.yaml   # Model D (proposed)
│   ├── classifiers.yaml   # Models E/F/G
│   └── synthetic_gen.yaml # Synthetic generator settings
│
├── src/
│   ├── data/              # Dataset loaders, augmentations, splits
│   ├── models/            # Model architectures (A–D + classifiers E–G)
│   ├── losses/            # BCE+Dice, Focal+Dice losses
│   ├── metrics/           # IoU, Dice, AUC, etc.
│   ├── training/          # Train loop, scheduler, checkpointing
│   ├── inference/         # Tiling inference pipeline
│   ├── api/               # FastAPI backend
│   └── utils.py           # Shared utilities
│
├── scripts/
│   ├── check_env.py       # Phase 1: environment check
│   ├── inspect_datasets.py# Phase 2: dataset inspection
│   ├── generate_synthetic.py  # Phase 3: synthetic data generation
│   └── train.py           # Phase 6+: training entry point
│
├── tests/                 # Unit tests (pytest)
├── notebooks/             # Exploratory notebooks + Colab/Kaggle training nb
├── outputs/
│   ├── checkpoints/       # Model checkpoints (.pth)
│   ├── logs/              # TensorBoard + CSV logs
│   ├── results/           # Metric CSVs, comparison tables
│   └── figures/           # Plots, heatmaps, overlays
│
├── data/                  # (not in git) – datasets stored here
│   ├── DocTamper/         # LMDB files + split lists
│   ├── synthetic/         # Generated data (images, masks, JSONs)
│   └── realtext_v2/       # HuggingFace cache
│
├── requirements.txt
└── README.md
`

---

## Datasets

### 1. DocTamper (primary)
~170K images with pixel-level masks, LMDB format.
See https://github.com/qcf-568/DocTamper for access instructions.
Place LMDB files under `data/DocTamper/lmdb/`.

### 2. Synthetic Dataset (self-built, >= 50 K samples)
Generated from `scripts/generate_synthetic.py`.
- **Source docs**: 15,000 unique documents (receipts, invoices, forms, certificates, marksheets)
- **Tamper types**: text_replacement, copy_move, splicing, text_insertion, text_removal
- **Held-out type**: splicing (generalization test)
- **Split**: grouped by `source_doc_id` to prevent leakage

### 3. RealText-V2 (supplementary / cross-domain)
HuggingFace: `vankey/RealText-V2` – used for cross-domain evaluation only.

---

## Models Compared

| ID | Name | Type |
|----|------|------|
| A | U-Net + ResNet34 | Segmentation baseline |
| B | DeepLabV3+ + EfficientNet-B3 | Segmentation |
| C | SegFormer-B2 | Segmentation (transformer) |
| D | **Dual-Branch Net** (proposed) | Multitask (seg + cls) |
| E | Custom CNN from scratch | Classification |
| F | ResNet-50 (pretrained) | Classification |
| G | EfficientNet-B0 (pretrained) | Classification |

Model selection is **purely metric-driven** — no pre-judging.

---

## Quick Start

### 1. Create virtual environment
`bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
`

### 2. Install dependencies
`bash
# CPU (testing):
pip install -r requirements.txt

# GPU (training) – install PyTorch first, then the rest:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
`

### 3. Verify environment
`bash
python scripts/check_env.py
`

### 4. Phase-by-phase execution
Each phase has its own script. Run them in order after verifying the previous phase.

---

## Training

`bash
python scripts/train.py --config configs/dual_branch.yaml --resume  # resumable
`

Logs written to `outputs/logs/`, checkpoints to `outputs/checkpoints/`.

---

## Inference

`bash
python scripts/infer.py --input path/to/document.pdf --output outputs/results/
`

---

## Colab / Kaggle Training

See `notebooks/train_colab.ipynb` for a ready-to-run notebook with:
- Drive/Kaggle dataset mounting
- GPU environment setup
- Training with checkpointing to Drive

---

## Limitations

1. **Synthetic-to-real gap**: The model is trained largely on synthetic data; performance on real-world forgeries may be lower.
2. **Perfect edits**: Forgeries with zero residual artifacts (font-matched, noise-matched) will fool the detector.
3. **No camera photos**: The model expects flat, digitally rendered document images.
4. **English text only**: Non-English scripts are out of scope.
5. **Not a legal verdict**: See disclaimer above.

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project skeleton, configs, README, env check | **Done** |
| 2 | Dataset inspection (scripts/inspect_datasets.py) | **Done** |
| 3 | Synthetic dataset generator + verification (src/data/synthetic_gen.py) | **Done** |
| 4 | PyTorch Dataset/DataLoader + augmentations (src/data/) | **Done** |
| 5 | Metrics, losses, training loop (src/losses/, src/metrics/, src/training/) | **Done** |
| 6 | Train & compare models A–G (src/models/, scripts/train.py) | **Done** |
| 7 | Ablations, robustness, cross-domain | Ready to benchmark |
| 8 | Retrain winner on full data | Ready |
| 9 | Inference pipeline (tiling, heatmaps) (src/inference/, scripts/infer.py) | **Done** |
| 10 | FastAPI backend + demo UI (src/api/app.py, src/api/demo.py) | **Done** |
| 11 | Unit tests (60/60 passing) (	ests/) | **Done** |

---

## Citation / Acknowledgements

- DocTamper: Cui et al., CVPR 2023
- SegFormer: Xie et al., NeurIPS 2021
- segmentation_models_pytorch library
