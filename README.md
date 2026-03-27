# STEW Mental Workload Reproduction Package "Hybrid-DL-for-MWL"

This repository provides a clean, organized, open-source reproduction pipeline for the manuscript:

**Hybrid deep learning for mental workload classification using EEG with enhanced preprocessing and interpretability**

It includes:

- a Jupyter notebook that runs end-to-end
- deterministic preprocessing and training settings
- support for the STEW dataset file structure
- baseline models, ablation studies, sensitivity analyses, and Grad-CAM
- helper modules under `src/stew_mwl/`

## What this package reproduces

The pipeline follows the manuscript design:

- dataset: STEW
- labels: baseline + low + medium + high
- preprocessing: band-pass filtering, optional ICA, segmentation
- feature representation: EEG topographical image sequences
- model: VAE + CBAM + BiLSTM
- evaluation: LOSO cross-validation
- additional analyses:
  - PSD-SVM baseline
  - CNN baseline
  - BLSTM-LSTM baseline
  - ablation studies
  - window-size sensitivity
  - temporal-length sensitivity
  - Grad-CAM qualitative and quantitative summaries

## Important reproducibility note

This package is configured for deterministic execution as far as practical:
- fixed random seeds
- deterministic TensorFlow ops
- pinned package versions

Even with those settings, exact floating-point agreement across operating systems, CPU/GPU stacks, and BLAS implementations cannot always be guaranteed. The pipeline is written to maximize reproducibility, but tiny variations may still occur across environments.

## Dataset

Download the STEW dataset from the official source and extract it under:

`data/STEW/`

Expected structure is flexible, but the code supports the common naming convention:

- `sub01_lo.txt` for baseline / rest
- `sub01_hi.txt` for task
- a ratings file containing subject ratings from 1 to 9

The code auto-detects the ratings file and maps:
- 1–3 => low
- 4–6 => medium
- 7–9 => high

The final four classes are:
- baseline
- low
- medium
- high

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/stew_mwl_reproduction.ipynb
```

## Repository layout

```text
.
├── notebooks/
│   └── stew_mwl_reproduction.ipynb
├── src/
│   └── stew_mwl/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── eval.py
│       └── gradcam.py
├── requirements.txt
└── README.md
```
