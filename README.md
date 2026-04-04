# STEW Mental Workload Reproduction — Hybrid-DL-for-MWL

Open-source reproduction pipeline for:

**Hybrid deep learning for mental workload classification using EEG with enhanced preprocessing and interpretability**

The implementation follows the product requirements in `PRD.md` (manuscript-aligned protocol, CSV exports, config-driven runs).

## What you get

- **Notebook:** `notebooks/stew_mwl_reproduction.ipynb` — end-to-end: dataset validation → LOSO proposed model → baselines → ablations → sensitivity (if enabled) → Grad-CAM CSV + figures → manuscript-style report tables → output manifest.
- **Package:** `src/stew_mwl/` — preprocessing cache, topomaps (Welch / Morlet), VAE + CBAM + BiLSTM, exponential or cosine LR schedule, early stopping on `val_macro_f1` or `val_loss`, LOSO, baselines, ablations, sensitivity, Grad-CAM, export, `reports`, plotting.
- **Configs:** `configs/default.yaml` (fast smoke), `configs/full_reproduction.yaml` (48-subject-style, **preprocessed signal cache on**, sensitivity on).
- **License:** `LICENSE` (MIT).
- **Determinism:** fixed seeds, `TF_DETERMINISTIC_OPS`, pinned versions in `requirements.txt`. Metrics may still vary slightly across OS / CPU / GPU.

## Classes (manuscript codes)

| Code | Meaning        | Rating (task) |
|------|----------------|---------------|
| BL   | Baseline (lo)  | —             |
| LW   | Low workload | 1–3           |
| MW   | Medium       | 4–6           |
| HW   | High         | 7–9           |

## Dataset

1. Download **STEW** from the official source (IEEE DataPort, DOI: 10.21227/44r8-ya50).
2. Extract under `data/STEW/` (or set `paths.raw_data_dir` in YAML).
3. Text signals: `sub01_lo.txt`, `sub01_hi.txt`, … (14 columns, 128 Hz).
4. A file whose name contains **`rating`** is required for task labels.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/stew_mwl_reproduction.ipynb
```

### Config (YAML) — always pass `project_root`

So `data_root`, `output_root`, `interim_dir`, and `processed_data_dir` resolve to the **repository root** (not `notebooks/`):

```python
from pathlib import Path
from stew_mwl.yaml_loader import load_config_from_yaml

ROOT = Path(__file__).resolve().parent  # or notebook: Path("..").resolve()
cfg = load_config_from_yaml(ROOT / "configs/default.yaml", project_root=ROOT)
```

Training-related YAML keys (see `configs/*.yaml`):

- `training.lr_schedule`: `exponential` (default, manuscript-style), `cosine`, or `none`
- `training.early_stopping_monitor`: `val_macro_f1` (default) or `val_loss`
- `cache.preprocessed`: if `true`, filtered signals are written under `paths.interim_data_dir` (invalidated when source `.txt` mtime changes)
- `cache.sequences`: if `true`, sequence tensors (topomap stacks) are cached as `.npz` under `paths.processed_data_dir` (keyed by config + source mtime)
- `preprocessing.reference_mode`: `none`, `average` (common average reference per sample), or `cz_proxy`. If `none`, legacy `cz_proxy_reference: true` still selects Cz-proxy referencing in `preprocess_signal`
- `dataset.strict_signal_audit` / `dataset.min_recording_samples`: optional full-file checks when building the manifest
- `dataset.verify_stew_conventions`: if `true`, `sfreq` must be **128** (STEW convention; PRD A2-style check on config, not file headers)
- `cbam.enabled`: if `false`, the **main LOSO proposed** run omits CBAM; ablations and sensitivity still use each variant’s explicit `use_cbam` flags (PRD J2 fairness)
- `cbam.attention_order`: `channel_spatial` (default), `spatial_channel`, or `parallel` (PRD G1)
- `vae.val_fraction`: held-out frame fraction for VAE training; `csv/vae_fold_losses.csv` includes `val_total_loss` when Keras reports `val_loss`
- **PSD–SVM baseline:** band-power features (theta, alpha, beta × 14 channels) per sequence window, same parent-window segmentation as the CNN inputs; pipeline is `StandardScaler` → `PCA` → RBF `SVC`

## Expected outputs

| Path | Description |
|------|-------------|
| `csv/dataset_manifest.csv` | Subjects, paths, ratings, class codes |
| `csv/segmentation_summary.csv` | Samples, `num_epochs`, sequences, window/steps |
| `csv/fold_metrics_all.csv` | LOSO fold metrics (proposed) |
| `csv/predictions_all.csv` | Per-sequence predictions + probabilities |
| `csv/classification_report_proposed.csv` | Per-class precision / recall / F1 |
| `csv/confusion_matrix_proposed.csv` | BL–HW confusion matrix |
| `csv/vae_fold_losses.csv` | VAE losses per fold |
| `csv/vae_latent_summary.csv` | Per-fold, per-class latent norms (train pool) |
| `csv/baseline_*_fold_metrics.csv` | PSD-SVM, CNN, BLSTM–LSTM |
| `csv/baseline_comparison_summary.csv` | Baseline means ± std |
| `csv/ablation_*.csv` | Ablations |
| `csv/sensitivity_*.csv` | Sensitivity sweeps |
| `csv/statistical_tests.csv` | Paired t-test + Wilcoxon signed-rank vs baselines/ablations (per subject) |
| `csv/gradcam_*.csv` | Regional / sample Grad-CAM scores |
| `csv/experiment_registry.csv` | Run metadata |
| `csv/cbam_config_results.csv` | Active CBAM YAML settings + optional sensitivity sweep rows (PRD G3) |
| `reports/table_*.csv`, `MANUSCRIPT_TABLES.md` | Consolidated tables (PRD Section 14) |
| `logs/output_manifest.csv` | Index of generated CSV/PNG/report paths |
| `models/fold_XX_proposed.keras` | Checkpoints |
| `figures/*.png` | Confusion matrix, baselines, ablations, VAE, Grad-CAM |

Generated `outputs/`, `data/STEW/interim/*.npz`, and `data/STEW/processed/*.npz` (when sequence cache is on) are **gitignored** by default.

## Repository layout

```text
.
├── LICENSE
├── PRD.md
├── README.md
├── configs/
│   ├── default.yaml
│   └── full_reproduction.yaml
├── notebooks/
│   └── stew_mwl_reproduction.ipynb
├── src/
│   └── stew_mwl/
│       ├── config.py
│       ├── yaml_loader.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── train.py
│       ├── eval.py
│       ├── gradcam.py
│       ├── attention.py
│       ├── export.py
│       ├── plotting.py
│       └── reports.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Reproducibility scope

The repo reproduces the **protocol**, **splits**, **metrics**, and **table structure** from the manuscript. It does **not** guarantee bit-identical weights on every machine; see `PRD.md` section 9.

## How to cite

If you use this reproduction code or protocol, cite the **original STEW dataset** (see IEEE DataPort / manuscript) and reference this repository by URL and access date. Add your own author line if you publish a derivative.

## License

MIT — see `LICENSE`. Update the copyright line if you redistribute under a different entity.
