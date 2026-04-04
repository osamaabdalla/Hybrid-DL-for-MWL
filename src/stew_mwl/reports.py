"""Manuscript-style consolidated tables from exported CSVs."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .eval import aggregate_fold_metrics


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def build_manuscript_tables(cfg: Config) -> dict[str, Path]:
    """
    Build consolidated tables under `outputs/reports/` from existing `outputs/csv/*.csv`.
    Safe to call after a full pipeline run; missing inputs are skipped.
    """
    cfg.ensure_dirs()
    out: dict[str, Path] = {}
    csv = cfg.csv_dir
    rep = cfg.reports_dir

    # Table A — proposed LOSO headline (mean ± std)
    fold_df = _read_if_exists(csv / "fold_metrics_all.csv")
    if fold_df is not None and len(fold_df):
        if "model_name" in fold_df.columns:
            sub = fold_df[fold_df["model_name"] == "proposed"]
            if len(sub) == 0:
                sub = fold_df
        else:
            sub = fold_df
        _, summ = aggregate_fold_metrics(sub.to_dict("records"))
        main = pd.DataFrame(
            [
                {
                    "model": "proposed_VAE_CBAM_BiLSTM",
                    "mean_accuracy": summ.get("accuracy"),
                    "std_accuracy": summ.get("std_accuracy"),
                    "mean_macro_f1": summ.get("macro_f1"),
                    "std_macro_f1": summ.get("std_macro_f1"),
                    "mean_balanced_accuracy": summ.get("balanced_accuracy"),
                    "mean_cohen_kappa": summ.get("cohen_kappa"),
                }
            ]
        )
        p = rep / "table_main_proposed_loso.csv"
        main.to_csv(p, index=False)
        out["main_proposed"] = p

    # Table B — baselines (from summary file)
    base = _read_if_exists(csv / "baseline_comparison_summary.csv")
    if base is not None and len(base):
        p = rep / "table_baselines.csv"
        base.to_csv(p, index=False)
        out["baselines"] = p

    # Table C — ablations
    abl = _read_if_exists(csv / "ablation_summary.csv")
    if abl is not None and len(abl):
        p = rep / "table_ablations.csv"
        abl.to_csv(p, index=False)
        out["ablations"] = p

    # Table D — statistical tests (paired t-tests)
    stat = _read_if_exists(csv / "statistical_tests.csv")
    if stat is not None and len(stat):
        p = rep / "table_statistical_tests.csv"
        stat.to_csv(p, index=False)
        out["statistical_tests"] = p

    # Table E — sensitivity (concatenate if present)
    sens_parts: list[pd.DataFrame] = []
    for name in (
        "sensitivity_latent_size.csv",
        "sensitivity_cbam.csv",
        "sensitivity_window.csv",
        "sensitivity_sequence_steps.csv",
    ):
        df = _read_if_exists(csv / name)
        if df is not None and len(df):
            sens_parts.append(df.assign(sensitivity_file=name))
    if sens_parts:
        p = rep / "table_sensitivity_combined.csv"
        pd.concat(sens_parts, ignore_index=True).to_csv(p, index=False)
        out["sensitivity_combined"] = p

    # Table F — Grad-CAM region summary
    gc = _read_if_exists(csv / "gradcam_region_summary.csv")
    if gc is not None and len(gc):
        p = rep / "table_gradcam_regions.csv"
        gc.to_csv(p, index=False)
        out["gradcam_regions"] = p

    # Master index markdown
    lines = ["# Reproduction tables (auto-generated)\n", ""]
    for k, v in sorted(out.items()):
        lines.append(f"- **{k}**: `{v.relative_to(cfg.output_root)}`\n")
    lines.append("\nSource CSVs: `outputs/csv/`.\n")
    idx = rep / "MANUSCRIPT_TABLES.md"
    idx.write_text("".join(lines), encoding="utf-8")
    out["index_md"] = idx
    return out


def write_run_manifest(cfg: Config, extra: dict[str, Any] | None = None) -> Path:
    """Single JSON-line friendly summary of output locations for auditing."""
    cfg.ensure_dirs()
    rows = []
    for p in sorted(cfg.csv_dir.glob("*.csv")):
        rows.append({"kind": "csv", "name": p.name, "path": str(p.resolve())})
    for p in sorted(cfg.figures_dir.glob("*.png")):
        rows.append({"kind": "figure", "name": p.name, "path": str(p.resolve())})
    for p in sorted(cfg.reports_dir.glob("*")):
        if p.is_file():
            rows.append({"kind": "report", "name": p.name, "path": str(p.resolve())})
    df = pd.DataFrame(rows)
    path = cfg.logs_dir / "output_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if extra:
        pd.DataFrame([{"meta": str(extra)}]).to_csv(cfg.logs_dir / "run_extra.csv", index=False)
    return path
