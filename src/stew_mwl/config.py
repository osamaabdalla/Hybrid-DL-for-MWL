
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

RGB_BANDS = ["theta", "alpha", "beta"]

CLASS_NAMES = ["BL", "LW", "MW", "HW"]

@dataclass
class Config:
    data_root: Path = Path("data/STEW")
    output_root: Path = Path("outputs")
    interim_dir: Path = Path("data/STEW/interim")
    processed_dir: Path = Path("data/STEW/processed")
    sfreq: int = 128
    image_h: int = 80
    image_w: int = 60
    epoch_seconds: float = 2.0
    parent_window_seconds: int = 10
    frame_hop_seconds: float = 1.0
    feature_method: str = "welch"  # welch | morlet
    apply_ica: bool = False
    low_cut: float = 1.0
    high_cut: float = 40.0
    notch_freq: float | None = 50.0
    reference_mode: str = "none"  # none | average | cz_proxy (cz_proxy_reference YAML still maps here)
    cz_proxy_reference: bool = False
    cache_preprocessed: bool = False
    cache_sequences: bool = False
    latent_dim: int = 128
    batch_size: int = 32
    vae_epochs: int = 20
    clf_epochs: int = 35
    learning_rate: float = 1e-3
    lr_schedule: str = "exponential"  # exponential | cosine | none
    early_stopping_monitor: str = "val_macro_f1"  # val_macro_f1 | val_loss
    early_stopping_patience: int = 10
    dropout: float = 0.3
    loso_subjects_limit: int | None = None
    seed: int = 42
    use_mixed_precision: bool = False
    blstm_units: int = 20
    cbam_reduction_ratio: int = 8
    cbam_spatial_kernel: int = 7
    cbam_enabled: bool = True
    cbam_attention_order: str = "channel_spatial"  # channel_spatial | spatial_channel | parallel
    vae_val_fraction: float = 0.1
    quick_mode: bool = False
    run_sensitivity: bool = False
    strict_subject_count: bool = False
    strict_signal_audit: bool = False
    verify_stew_conventions: bool = False
    min_recording_samples: int = 0
    expected_n_subjects: int = 48
    config_path: Optional[Path] = None
    class_to_id: dict = field(default_factory=lambda: {
        "BL": 0,
        "LW": 1,
        "MW": 2,
        "HW": 3,
    })

    @property
    def csv_dir(self) -> Path:
        return self.output_root / "csv"

    @property
    def models_dir(self) -> Path:
        return self.output_root / "models"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def seq_len(self) -> int:
        hop = max(float(self.frame_hop_seconds), 1e-6)
        return max(1, int(round(self.parent_window_seconds / hop)))

    def ensure_dirs(self) -> None:
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_preprocessed:
            self.interim_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_sequences:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
