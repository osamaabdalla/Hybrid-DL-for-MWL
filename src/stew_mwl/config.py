
from dataclasses import dataclass, field
from pathlib import Path

CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

RGB_BANDS = ["theta", "alpha", "beta"]

CLASS_NAMES = ["baseline", "low", "medium", "high"]

@dataclass
class Config:
    data_root: Path = Path("data/STEW")
    output_root: Path = Path("outputs")
    sfreq: int = 128
    image_h: int = 80
    image_w: int = 60
    epoch_seconds: float = 2.0
    parent_window_seconds: int = 10
    frame_hop_seconds: float = 1.0
    apply_ica: bool = False
    low_cut: float = 1.0
    high_cut: float = 40.0
    notch_freq: float | None = 50.0
    latent_dim: int = 128
    batch_size: int = 32
    vae_epochs: int = 20
    clf_epochs: int = 35
    learning_rate: float = 1e-3
    dropout: float = 0.3
    loso_subjects_limit: int | None = None
    seed: int = 42
    use_mixed_precision: bool = False
    class_to_id: dict = field(default_factory=lambda: {
        "baseline": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    })
