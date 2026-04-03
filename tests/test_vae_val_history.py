import numpy as np

from stew_mwl.models import build_vae


def test_vae_fit_populates_val_loss():
    vae, _enc, _dec = build_vae(image_shape=(80, 60, 3), latent_dim=8)
    vae.compile(optimizer="adam")
    x = np.random.rand(6, 80, 60, 3).astype(np.float32)
    xv = np.random.rand(2, 80, 60, 3).astype(np.float32)
    hist = vae.fit(x, validation_data=xv, epochs=1, verbose=0)
    assert "val_loss" in hist.history
    assert len(hist.history["val_loss"]) == 1
