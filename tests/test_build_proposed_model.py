import pytest

from stew_mwl.config import Config
from stew_mwl.models import build_proposed_model


@pytest.mark.parametrize("order", ["channel_spatial", "spatial_channel", "parallel"])
def test_build_proposed_model_output_shape(order):
    cfg = Config(
        image_h=80,
        image_w=60,
        parent_window_seconds=10,
        frame_hop_seconds=1.0,
        cbam_attention_order=order,
    )
    m = build_proposed_model(cfg, n_channels=3)
    assert m.name == "vae_cbam_bilstm_classifier"
    assert m.output_shape[-1] == 4


def test_build_proposed_model_respects_cbam_enabled_off():
    cfg = Config(
        image_h=80,
        image_w=60,
        parent_window_seconds=10,
        frame_hop_seconds=1.0,
        cbam_enabled=False,
    )
    m = build_proposed_model(cfg, n_channels=3)
    names = [L.name for L in m.layers if L.name]
    assert not any("cbam" in n.lower() for n in names)
