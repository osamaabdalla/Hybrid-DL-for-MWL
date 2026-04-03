from stew_mwl.attention import cbam_block


def test_attention_cbam_is_models_cbam():
    from stew_mwl.models import cbam_block as m_cbam

    assert cbam_block is m_cbam
