from scripts.rollout_oft_prefix_ablation import _adapter_config


def test_adapter_config_uses_explicit_options_with_import_string() -> None:
    cfg = {
        "adapter": "rase.collect.example:make_adapter",
        "adapter_config": {"libero_plus_root": "/tmp/libero-plus"},
    }

    assert _adapter_config(cfg) == {"libero_plus_root": "/tmp/libero-plus"}


def test_adapter_config_preserves_legacy_mapping() -> None:
    assert _adapter_config({"adapter": {"libero_plus_root": "/legacy"}}) == {
        "libero_plus_root": "/legacy"
    }
