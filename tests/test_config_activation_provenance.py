from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(relative_path: str):
    with (REPO_ROOT / relative_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_wt103_neo_configs_use_tanh_with_activation_bearing_run_tags():
    for size in ("20m", "30m", "50m"):
        cfg = _load_config(f"configs/wt103/neo_{size}.yaml")

        assert cfg["activation_id"] == "tanh"
        assert cfg["run_tag"] == f"wt103_neo_{size}_tanh"


def test_wt2_legacy_config_paths_keep_their_historical_run_tags():
    expected_tags = {
        "configs/wt2/neo_6m.yaml": "wt2_neo_6m",
        "configs/wt2/lstm_6m.yaml": "wt2_lstm_6m",
        "configs/wt2/lstm_25m.yaml": "wt2_lstm_25m",
    }

    for relative_path, expected_tag in expected_tags.items():
        assert _load_config(relative_path)["run_tag"] == expected_tag
