"""Tests for Aloha teleop dataset viz helpers (no Rerun / Foxglove spawn)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "lerobot_recorder"))

from viz_helpers import (  # noqa: E402
    ACTION_DIM,
    ACTION_GROUPS,
    ACTION_NAMES,
    CAMERA_FEATURE_KEYS,
    STATE_DIM,
    STATE_GROUPS,
    STATE_NAMES,
    assert_dataset_ready,
    load_info_features,
    load_repo_id,
    resolve_dataset_root,
    slice_vector,
    validate_teleop_contract,
)


def _features_ok() -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": list(ACTION_NAMES),
        },
    }
    for key in CAMERA_FEATURE_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [240, 320, 3],
            "names": ["height", "width", "channels"],
        }
    return features


def test_state_action_groups_cover_contract() -> None:
    state_idx = [i for g in STATE_GROUPS for i in g.indices]
    action_idx = [i for g in ACTION_GROUPS for i in g.indices]
    assert state_idx == list(range(STATE_DIM))
    assert action_idx == list(range(ACTION_DIM))

    for group in STATE_GROUPS:
        assert tuple(STATE_NAMES[i] for i in group.indices) == group.names
    for group in ACTION_GROUPS:
        assert tuple(ACTION_NAMES[i] for i in group.indices) == group.names


def test_slice_vector() -> None:
    assert slice_vector(list(range(10)), (1, 3, 5)) == [1.0, 3.0, 5.0]


def test_validate_teleop_contract_ok() -> None:
    validate_teleop_contract(_features_ok())


def test_validate_teleop_contract_bad_shape() -> None:
    features = _features_ok()
    features["observation.state"]["shape"] = [43]
    with pytest.raises(ValueError, match="observation.state shape"):
        validate_teleop_contract(features)


def test_validate_teleop_contract_bad_names() -> None:
    features = _features_ok()
    features["action"]["names"] = list(ACTION_NAMES)
    features["action"]["names"][0] = "wrong"
    with pytest.raises(ValueError, match="ACTION_NAMES"):
        validate_teleop_contract(features)


def test_validate_teleop_contract_missing_camera() -> None:
    features = _features_ok()
    del features["observation.images.head"]
    with pytest.raises(ValueError, match="observation.images.head"):
        validate_teleop_contract(features)


def test_resolve_dataset_root_path(tmp_path: Path) -> None:
    ds = tmp_path / "run_a"
    ds.mkdir()
    assert resolve_dataset_root(ds) == ds.resolve()


def test_resolve_dataset_root_run_id_and_latest(tmp_path: Path) -> None:
    folder = tmp_path / "local_roco_aloha_teleop"
    (folder / "20260816_114641").mkdir(parents=True)
    (folder / "20260816_122330").mkdir(parents=True)

    assert (
        resolve_dataset_root(run_id="20260816_114641", runs_root=tmp_path).name
        == "20260816_114641"
    )
    assert resolve_dataset_root(latest=True, runs_root=tmp_path).name == "20260816_122330"


def test_resolve_dataset_root_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_dataset_root(latest=True, run_id="x")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_dataset_root()


def test_load_repo_id_and_assert(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "run_meta.json").write_text(
        json.dumps({"repo_id": "local/roco_aloha_teleop"}), encoding="utf-8"
    )
    (root / "meta" / "info.json").write_text(
        json.dumps({"features": _features_ok()}), encoding="utf-8"
    )

    assert load_repo_id(root) == "local/roco_aloha_teleop"
    assert_dataset_ready(root)
    validate_teleop_contract(load_info_features(root))


def test_assert_dataset_ready_rejects_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "quarantine" / "run"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="quarantine"):
        assert_dataset_ready(root)


def test_real_sample_info_matches_contract() -> None:
    sample = (
        ROOT
        / "runs"
        / "datasets"
        / "local_roco_aloha_teleop"
        / "20260816_122330"
        / "meta"
        / "info.json"
    )
    if not sample.is_file():
        pytest.skip("sample dataset not present")
    features = json.loads(sample.read_text(encoding="utf-8"))["features"]
    validate_teleop_contract(features)
