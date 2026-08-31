#!/usr/bin/env python
"""Run lerobot_train with a fix for episode-filtered absolute frame indices.

Current LeRobot EpisodeAwareSampler yields absolute ``index`` values from
``meta/episodes``, while ``DatasetReader.get_item`` indexes the filtered HF
dataset relatively. Map absolute→relative when a subset of episodes is loaded.
"""
from __future__ import annotations

from lerobot.datasets.dataset_reader import DatasetReader

_orig_get_item = DatasetReader.get_item


def _get_item(self, idx):
    mapping = getattr(self, "_absolute_to_relative_idx", None)
    if mapping is not None:
        try:
            idx = mapping[idx]
        except KeyError as exc:
            raise IndexError(
                f"Absolute frame index {idx} is not in the loaded episode subset"
            ) from exc
    return _orig_get_item(self, idx)


DatasetReader.get_item = _get_item

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
