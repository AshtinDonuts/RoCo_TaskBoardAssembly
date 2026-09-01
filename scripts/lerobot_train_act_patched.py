#!/usr/bin/env python
"""Run the local LeRobot trainer.

The installed LeRobot maps episode-filtered sampler indices to the relative
dataset indices itself, so no DatasetReader monkey patch is needed.
"""
from __future__ import annotations

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
