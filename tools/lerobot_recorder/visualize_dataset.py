#!/usr/bin/env python3
"""Visualize local Aloha teleop LeRobot v3 recordings (Rerun / Foxglove).

Mirrors LeRobot 0.6.1 ``lerobot-dataset-viz``: load one episode via
``LeRobotDataset``, then either stream frames into a Rerun blueprint or serve
seekable Foxglove playback. Adds RoCo path discovery and grouped state/action
panels for the 44-D / 14-D teleop contract.

Requires the conda ``lerobot`` env::

    conda run -n lerobot python tools/lerobot_recorder/visualize_dataset.py \\
      runs/datasets/local_roco_aloha_teleop/<run_id> --episode-index 0
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path

from viz_helpers import (
    ACTION_GROUPS,
    STATE_GROUPS,
    assert_dataset_ready,
    episode_summary_lines,
    load_info_features,
    load_repo_id,
    resolve_dataset_root,
    slice_vector,
    validate_teleop_contract,
)

logger = logging.getLogger(__name__)

DEFAULT_FOXGLOVE_PORT = 8765
DEFAULT_RERUN_PORT = 9090


def build_roco_blueprint(camera_keys: list[str]):
    """Rerun blueprint: cameras + grouped state/action time series."""
    import rerun as rr
    import rerun.blueprint as rrb

    views = [rrb.Spatial2DView(origin=key, name=key) for key in camera_keys]
    for group in (*STATE_GROUPS, *ACTION_GROUPS):
        styling = rr.SeriesLines(names=list(group.names))
        views.append(
            rrb.TimeSeriesView(
                origin=group.entity,
                name=group.entity,
                overrides={group.entity: styling},
            )
        )
    return rrb.Blueprint(rrb.Grid(*views))


def visualize_rerun(
    dataset,
    *,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int | None = None,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
) -> Path | None:
    """Stream one episode into Rerun with RoCo grouped scalar panels."""
    import numpy as np
    import torch.utils.data
    import tqdm
    from lerobot.scripts.lerobot_dataset_viz import to_hwc_uint8_numpy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from lerobot.utils.import_utils import require_package

    if save and output_dir is None:
        raise ValueError("Set --output-dir when --save 1 is set.")

    if mode not in ("local", "distant"):
        raise ValueError(mode)

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    spawn_local_viewer = mode == "local" and not save
    blueprint = build_roco_blueprint(list(dataset.meta.camera_keys))
    rr.init(
        f"{dataset.repo_id}/episode_{episode_index}",
        spawn=spawn_local_viewer,
        default_blueprint=blueprint,
    )
    # Avoid hanging flush with DataLoader workers after rr.init (LeRobot pattern).
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info("Connect with: rerun rerun+http://IP:%s/proxy", grpc_port)
        rr.serve_web_viewer(
            open_browser=False,
            web_port=web_port if web_port is not None else DEFAULT_RERUN_PORT,
            connect_to=server_uri,
        )

    first_index = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()

        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                rr.log(key, entity=entity)

            if ACTION in batch:
                action = batch[ACTION][i].numpy()
                for group in ACTION_GROUPS:
                    values = np.asarray(slice_vector(action, group.indices), dtype=np.float64)
                    rr.log(group.entity, rr.Scalars(values))

            if OBS_STATE in batch:
                state = batch[OBS_STATE][i].numpy()
                for group in STATE_GROUPS:
                    values = np.asarray(slice_vector(state, group.indices), dtype=np.float64)
                    rr.log(group.entity, rr.Scalars(values))

    if mode == "local" and save:
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = dataset.repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    if mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ctrl-C received. Exiting.")
    return None


def visualize_foxglove(
    dataset,
    *,
    episode_index: int,
    host: str = "127.0.0.1",
    port: int = DEFAULT_FOXGLOVE_PORT,
    display_compressed_images: bool = False,
    autoplay: bool = True,
) -> None:
    from lerobot.utils.foxglove_visualization import serve_foxglove_dataset_playback

    logging.info("Starting Foxglove server at ws://%s:%s", host, port)
    serve_foxglove_dataset_playback(
        dataset,
        episode_index,
        host=host,
        port=port,
        compress_images=display_compressed_images,
        autoplay=autoplay,
    )


def open_dataset(root: Path, *, episode_index: int, tolerance_s: float = 1e-4):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Run this with the lerobot conda env: conda run -n lerobot python ..."
        ) from exc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    assert_dataset_ready(root)
    features = load_info_features(root)
    validate_teleop_contract(features)

    repo_id = load_repo_id(root)
    logging.info("Loading %s episode %s from %s", repo_id, episode_index, root)
    return LeRobotDataset(
        repo_id,
        episodes=[episode_index],
        root=root,
        tolerance_s=tolerance_s,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize local Aloha teleop LeRobot datasets (Rerun / Foxglove)."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a local dataset root (runs/datasets/.../<run_id>).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id under runs/datasets/<dataset-folder>/ (alternative to path).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the lexicographically latest run under the dataset folder.",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="local_roco_aloha_teleop",
        help="Folder name under runs/datasets/ when using --run-id / --latest.",
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Episode to visualize.")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance passed to LeRobotDataset.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="Rerun viewer mode (ignored for Foxglove).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help="Rerun distant web port (default 9090) or Foxglove WS port (default 8765).",
    )
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help="If 1, write a .rrd under --output-dir and do not spawn a viewer.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="JPEG-compress images before logging (Rerun) / Foxglove.",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        default="rerun",
        choices=["rerun", "foxglove"],
        help="Visualization backend.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Foxglove WebSocket bind host.",
    )
    parser.add_argument(
        "--no-autoplay",
        dest="autoplay",
        action="store_false",
        help="Foxglove: do not autoplay when a client connects.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.display_mode == "foxglove":
        ignored = []
        if args.mode != "local":
            ignored.append("--mode")
        if args.save:
            ignored.append("--save")
        if args.output_dir is not None:
            ignored.append("--output-dir")
        if args.grpc_port != 9876:
            ignored.append("--grpc-port")
        if args.batch_size != 32:
            ignored.append("--batch-size")
        if args.num_workers != 4:
            ignored.append("--num-workers")
        if ignored:
            logging.warning(
                "These flags only apply to --display-mode rerun and are ignored with foxglove: %s",
                ", ".join(ignored),
            )

    root = resolve_dataset_root(
        args.dataset,
        run_id=args.run_id,
        latest=args.latest,
        dataset_folder=args.dataset_folder,
    )
    for line in episode_summary_lines(root, args.episode_index):
        logging.info("episode %s: %s", args.episode_index, line)

    dataset = open_dataset(root, episode_index=args.episode_index, tolerance_s=args.tolerance_s)

    if args.display_mode == "foxglove":
        visualize_foxglove(
            dataset,
            episode_index=args.episode_index,
            host=args.host,
            port=args.web_port if args.web_port is not None else DEFAULT_FOXGLOVE_PORT,
            display_compressed_images=args.display_compressed_images,
            autoplay=args.autoplay,
        )
        return

    out = visualize_rerun(
        dataset,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
        save=bool(args.save),
        output_dir=args.output_dir,
        display_compressed_images=args.display_compressed_images,
    )
    if out is not None:
        logging.info("Wrote %s", out)


if __name__ == "__main__":
    main()
