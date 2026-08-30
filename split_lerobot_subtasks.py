#!/usr/bin/env python3
"""Split the RoCo full-assembly LeRobot v3 dataset into per-part episodes.

Observation/action rows and every camera video are cut at the same inferred
frame boundaries.  Each output episode has episode-local frame/video timestamps
and can therefore be loaded normally by LeRobot without referring to a time
window inside the original full-assembly MP4.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Actual order in the published trajectories (the two gears differ from the
# current simulator config order). XY values are the scripted pick targets.
PARTS = (
    ("gear_20teeth", (0.14366, -0.04300), "pick up the 20-tooth gear and place it on the task board"),
    ("gear_60teeth", (0.12280666, -0.08788925), "pick up the 60-tooth gear and place it on the task board"),
    ("rod_16mm", (0.08863274, 0.05815318), "pick up the 16 millimeter rod and insert it into its target slot"),
    ("bolt_8mm", (0.04415, 0.00093), "pick up the 8 millimeter bolt and insert it into its target slot"),
    ("usb_a", (0.02506, -0.07027), "pick up the USB-A connector and insert it into its target slot"),
    ("hdmi", (0.27285, -0.01949), "pick up the HDMI connector and insert it into its target slot"),
    ("pin", (0.18600, -0.01476), "pick up the pin and insert it into its target slot"),
    ("battery_size1", (0.03362, 0.15554), "pick up the small battery and place it into its holder"),
    ("battery_size5", (-0.01071, 0.16490), "pick up the large battery and place it into its holder"),
)


def _replace(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise ValueError(f"missing required column: {name}")
    return table.set_column(idx, name, pa.array(values, type=table.schema.field(idx).type))


def _load_nested_parquet(path: Path) -> pa.Table:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return pa.concat_tables([pq.read_table(p) for p in files], promote_options="default")


def _segment_starts(action: np.ndarray, min_segment: int, tolerance: float) -> tuple[list[int], list[float]]:
    """Return half-open segment starts, inferred from fixed-order XY pick commands.

    Frame zero is always the first part's start.  For every later part, find
    the closest action XY to its configured target in the range that leaves
    ``min_segment`` frames for both adjacent segments.  The returned starts,
    plus the episode length, form the ``[begin, end)`` segment boundaries.
    """
    xy = action[:, :2]
    n = len(xy)
    starts = [0]
    errors = [float(np.linalg.norm(xy[0] - np.asarray(PARTS[0][1])))]

    for part_idx in range(1, len(PARTS)):
        # Do not search early enough to make the preceding segment too short.
        lo = starts[-1] + min_segment
        # Likewise, reserve the minimum length for every later segment.
        remaining = len(PARTS) - part_idx - 1
        hi = n - remaining * min_segment
        if lo >= hi:
            raise ValueError(f"episode too short while locating part {PARTS[part_idx][0]}")

        anchor = np.asarray(PARTS[part_idx][1], dtype=np.float64)
        # Select the strongest candidate for this part's scripted pick target.
        dist = np.linalg.norm(xy[lo:hi] - anchor, axis=1)
        rel_best = int(np.argmin(dist))
        best = lo + rel_best
        best_error = float(dist[rel_best])
        if best_error > tolerance:
            raise ValueError(
                f"pick anchor for {PARTS[part_idx][0]} is {best_error:.4f} m away "
                f"(limit {tolerance:.4f} m)"
            )

        # Walk back to the beginning of this target plateau. A small margin
        # handles randomized offsets without absorbing the preceding skill.
        near = max(tolerance, best_error + 0.005)
        start = best
        while start > lo and np.linalg.norm(xy[start - 1] - anchor) <= near:
            start -= 1
        starts.append(start)
        errors.append(best_error)

    bounds = starts + [n]
    lengths = np.diff(bounds)
    if np.any(lengths < min_segment):
        raise ValueError(f"short segment(s): {lengths.tolist()}")
    return starts, errors


def _scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    x = np.asarray(values)
    return {
        "min": [x.min().item()],
        "max": [x.max().item()],
        "mean": [float(x.mean())],
        "std": [float(x.std())],
        "count": [int(x.size)],
        "q01": [float(np.quantile(x, 0.01))],
        "q10": [float(np.quantile(x, 0.10))],
        "q50": [float(np.quantile(x, 0.50))],
        "q90": [float(np.quantile(x, 0.90))],
        "q99": [float(np.quantile(x, 0.99))],
    }


def _video_path(info: dict, root: Path, key: str, chunk_index: int, file_index: int) -> Path:
    """Resolve a LeRobot video path from the template in ``meta/info.json``."""
    relative = info["video_path"].format(
        video_key=key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return root / relative


def _splice_episode_videos(
    source: Path,
    destination: Path,
    info: dict,
    parent: dict,
    parent_idx: int,
    bounds: list[int],
    video_keys: list[str],
    fps: int,
    ffmpeg: str,
) -> None:
    """Decode each parent camera once and encode its nine frame-exact clips.

    A source MP4 may contain more than one parent episode.  Its episode metadata
    supplies the absolute start time, which is converted to a source frame and
    added to each episode-local boundary.  Frame-based ``trim`` keeps images in
    exact one-to-one correspondence with the sliced parquet rows.
    """
    chunks_size = int(info["chunks_size"])

    for key in video_keys:
        source_chunk = int(parent[f"videos/{key}/chunk_index"])
        source_file = int(parent[f"videos/{key}/file_index"])
        source_video = _video_path(info, source, key, source_chunk, source_file)
        if not source_video.is_file():
            raise FileNotFoundError(f"missing source video: {source_video}")

        base_time = float(parent[f"videos/{key}/from_timestamp"])
        end_time = float(parent[f"videos/{key}/to_timestamp"])
        base_frame = round(base_time * fps)
        if abs(base_time - base_frame / fps) > 1e-4:
            raise ValueError(
                f"episode {parent_idx} video {key} starts off the {fps} fps frame grid: "
                f"{base_time} s"
            )
        video_frames = round((end_time - base_time) * fps)
        if video_frames != bounds[-1]:
            raise ValueError(
                f"episode {parent_idx} video {key} has {video_frames} metadata frames, "
                f"but its observation/action table has {bounds[-1]} rows"
            )

        filters: list[str] = []
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_video)]
        for part_idx, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            filters.append(
                f"[0:v]trim=start_frame={base_frame + begin}:end_frame={base_frame + end},"
                f"setpts=PTS-STARTPTS[v{part_idx}]"
            )

        command.extend(["-filter_complex", ";".join(filters)])
        for part_idx in range(len(PARTS)):
            episode_idx = parent_idx * len(PARTS) + part_idx
            chunk_index, file_index = divmod(episode_idx, chunks_size)
            output_video = _video_path(info, destination, key, chunk_index, file_index)
            output_video.parent.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    "-map",
                    f"[v{part_idx}]",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(fps),
                    str(output_video),
                ]
            )

        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg failed while splicing episode {parent_idx}, video {key}"
            ) from exc


def split_dataset(
    source: Path,
    destination: Path,
    min_segment: int,
    tolerance: float,
    ffmpeg: str = "ffmpeg",
) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    info = json.loads((source / "meta/info.json").read_text())
    fps = int(info["fps"])
    if int(info["total_episodes"]) != 200:
        raise ValueError("this splitter expects the published 200-episode RoCo dataset")

    parent_meta_table = _load_nested_parquet(source / "meta/episodes")
    parent_meta = {int(row["episode_index"]): row for row in parent_meta_table.to_pylist()}
    video_keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    episode_rows: list[dict] = []
    all_episode_index: list[np.ndarray] = []
    all_frame_index: list[np.ndarray] = []
    all_timestamp: list[np.ndarray] = []
    all_task_index: list[np.ndarray] = []
    max_errors: list[float] = []

    try:
        for data_path in sorted((source / "data").rglob("*.parquet")):
            table = pq.read_table(data_path)
            old_episode = np.asarray(table["episode_index"], dtype=np.int64)
            old_index = np.asarray(table["index"], dtype=np.int64)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)

            new_episode = np.empty(len(table), dtype=np.int64)
            new_frame = np.empty(len(table), dtype=np.int64)
            new_timestamp = np.empty(len(table), dtype=np.float32)
            new_task = np.empty(len(table), dtype=np.int64)

            for parent_idx in np.unique(old_episode):
                rows = np.flatnonzero(old_episode == parent_idx)
                if not np.all(np.diff(rows) == 1):
                    raise ValueError(f"episode {parent_idx} is not contiguous in {data_path}")
                local_action = actions[rows]
                starts, errors = _segment_starts(local_action, min_segment, tolerance)
                max_errors.extend(errors[1:])
                bounds = starts + [len(rows)]
                parent = parent_meta[int(parent_idx)]

                # Materialize frame-exact clips before writing metadata that
                # points LeRobot at those new files.
                _splice_episode_videos(
                    source,
                    temp,
                    info,
                    parent,
                    int(parent_idx),
                    bounds,
                    video_keys,
                    fps,
                    ffmpeg,
                )

                # Each range is half-open: it contains ``begin`` and excludes
                # ``end``. This is also the convention used for video times.
                for part_idx, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
                    seg_rows = rows[begin:end]
                    episode_idx = int(parent_idx) * len(PARTS) + part_idx
                    length = end - begin
                    # Data indices are episode-local in the derived dataset.
                    # ``old_index`` remains the global source-dataset index.
                    frame = np.arange(length, dtype=np.int64)
                    timestamp = frame.astype(np.float32) / fps

                    new_episode[seg_rows] = episode_idx
                    new_frame[seg_rows] = frame
                    new_timestamp[seg_rows] = timestamp
                    new_task[seg_rows] = part_idx

                    meta = dict(parent)
                    meta.update(
                        episode_index=episode_idx,
                        tasks=[PARTS[part_idx][2]],
                        length=length,
                        dataset_from_index=int(old_index[seg_rows[0]]),
                        dataset_to_index=int(old_index[seg_rows[-1]]) + 1,
                    )
                    for key in video_keys:
                        # Each derived episode owns one MP4 and starts at t=0.
                        chunk_index, file_index = divmod(episode_idx, int(info["chunks_size"]))
                        meta[f"videos/{key}/chunk_index"] = chunk_index
                        meta[f"videos/{key}/file_index"] = file_index
                        meta[f"videos/{key}/from_timestamp"] = 0.0
                        meta[f"videos/{key}/to_timestamp"] = length / fps
                    # Keep expensive visual/state/action per-episode stats from
                    # the parent; refresh all changed bookkeeping stats.
                    for key, vals in {
                        "episode_index": np.full(length, episode_idx),
                        "frame_index": frame,
                        "timestamp": timestamp,
                        "task_index": np.full(length, part_idx),
                        "index": old_index[seg_rows],
                    }.items():
                        for stat, value in _scalar_stats(vals).items():
                            meta[f"stats/{key}/{stat}"] = value
                    episode_rows.append(meta)

            table = _replace(table, "episode_index", new_episode)
            table = _replace(table, "frame_index", new_frame)
            table = _replace(table, "timestamp", new_timestamp)
            table = _replace(table, "task_index", new_task)
            out_path = temp / data_path.relative_to(source)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_path, compression="snappy", use_dictionary=True)

            all_episode_index.append(new_episode)
            all_frame_index.append(new_frame)
            all_timestamp.append(new_timestamp)
            all_task_index.append(new_task)

        expected = int(info["total_episodes"]) * len(PARTS)
        if len(episode_rows) != expected:
            raise ValueError(f"expected {expected} new episodes, built {len(episode_rows)}")
        episode_rows.sort(key=lambda row: int(row["episode_index"]))

        # Store episode metadata in two bounded files (1000 rows each).
        for file_idx, start in enumerate(range(0, len(episode_rows), 1000)):
            rows = episode_rows[start : start + 1000]
            for row in rows:
                row["meta/episodes/chunk_index"] = 0
                row["meta/episodes/file_index"] = file_idx
            path = temp / f"meta/episodes/chunk-000/file-{file_idx:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy", use_dictionary=True)

        # LeRobot stores language strings in the pandas index. Using pandas
        # here preserves the parquet index metadata that load_tasks relies on.
        task_frame = pd.DataFrame(
            {"task_index": range(len(PARTS))},
            index=pd.Index([part[2] for part in PARTS], name="task"),
        )
        task_frame.to_parquet(temp / "meta/tasks.parquet")

        new_info = dict(info)
        new_info.update(total_episodes=expected, total_tasks=len(PARTS), splits={"train": f"0:{expected}"})
        (temp / "meta/info.json").write_text(json.dumps(new_info, indent=4) + "\n")

        stats = json.loads((source / "meta/stats.json").read_text())
        for key, chunks in {
            "episode_index": all_episode_index,
            "frame_index": all_frame_index,
            "timestamp": all_timestamp,
            "task_index": all_task_index,
        }.items():
            stats[key] = _scalar_stats(np.concatenate(chunks))
        (temp / "meta/stats.json").write_text(json.dumps(stats) + "\n")

        for filename in (".gitattributes",):
            if (source / filename).exists():
                shutil.copy2(source / filename, temp / filename)
        readme = (source / "README.md").read_text()
        derived_note = (
            "\n## Derived per-part episodes\n\n"
            "This local derivative splits each full assembly into nine logical episodes "
            "and assigns a part-specific language instruction. Camera MP4s are physically "
            "cut at the same frame boundaries as observation/action data. Generated by "
            "`split_lerobot_subtasks.py`.\n"
        )
        (temp / "README.md").write_text(readme + derived_note)

        temp.rename(destination)
        return {
            "episodes": expected,
            "frames": int(info["total_frames"]),
            "tasks": len(PARTS),
            "max_pick_anchor_error_m": max(max_errors, default=0.0),
            "destination": str(destination),
        }
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--min-segment-frames", type=int, default=20)
    parser.add_argument("--pick-tolerance-m", type=float, default=0.04)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable used to encode clips")
    args = parser.parse_args()
    result = split_dataset(
        args.source,
        args.destination,
        args.min_segment_frames,
        args.pick_tolerance_m,
        args.ffmpeg,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
