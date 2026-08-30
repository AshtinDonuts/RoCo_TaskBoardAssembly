#!/usr/bin/env python3
"""Split the RoCo full-assembly LeRobot v3 dataset into per-part episodes.

Observation/action rows and every camera video are cut at the same boundaries.
New collections provide exact boundaries and grading outcomes in a rollout
manifest; published legacy datasets fall back to action-anchor inference. Each
output episode has episode-local frame/video timestamps and can therefore be
loaded normally by LeRobot without referring to a time window inside the
original full-assembly MP4.
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
PART_BY_NAME = {name: (xy, task) for name, xy, task in PARTS}


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


def _feature_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    """LeRobot-style statistics for a scalar or fixed-width numeric feature."""
    x = np.asarray(values)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"expected nonempty [frames, dimensions] values, got {x.shape}")
    return {
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "mean": x.mean(axis=0).astype(np.float64).tolist(),
        "std": x.std(axis=0).astype(np.float64).tolist(),
        "count": [int(x.shape[0])],
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q10": np.quantile(x, 0.10, axis=0).tolist(),
        "q50": np.quantile(x, 0.50, axis=0).tolist(),
        "q90": np.quantile(x, 0.90, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def _load_rollout_manifest(path: Path, total_episodes: int) -> dict[int, dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid rollout manifest JSON at {path}:{lineno}: {exc}") from exc
    by_episode: dict[int, dict] = {}
    seeds: set[int] = set()
    for row in rows:
        episode = int(row["episode_index"])
        seed = int(row["seed"])
        if episode in by_episode:
            raise ValueError(f"duplicate manifest episode_index {episode}")
        if seed in seeds:
            raise ValueError(f"duplicate manifest seed {seed}")
        by_episode[episode] = row
        seeds.add(seed)
    expected = set(range(total_episodes))
    if set(by_episode) != expected:
        raise ValueError(
            "rollout manifest episode set does not match source dataset: "
            f"missing={sorted(expected - set(by_episode))} "
            f"extra={sorted(set(by_episode) - expected)}"
        )
    return by_episode


def _manifest_segments(row: dict, episode_length: int) -> list[dict]:
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"episode {row.get('episode_index')} has no segments")
    seen = set()
    previous_end = 0
    normalized = []
    for segment in segments:
        name = str(segment.get("name"))
        if name not in PART_BY_NAME:
            raise ValueError(f"unknown manifest part {name!r}")
        if name in seen:
            raise ValueError(f"duplicate manifest part {name!r}")
        begin, end = int(segment["begin"]), int(segment["end"])
        if begin != previous_end or end <= begin or end > episode_length:
            raise ValueError(
                f"invalid segment for episode {row.get('episode_index')}, {name}: "
                f"[{begin}, {end}) after {previous_end}, length={episode_length}"
            )
        if "pass" not in segment:
            raise ValueError(f"segment {name!r} has no grading outcome")
        normalized.append({**segment, "name": name, "begin": begin, "end": end, "pass": bool(segment["pass"])})
        seen.add(name)
        previous_end = end
    if previous_end != episode_length:
        raise ValueError(
            f"episode {row.get('episode_index')} segments end at {previous_end}, expected {episode_length}"
        )
    if seen != set(PART_BY_NAME):
        raise ValueError(f"episode {row.get('episode_index')} does not contain all nine parts")
    return normalized


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
    parent_length: int,
    selected_segments: list[dict],
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
        if video_frames != parent_length:
            raise ValueError(
                f"episode {parent_idx} video {key} has {video_frames} metadata frames, "
                f"but its observation/action table has {parent_length} rows"
            )

        filters: list[str] = []
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_video)]
        for output_idx, segment in enumerate(selected_segments):
            begin, end = segment["begin"], segment["end"]
            filters.append(
                f"[0:v]trim=start_frame={base_frame + begin}:end_frame={base_frame + end},"
                f"setpts=PTS-STARTPTS[v{output_idx}]"
            )

        command.extend(["-filter_complex", ";".join(filters)])
        for output_idx, segment in enumerate(selected_segments):
            episode_idx = int(segment["episode_index"])
            chunk_index, file_index = divmod(episode_idx, chunks_size)
            output_video = _video_path(info, destination, key, chunk_index, file_index)
            output_video.parent.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    "-map",
                    f"[v{output_idx}]",
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
    rollout_manifest: Path | None = None,
    successful_parts_only: bool = False,
    replace: bool = False,
) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() and not replace:
        raise FileExistsError(f"destination already exists: {destination}")

    info = json.loads((source / "meta/info.json").read_text())
    fps = int(info["fps"])
    total_parents = int(info["total_episodes"])
    if total_parents <= 0:
        raise ValueError("source dataset has no episodes")

    parent_meta_table = _load_nested_parquet(source / "meta/episodes")
    parent_meta = {int(row["episode_index"]): row for row in parent_meta_table.to_pylist()}
    if set(parent_meta) != set(range(total_parents)):
        raise ValueError("source episode metadata is incomplete or non-dense")
    video_keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]

    if rollout_manifest is None:
        default_manifest = source / "meta/roco_rollouts.jsonl"
        rollout_manifest = default_manifest if default_manifest.is_file() else None
    elif not rollout_manifest.is_absolute():
        rollout_manifest = (Path.cwd() / rollout_manifest).resolve()
    manifest = (
        _load_rollout_manifest(rollout_manifest, total_parents)
        if rollout_manifest is not None else None
    )
    if successful_parts_only and manifest is None:
        raise ValueError("--successful-parts-only requires a rollout manifest with grading outcomes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    episode_rows: list[dict] = []
    lineage_rows: list[dict] = []
    numeric_features = {
        key for key, spec in info["features"].items()
        if spec.get("dtype") not in {"image", "video", "string"}
    }
    all_numeric: dict[str, list[np.ndarray]] = {key: [] for key in numeric_features}
    max_errors: list[float] = []
    next_episode_index = 0
    next_dataset_index = 0

    try:
        for data_path in sorted((source / "data").rglob("*.parquet")):
            table = pq.read_table(data_path)
            old_episode = np.asarray(table["episode_index"], dtype=np.int64)
            actions = (
                np.asarray(table["action"].to_pylist(), dtype=np.float64)
                if "action" in table.column_names else None
            )
            output_tables: list[pa.Table] = []

            for parent_idx in np.unique(old_episode):
                rows = np.flatnonzero(old_episode == parent_idx)
                if not np.all(np.diff(rows) == 1):
                    raise ValueError(f"episode {parent_idx} is not contiguous in {data_path}")
                if int(parent_meta[int(parent_idx)]["length"]) != len(rows):
                    raise ValueError(
                        f"episode {parent_idx} spans data files or has inconsistent length"
                    )

                if manifest is not None:
                    segments = _manifest_segments(manifest[int(parent_idx)], len(rows))
                else:
                    if actions is None:
                        raise ValueError("legacy boundary inference requires an action column")
                    starts, errors = _segment_starts(actions[rows], min_segment, tolerance)
                    max_errors.extend(errors[1:])
                    bounds = starts + [len(rows)]
                    segments = [
                        {
                            "name": PARTS[part_idx][0],
                            "begin": begin,
                            "end": end,
                            "pass": True,
                            "completion_reason": "legacy_inferred",
                        }
                        for part_idx, (begin, end) in enumerate(
                            zip(bounds[:-1], bounds[1:], strict=True)
                        )
                    ]

                selected = [
                    dict(segment, episode_index=next_episode_index + selected_idx)
                    for selected_idx, segment in enumerate(
                        segment for segment in segments
                        if not successful_parts_only or segment["pass"]
                    )
                ]
                parent = parent_meta[int(parent_idx)]

                if selected:
                    _splice_episode_videos(
                        source,
                        temp,
                        info,
                        parent,
                        int(parent_idx),
                        len(rows),
                        selected,
                        video_keys,
                        fps,
                        ffmpeg,
                    )

                for segment in selected:
                    begin, end = segment["begin"], segment["end"]
                    episode_idx = int(segment["episode_index"])
                    name = segment["name"]
                    part_idx = list(PART_BY_NAME).index(name)
                    length = end - begin
                    frame = np.arange(length, dtype=np.int64)
                    timestamp = frame.astype(np.float32) / fps
                    index = np.arange(
                        next_dataset_index, next_dataset_index + length, dtype=np.int64
                    )
                    seg_table = table.slice(int(rows[0] + begin), length)
                    for key, values in {
                        "episode_index": np.full(length, episode_idx, dtype=np.int64),
                        "frame_index": frame,
                        "timestamp": timestamp,
                        "task_index": np.full(length, part_idx, dtype=np.int64),
                        "index": index,
                    }.items():
                        seg_table = _replace(seg_table, key, values)
                    output_tables.append(seg_table)

                    meta = dict(parent)
                    meta.update(
                        episode_index=episode_idx,
                        tasks=[PART_BY_NAME[name][1]],
                        length=length,
                        dataset_from_index=next_dataset_index,
                        dataset_to_index=next_dataset_index + length,
                    )
                    for key in video_keys:
                        chunk_index, file_index = divmod(episode_idx, int(info["chunks_size"]))
                        meta[f"videos/{key}/chunk_index"] = chunk_index
                        meta[f"videos/{key}/file_index"] = file_index
                        meta[f"videos/{key}/from_timestamp"] = 0.0
                        meta[f"videos/{key}/to_timestamp"] = length / fps

                    for key in numeric_features:
                        if key not in seg_table.column_names:
                            continue
                        vals = np.asarray(seg_table[key].to_pylist())
                        for stat, value in _feature_stats(vals).items():
                            meta[f"stats/{key}/{stat}"] = value
                        all_numeric[key].append(vals)
                    episode_rows.append(meta)
                    lineage_rows.append({
                        "episode_index": episode_idx,
                        "source_episode_index": int(parent_idx),
                        "seed": (None if manifest is None else int(manifest[int(parent_idx)]["seed"])),
                        "part": name,
                        "source_begin": begin,
                        "source_end": end,
                        "pass": bool(segment["pass"]),
                        "completion_reason": segment.get("completion_reason"),
                    })
                    next_dataset_index += length
                next_episode_index += len(selected)

            if output_tables:
                out_path = temp / data_path.relative_to(source)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    pa.concat_tables(output_tables, promote_options="default"),
                    out_path,
                    compression="snappy",
                    use_dictionary=True,
                )

        if not episode_rows:
            raise ValueError("no subtask episodes were selected")
        episode_rows.sort(key=lambda row: int(row["episode_index"]))

        for file_idx, start in enumerate(range(0, len(episode_rows), 1000)):
            rows = episode_rows[start : start + 1000]
            for row in rows:
                row["meta/episodes/chunk_index"] = 0
                row["meta/episodes/file_index"] = file_idx
            path = temp / f"meta/episodes/chunk-000/file-{file_idx:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy", use_dictionary=True)

        task_frame = pd.DataFrame(
            {"task_index": range(len(PARTS))},
            index=pd.Index([part[2] for part in PARTS], name="task"),
        )
        task_frame.to_parquet(temp / "meta/tasks.parquet")

        new_info = dict(info)
        new_info.update(
            total_episodes=len(episode_rows),
            total_frames=next_dataset_index,
            total_tasks=len(PARTS),
            splits={"train": f"0:{len(episode_rows)}"},
        )
        (temp / "meta/info.json").write_text(json.dumps(new_info, indent=4) + "\n")

        stats = json.loads((source / "meta/stats.json").read_text())
        for key, chunks in all_numeric.items():
            if chunks:
                stats[key] = _feature_stats(np.concatenate(chunks, axis=0))
        (temp / "meta/stats.json").write_text(json.dumps(stats) + "\n")
        (temp / "meta/roco_subtasks.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in lineage_rows),
            encoding="utf-8",
        )

        for filename in (".gitattributes",):
            if (source / filename).exists():
                shutil.copy2(source / filename, temp / filename)
        readme_path = source / "README.md"
        readme = readme_path.read_text() if readme_path.exists() else "# RoCo LeRobot dataset\n"
        derived_note = (
            "\n## Derived per-part episodes\n\n"
            "This local derivative splits full assemblies into logical part episodes "
            "and assigns part-specific language instructions. When success filtering is enabled, "
            "failed parts are omitted. Camera MP4s are physically "
            "cut at the same frame boundaries as observation/action data. Generated by "
            "`split_lerobot_subtasks.py`.\n"
        )
        (temp / "README.md").write_text(readme + derived_note)

        backup = None
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            destination.rename(backup)
        try:
            temp.rename(destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return {
            "source_episodes": total_parents,
            "episodes": len(episode_rows),
            "frames": next_dataset_index,
            "tasks": len(PARTS),
            "max_pick_anchor_error_m": max(max_errors, default=0.0),
            "successful_parts_only": successful_parts_only,
            "manifest": None if rollout_manifest is None else str(rollout_manifest),
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
    parser.add_argument(
        "--rollout-manifest",
        type=Path,
        default=None,
        help="JSONL rollout metadata; defaults to SOURCE/meta/roco_rollouts.jsonl when present",
    )
    parser.add_argument(
        "--successful-parts-only",
        action="store_true",
        help="Keep only segments whose manifest grading outcome passed",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing destination after a successful split",
    )
    args = parser.parse_args()
    result = split_dataset(
        args.source,
        args.destination,
        args.min_segment_frames,
        args.pick_tolerance_m,
        args.ffmpeg,
        args.rollout_manifest,
        args.successful_parts_only,
        args.replace,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
