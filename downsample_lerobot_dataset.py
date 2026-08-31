#!/usr/bin/env python3
"""Create a frame-exact lower-FPS derivative of a LeRobot v3 dataset.

Rows and camera frames are decimated on the same episode-local grid.  The
output is a self-contained dataset: frame/index columns, episode metadata,
timestamps, video FPS metadata, statistics, and RoCo lineage are rewritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from split_lerobot_subtasks import _feature_stats, _load_nested_parquet, _replace


def _video_path(info: dict, root: Path, key: str, chunk: int, file: int) -> Path:
    return root / info["video_path"].format(
        video_key=key, chunk_index=chunk, file_index=file
    )


def _transcode_video(
    source: Path,
    destination: Path,
    factor: int,
    output_fps: int,
    expected_frames: int,
    ffmpeg: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"select=not(mod(n\\,{factor})),setpts=N/({output_fps}*TB)",
        "-frames:v",
        str(expected_frames),
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
        str(output_fps),
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while downsampling {source}") from exc


def downsample_dataset(
    source: Path,
    destination: Path,
    output_fps: int,
    parts: set[str] | None = None,
    ffmpeg: str = "ffmpeg",
    replace: bool = False,
) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() and not replace:
        raise FileExistsError(f"destination already exists: {destination}")
    if output_fps <= 0:
        raise ValueError("output FPS must be positive")

    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    input_fps = int(info["fps"])
    if input_fps % output_fps:
        raise ValueError(
            f"input FPS {input_fps} must be an integer multiple of output FPS {output_fps}"
        )
    factor = input_fps // output_fps
    if factor <= 1:
        raise ValueError("output FPS must be lower than input FPS")

    lineage_path = source / "meta/roco_subtasks.jsonl"
    if not lineage_path.is_file():
        raise FileNotFoundError(f"missing RoCo subtask manifest: {lineage_path}")
    source_lineage = [
        json.loads(line)
        for line in lineage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lineage_by_episode = {int(row["episode_index"]): row for row in source_lineage}
    if len(lineage_by_episode) != len(source_lineage):
        raise ValueError("duplicate episode_index in RoCo subtask manifest")

    source_episode_table = _load_nested_parquet(source / "meta/episodes")
    source_episodes = {
        int(row["episode_index"]): row for row in source_episode_table.to_pylist()
    }
    if set(source_episodes) != set(lineage_by_episode):
        raise ValueError("episode metadata and RoCo subtask manifest do not match")
    selected_old = [
        episode
        for episode in sorted(source_episodes)
        if parts is None or str(lineage_by_episode[episode].get("part")) in parts
    ]
    if not selected_old:
        raise ValueError(f"no episodes selected for parts={sorted(parts) if parts else None}")
    if parts is not None:
        found = {str(lineage_by_episode[index].get("part")) for index in selected_old}
        if found != parts:
            raise ValueError(f"requested parts not found: {sorted(parts - found)}")

    old_to_new = {old: new for new, old in enumerate(selected_old)}
    video_keys = [
        key for key, spec in info["features"].items() if spec.get("dtype") == "video"
    ]
    numeric_features = {
        key
        for key, spec in info["features"].items()
        if spec.get("dtype") not in {"image", "video", "string"}
    }
    all_numeric: dict[str, list[np.ndarray]] = {key: [] for key in numeric_features}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    output_episode_rows: list[dict] = []
    output_lineage: list[dict] = []
    next_index = 0

    try:
        tables = [_load_nested_parquet(source / "data")]
        full_table = pa.concat_tables(tables, promote_options="default")
        episode_column = np.asarray(full_table["episode_index"], dtype=np.int64)

        for old_episode in selected_old:
            row_indices = np.flatnonzero(episode_column == old_episode)
            if not len(row_indices) or not np.all(np.diff(row_indices) == 1):
                raise ValueError(f"episode {old_episode} is missing or non-contiguous")
            source_meta = source_episodes[old_episode]
            if len(row_indices) != int(source_meta["length"]):
                raise ValueError(f"episode {old_episode} length metadata is inconsistent")

            local = np.arange(0, len(row_indices), factor, dtype=np.int64)
            selected_rows = row_indices[local]
            table = full_table.take(pa.array(selected_rows))
            new_episode = old_to_new[old_episode]
            length = len(local)
            frame = np.arange(length, dtype=np.int64)
            for key, values in {
                "episode_index": np.full(length, new_episode, dtype=np.int64),
                "frame_index": frame,
                "timestamp": frame.astype(np.float32) / output_fps,
                "index": np.arange(next_index, next_index + length, dtype=np.int64),
            }.items():
                table = _replace(table, key, values)
            chunk, file = divmod(new_episode, int(info["chunks_size"]))

            meta = dict(source_meta)
            meta.update(
                episode_index=new_episode,
                length=length,
                dataset_from_index=next_index,
                dataset_to_index=next_index + length,
                **{
                    "data/chunk_index": chunk,
                    "data/file_index": file,
                    "meta/episodes/chunk_index": chunk,
                    "meta/episodes/file_index": file,
                },
            )
            for key in numeric_features:
                if key not in table.column_names:
                    continue
                values = np.asarray(table[key].to_pylist())
                for stat, value in _feature_stats(values).items():
                    meta[f"stats/{key}/{stat}"] = value
                all_numeric[key].append(values)

            for key in video_keys:
                source_video = _video_path(
                    info,
                    source,
                    key,
                    int(source_meta[f"videos/{key}/chunk_index"]),
                    int(source_meta[f"videos/{key}/file_index"]),
                )
                output_video = _video_path(info, temp, key, chunk, file)
                _transcode_video(
                    source_video, output_video, factor, output_fps, length, ffmpeg
                )
                meta[f"videos/{key}/chunk_index"] = chunk
                meta[f"videos/{key}/file_index"] = file
                meta[f"videos/{key}/from_timestamp"] = 0.0
                meta[f"videos/{key}/to_timestamp"] = length / output_fps

            output_episode_rows.append(meta)
            data_path = temp / info["data_path"].format(
                chunk_index=chunk, file_index=file
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                data_path,
                compression="snappy",
                use_dictionary=True,
            )
            episode_path = (
                temp
                / f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet"
            )
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist([meta]),
                episode_path,
                compression="snappy",
                use_dictionary=True,
            )
            lineage = dict(lineage_by_episode[old_episode])
            lineage.update(
                episode_index=new_episode,
                downsample_source_episode_index=old_episode,
                downsample_source_fps=input_fps,
                downsample_factor=factor,
                fps=output_fps,
            )
            output_lineage.append(lineage)
            next_index += length

        new_info = dict(info)
        new_info.update(
            fps=output_fps,
            total_episodes=len(output_episode_rows),
            total_frames=next_index,
            splits={"train": f"0:{len(output_episode_rows)}"},
        )
        for spec in new_info["features"].values():
            if spec.get("dtype") == "video" and "info" in spec:
                spec["info"]["video.fps"] = output_fps
        (temp / "meta/info.json").write_text(
            json.dumps(new_info, indent=4) + "\n", encoding="utf-8"
        )
        stats = json.loads((source / "meta/stats.json").read_text(encoding="utf-8"))
        for key, chunks in all_numeric.items():
            if chunks:
                stats[key] = _feature_stats(np.concatenate(chunks, axis=0))
        (temp / "meta/stats.json").write_text(
            json.dumps(stats) + "\n", encoding="utf-8"
        )
        shutil.copy2(source / "meta/tasks.parquet", temp / "meta/tasks.parquet")
        (temp / "meta/roco_subtasks.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_lineage),
            encoding="utf-8",
        )
        for filename in ("README.md", ".gitattributes"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, temp / filename)

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
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    return {
        "source": str(source),
        "destination": str(destination),
        "source_fps": input_fps,
        "fps": output_fps,
        "factor": factor,
        "parts": sorted(parts) if parts else "all",
        "episodes": len(output_episode_rows),
        "frames": next_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fps", type=int, required=True, help="output dataset FPS")
    parser.add_argument(
        "--part",
        action="append",
        dest="parts",
        help="only retain this RoCo part; repeat to select multiple (default: all)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    result = downsample_dataset(
        args.source,
        args.destination,
        args.fps,
        set(args.parts) if args.parts else None,
        args.ffmpeg,
        args.replace,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
