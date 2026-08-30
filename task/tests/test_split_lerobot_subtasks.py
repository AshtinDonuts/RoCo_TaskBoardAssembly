import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import split_lerobot_subtasks as splitter


class SplitLeRobotSubtasksTest(unittest.TestCase):
    def test_manifest_validation_rejects_gaps(self):
        row = {
            "episode_index": 0,
            "segments": [
                {"name": name, "begin": i + (1 if i else 0), "end": i + 1, "pass": True}
                for i, (name, _, _) in enumerate(splitter.PARTS)
            ],
        }
        with self.assertRaisesRegex(ValueError, "invalid segment"):
            splitter._manifest_segments(row, 9)

    def test_feature_stats_preserve_vector_dimensions(self):
        stats = splitter._feature_stats(np.array([[1, 2], [3, 4]], dtype=np.float32))
        self.assertEqual(stats["min"], [1.0, 2.0])
        self.assertEqual(stats["max"], [3.0, 4.0])
        self.assertEqual(stats["count"], [2])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_successful_manifest_split_is_dense_and_frame_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "derived"
            (source / "meta/episodes/chunk-000").mkdir(parents=True)
            (source / "data/chunk-000").mkdir(parents=True)
            video_dir = source / "videos/observation.images.head/chunk-000"
            video_dir.mkdir(parents=True)

            video = video_dir / "file-000.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=blue:s=32x24:r=10:d=1.8",
                    "-frames:v", "18", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
            )

            features = {
                "action": {"dtype": "float32", "shape": [14], "names": [str(i) for i in range(14)]},
                "observation.state": {"dtype": "float32", "shape": [44], "names": [str(i) for i in range(44)]},
                "observation.images.head": {"dtype": "video", "shape": [24, 32, 3], "names": ["height", "width", "rgb"]},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
            info = {
                "codebase_version": "v3.0", "robot_type": "vega_1u", "fps": 10,
                "total_episodes": 2, "total_frames": 18, "total_tasks": 1,
                "chunks_size": 1000, "splits": {"train": "0:2"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": features,
            }
            (source / "meta/info.json").write_text(json.dumps(info))
            (source / "meta/stats.json").write_text(json.dumps({}))
            pd.DataFrame({"task_index": [0]}, index=pd.Index(["assemble"], name="task")).to_parquet(
                source / "meta/tasks.parquet"
            )

            episode_meta = []
            for episode in range(2):
                episode_meta.append({
                    "episode_index": episode, "tasks": ["assemble"], "length": 9,
                    "dataset_from_index": episode * 9, "dataset_to_index": (episode + 1) * 9,
                    "data/chunk_index": 0, "data/file_index": 0,
                    "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                    "videos/observation.images.head/chunk_index": 0,
                    "videos/observation.images.head/file_index": 0,
                    "videos/observation.images.head/from_timestamp": episode * 0.9,
                    "videos/observation.images.head/to_timestamp": (episode + 1) * 0.9,
                })
            pq.write_table(pa.Table.from_pylist(episode_meta), source / "meta/episodes/chunk-000/file-000.parquet")

            episode_index = np.repeat(np.arange(2, dtype=np.int64), 9)
            frame_index = np.tile(np.arange(9, dtype=np.int64), 2)
            table = pa.table({
                "action": pa.array(np.arange(18 * 14, dtype=np.float32).reshape(18, 14).tolist(), type=pa.list_(pa.float32(), 14)),
                "observation.state": pa.array(np.arange(18 * 44, dtype=np.float32).reshape(18, 44).tolist(), type=pa.list_(pa.float32(), 44)),
                "timestamp": pa.array(frame_index.astype(np.float32) / 10),
                "frame_index": pa.array(frame_index),
                "episode_index": pa.array(episode_index),
                "index": pa.array(np.arange(18, dtype=np.int64)),
                "task_index": pa.array(np.zeros(18, dtype=np.int64)),
            })
            pq.write_table(table, source / "data/chunk-000/file-000.parquet")

            manifest = source / "meta/roco_rollouts.jsonl"
            manifest.write_text("".join(
                json.dumps({
                    "episode_index": episode, "seed": episode,
                    "segments": [
                        {"name": name, "begin": i, "end": i + 1, "pass": i % 2 == 0, "completion_reason": "policy_done"}
                        for i, (name, _, _) in enumerate(splitter.PARTS)
                    ],
                }) + "\n"
                for episode in range(2)
            ))

            result = splitter.split_dataset(
                source, destination, 1, 0.04,
                rollout_manifest=manifest, successful_parts_only=True,
            )
            self.assertEqual(result["episodes"], 10)
            self.assertEqual(result["frames"], 10)
            out = pq.read_table(destination / "data/chunk-000/file-000.parquet")
            self.assertEqual(out["index"].to_pylist(), list(range(10)))
            self.assertEqual(out["episode_index"].to_pylist(), list(range(10)))
            self.assertEqual(json.loads((destination / "meta/stats.json").read_text())["action"]["count"], [10])
            self.assertEqual(len((destination / "meta/roco_subtasks.jsonl").read_text().splitlines()), 10)
            for episode in range(10):
                clip = destination / f"videos/observation.images.head/chunk-000/file-{episode:03d}.mp4"
                frames = subprocess.check_output([
                    "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(clip),
                ], text=True).strip()
                self.assertEqual(frames, "1")


if __name__ == "__main__":
    unittest.main()
