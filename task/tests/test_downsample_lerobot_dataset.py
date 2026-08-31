import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import downsample_lerobot_dataset as downsampler


class DownsampleLeRobotDatasetTest(unittest.TestCase):
    def test_writes_one_data_and_metadata_file_per_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "downsampled"
            (source / "data/chunk-000").mkdir(parents=True)
            (source / "meta/episodes/chunk-000").mkdir(parents=True)

            info = {
                "fps": 30,
                "total_episodes": 2,
                "total_frames": 12,
                "chunks_size": 1,
                "splits": {"train": "0:2"},
                "data_path": (
                    "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                ),
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
                "features": {
                    "action": {"dtype": "float32", "shape": [1]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "index": {"dtype": "int64", "shape": [1]},
                },
            }
            (source / "meta/info.json").write_text(json.dumps(info))
            (source / "meta/stats.json").write_text(json.dumps({}))
            pq.write_table(pa.table({"task_index": [0]}), source / "meta/tasks.parquet")

            episode_rows = []
            for episode in range(2):
                episode_rows.append(
                    {
                        "episode_index": episode,
                        "length": 6,
                        "dataset_from_index": episode * 6,
                        "dataset_to_index": (episode + 1) * 6,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                        "meta/episodes/chunk_index": 0,
                        "meta/episodes/file_index": 0,
                    }
                )
            pq.write_table(
                pa.Table.from_pylist(episode_rows),
                source / "meta/episodes/chunk-000/file-000.parquet",
            )
            episode_index = np.repeat(np.arange(2, dtype=np.int64), 6)
            frame_index = np.tile(np.arange(6, dtype=np.int64), 2)
            pq.write_table(
                pa.table(
                    {
                        "action": pa.array(
                            np.arange(12, dtype=np.float32)[:, None].tolist(),
                            type=pa.list_(pa.float32(), 1),
                        ),
                        "timestamp": pa.array(frame_index.astype(np.float32) / 30),
                        "frame_index": pa.array(frame_index),
                        "episode_index": pa.array(episode_index),
                        "index": pa.array(np.arange(12, dtype=np.int64)),
                    }
                ),
                source / "data/chunk-000/file-000.parquet",
            )
            (source / "meta/roco_subtasks.jsonl").write_text(
                "".join(
                    json.dumps({"episode_index": episode, "part": "gear_60teeth"})
                    + "\n"
                    for episode in range(2)
                )
            )

            downsampler.downsample_dataset(source, destination, 10)

            data_files = sorted((destination / "data").glob("chunk-*/*.parquet"))
            meta_files = sorted(
                (destination / "meta/episodes").glob("chunk-*/*.parquet")
            )
            self.assertEqual(
                [path.relative_to(destination).as_posix() for path in data_files],
                [
                    "data/chunk-000/file-000.parquet",
                    "data/chunk-001/file-000.parquet",
                ],
            )
            self.assertEqual(
                [path.relative_to(destination).as_posix() for path in meta_files],
                [
                    "meta/episodes/chunk-000/file-000.parquet",
                    "meta/episodes/chunk-001/file-000.parquet",
                ],
            )
            for episode, (data_path, meta_path) in enumerate(
                zip(data_files, meta_files, strict=True)
            ):
                table = pq.read_table(data_path)
                self.assertEqual(table.num_rows, 2)
                self.assertEqual(table["episode_index"].to_pylist(), [episode] * 2)
                meta = pq.read_table(meta_path).to_pylist()
                self.assertEqual(len(meta), 1)
                self.assertEqual(meta[0]["data/chunk_index"], episode)
                self.assertEqual(meta[0]["data/file_index"], 0)
                self.assertEqual(meta[0]["meta/episodes/chunk_index"], episode)
                self.assertEqual(meta[0]["meta/episodes/file_index"], 0)


if __name__ == "__main__":
    unittest.main()
