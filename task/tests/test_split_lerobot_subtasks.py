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

    @staticmethod
    def _states(length=12):
        return np.zeros((length, 44), dtype=np.float64)

    @staticmethod
    def _segment(length=12, name="gear_20teeth", begin=10):
        # Non-zero begin: first-episode segments (begin==0) skip home pruning.
        return {"name": name, "begin": begin, "end": begin + length, "pass": True}

    def _states_for_segment(self, segment, length=None):
        length = int(segment["end"] - segment["begin"] if length is None else length)
        total = int(segment["end"])
        return self._states(total)

    def test_home_pruning_uses_first_stable_window(self):
        segment = self._segment()
        states = self._states_for_segment(segment)
        states[segment["begin"]:segment["end"], 14:21] = splitter.DEFAULT_LEFT_HOME_Q
        states[segment["begin"]:segment["begin"] + 3, 14:21] += 1.0
        refined = splitter._refine_segment(
            segment, {"left_arm_home_q": splitter.DEFAULT_LEFT_HOME_Q.tolist()},
            states, "home", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], segment["begin"] + 3)
        self.assertEqual(refined["frames_removed"], 3)

    def test_home_pruning_uses_35ec027_fallback_and_rejects_no_home(self):
        segment = self._segment()
        states = self._states_for_segment(segment)
        states[segment["begin"]:segment["end"], 14:21] = splitter.DEFAULT_LEFT_HOME_Q
        refined = splitter._refine_segment(
            segment, {}, states, "home", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["pruning_evidence"]["home_source"], "default_35ec027")
        states[segment["begin"]:segment["end"], 14:21] += 1.0
        with self.assertRaisesRegex(ValueError, "never reaches a stable home pose"):
            splitter._refine_segment(
                segment, {}, states, "home", 2, 0.03, 3, 0.5,
            )

    def test_home_pruning_skips_first_episode_segment(self):
        # gear_60teeth is the first sim asset and has no return_home prefix.
        segment = {"name": "gear_60teeth", "begin": 0, "end": 12, "pass": True}
        states = self._states(12)
        states[:, 14:21] += 1.0  # never near home
        refined = splitter._refine_segment(
            segment, {}, states, "home", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], 0)
        self.assertEqual(refined["frames_removed"], 0)
        self.assertTrue(refined["pruning_evidence"]["skipped"])
        self.assertEqual(
            refined["pruning_evidence"]["reason"],
            "first_part_has_no_home_prefix",
        )

    def test_velocity_pruning_only_removes_startup_prefix(self):
        segment = self._segment(length=30, begin=10)
        states = self._states(40)
        local = slice(segment["begin"], segment["end"])
        states[local, 28:35] = 0.1
        states[segment["begin"]:segment["begin"] + 3, 28:35] = 2.0
        states[segment["begin"] + 15:segment["end"], 28:35] = 0.1
        states[segment["begin"] + 15:segment["end"], 14:21] = (
            np.arange(1, 16, dtype=np.float64)[:, None] * 0.01
        )
        states[segment["begin"] + 24, 28:35] = 3.0
        refined = splitter._refine_segment(
            segment, None, states, "velocity", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], segment["begin"] + 15)
        self.assertEqual(refined["end"], segment["end"])
        self.assertEqual(refined["pruning_evidence"]["freeze_begin"], segment["begin"] + 3)
        self.assertEqual(refined["pruning_evidence"]["freeze_end"], segment["begin"] + 15)

    def test_velocity_pruning_leaves_segment_without_immediate_reset_unchanged(self):
        segment = self._segment(length=30, begin=10)
        states = self._states(40)
        local = slice(segment["begin"], segment["end"])
        states[local, 28:35] = 0.1
        states[local, 14:21] = np.arange(30, dtype=np.float64)[:, None] * 0.01
        states[segment["begin"] + 20, 28:35] = 2.0
        refined = splitter._refine_segment(
            segment, None, states, "velocity", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], segment["begin"])
        self.assertFalse(refined["pruning_evidence"]["reset_detected"])

    def test_velocity_pruning_rejects_reset_without_freeze(self):
        # Missing freeze no longer aborts the split: keep the full segment.
        segment = self._segment(length=30, begin=10)
        states = self._states(40)
        local = slice(segment["begin"], segment["end"])
        states[local, 28:35] = 0.1
        states[segment["begin"]:segment["begin"] + 3, 28:35] = 2.0
        states[local, 14:21] = np.arange(30, dtype=np.float64)[:, None] * 0.01
        refined = splitter._refine_segment(
            segment, None, states, "velocity", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], segment["begin"])
        self.assertEqual(refined["frames_removed"], 0)
        self.assertTrue(refined["pruning_evidence"]["reset_detected"])
        self.assertFalse(refined["pruning_evidence"]["freeze_found"])
        self.assertEqual(
            refined["pruning_evidence"]["reason"],
            "reset_without_required_freeze",
        )

    def test_velocity_pruning_skips_first_episode_segment(self):
        segment = {"name": "gear_60teeth", "begin": 0, "end": 30, "pass": True}
        states = self._states(30)
        states[:3, 28:35] = 2.0  # would look like reset without a freeze
        states[:, 14:21] = np.arange(30, dtype=np.float64)[:, None] * 0.01
        refined = splitter._refine_segment(
            segment, None, states, "velocity", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], 0)
        self.assertEqual(refined["frames_removed"], 0)
        self.assertTrue(refined["pruning_evidence"]["skipped"])
        self.assertEqual(
            refined["pruning_evidence"]["reason"],
            "first_part_has_no_reset_prefix",
        )

    def test_waypoint_pruning_removes_only_leading_transition_phases(self):
        segment = self._segment(begin=0)
        segment["phases"] = [
            {"name": "safe_retract", "waypoint_index": 0, "begin": 0, "end": 2},
            {"name": "return_home", "waypoint_index": 1, "begin": 2, "end": 4},
            {"name": "hover_pick", "waypoint_index": 2, "begin": 4, "end": 12},
        ]
        refined = splitter._refine_segment(
            segment, {"waypoint_annotations_complete": True}, self._states(),
            "waypoint", 2, 0.03, 3, 0.5,
        )
        self.assertEqual(refined["begin"], 4)
        self.assertEqual(
            refined["pruning_evidence"]["trimmed_phases"],
            ["safe_retract", "return_home"],
        )

    def test_waypoint_pruning_requires_complete_annotations(self):
        with self.assertRaisesRegex(ValueError, "complete waypoint annotations"):
            splitter._refine_segment(
                self._segment(begin=0), {"waypoint_annotations_complete": False},
                self._states(), "waypoint", 2, 0.03, 3, 0.5,
            )

    def test_manifest_rejects_phase_gap(self):
        segments = []
        for i, (name, _, _) in enumerate(splitter.PARTS):
            segment = {"name": name, "begin": i, "end": i + 1, "pass": True}
            segment["phases"] = [{
                "name": "hover_pick", "waypoint_index": 0,
                "begin": i + (1 if i == 0 else 0), "end": i + 1,
            }]
            segments.append(segment)
        with self.assertRaisesRegex(ValueError, "invalid waypoint phase"):
            splitter._manifest_segments({"episode_index": 0, "segments": segments}, 9)

    def test_manifest_rejects_claimed_complete_missing_phase_name(self):
        segments = []
        for i, (name, _, _) in enumerate(splitter.PARTS):
            segments.append({
                "name": name, "begin": i, "end": i + 1, "pass": True,
                "phases": [{
                    "name": None if i == 0 else "hover_pick",
                    "waypoint_index": 0, "begin": i, "end": i + 1,
                }],
            })
        with self.assertRaisesRegex(ValueError, "missing name or index"):
            splitter._manifest_segments({
                "episode_index": 0,
                "waypoint_annotations_complete": True,
                "segments": segments,
            }, 9)

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
                    "waypoint_annotations_complete": True,
                    "segments": [
                        {
                            "name": name, "begin": i, "end": i + 1,
                            "pass": i % 2 == 0, "completion_reason": "policy_done",
                            "phases": [{
                                "name": "hover_pick", "waypoint_index": 0,
                                "begin": i, "end": i + 1,
                            }],
                        }
                        for i, (name, _, _) in enumerate(splitter.PARTS)
                    ],
                }) + "\n"
                for episode in range(2)
            ))

            result = splitter.split_dataset(
                source, destination, 1, 0.04,
                rollout_manifest=manifest, successful_parts_only=True,
                pruning_strategy="waypoint",
            )
            self.assertEqual(result["episodes"], 10)
            self.assertEqual(result["frames"], 10)
            self.assertEqual(result["pruning_strategy"], "waypoint")
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
