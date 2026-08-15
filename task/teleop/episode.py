"""Wall-clock episode session for human teleop recording.

Task success is not modeled here. The session only decides warmup, recording,
save, rerecord, timeout, and stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EpisodeEvent:
    kind: str
    reason: str = ""
    end_session: bool = False
    episode_index: int = 0
    attempt_index: int = 0
    saved_episodes: int = 0

    @property
    def is_noop(self) -> bool:
        return self.kind == "none"


class EpisodeSession:
    """LeRobot-style record / rerecord / stop with a warmup before each try.

    Phases: idle -> warmup -> recording -> idle (reset) or done.
    """

    def __init__(
        self,
        episode_time_s: float = 600.0,
        warmup_time_s: float = 5.0,
        num_episodes: int = 1,
    ) -> None:
        if episode_time_s <= 0:
            raise ValueError("episode_time_s must be positive")
        if warmup_time_s < 0:
            raise ValueError("warmup_time_s must be >= 0")
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        self.episode_time_s = float(episode_time_s)
        self.warmup_time_s = float(warmup_time_s)
        self.num_episodes = int(num_episodes)
        self.phase = "idle"
        self.saved_episodes = 0
        self.attempt_index = 0
        self.needs_reset = False
        self._warmup_t0: Optional[float] = None
        self._record_t0: Optional[float] = None

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def is_warmup(self) -> bool:
        return self.phase in ("idle", "warmup") and not self.done

    @property
    def is_recording(self) -> bool:
        return self.phase == "recording"

    def remaining_warmup_s(self, now: float) -> float:
        if self.phase != "warmup" or self._warmup_t0 is None:
            return self.warmup_time_s if self.phase == "idle" else 0.0
        return max(0.0, self.warmup_time_s - (now - self._warmup_t0))

    def remaining_episode_s(self, now: float) -> float:
        if self.phase != "recording" or self._record_t0 is None:
            return 0.0
        return max(0.0, self.episode_time_s - (now - self._record_t0))

    def elapsed_episode_s(self, now: float) -> float:
        if self._record_t0 is None:
            return 0.0
        return max(0.0, now - self._record_t0)

    def start(self, now: float) -> EpisodeEvent:
        if self.done:
            return self._event("none")
        self.attempt_index += 1
        self.phase = "warmup"
        self.needs_reset = False
        self._warmup_t0 = now
        self._record_t0 = None
        if self.warmup_time_s <= 0.0:
            return self._enter_recording(now, "warmup_skipped")
        return self._event("warmup_start", "start")

    def step(self, cmd: str, now: float) -> EpisodeEvent:
        ev = self.handle_cmd(cmd, now)
        if not ev.is_noop:
            return ev
        return self.tick(now)

    def handle_cmd(self, cmd: str, now: float) -> EpisodeEvent:
        if cmd in (None, "none", ""):
            return self._event("none")
        if self.done:
            return self._event("none")
        if self.phase == "warmup":
            if cmd == "rerecord_episode":
                return self.start(now)
            if cmd == "stop_recording":
                self.phase = "done"
                self.needs_reset = False
                return self._event("session_end", "stop_recording", end_session=True)
            if cmd == "save_episode":
                return self._event("none")
            return self._event("none")
        if self.phase == "recording":
            if cmd == "save_episode":
                return self._event("save", "save_episode")
            if cmd == "rerecord_episode":
                return self._event("discard", "rerecord_episode")
            if cmd == "stop_recording":
                return self._event("save", "stop_recording", end_session=True)
        return self._event("none")

    def tick(self, now: float) -> EpisodeEvent:
        if self.phase == "warmup" and self._warmup_t0 is not None:
            if (now - self._warmup_t0) >= self.warmup_time_s:
                return self._enter_recording(now, "warmup_done")
        if self.phase == "recording" and self._record_t0 is not None:
            if (now - self._record_t0) >= self.episode_time_s:
                return self._event("save", "timeout")
        return self._event("none")

    def complete_save(self, frames: int, end_session: bool, reason: str) -> None:
        if frames > 0:
            self.saved_episodes += 1
        self.phase = "idle"
        self._record_t0 = None
        self._warmup_t0 = None
        if end_session or self.saved_episodes >= self.num_episodes:
            self.phase = "done"
            self.needs_reset = False
            return
        if frames == 0 and reason == "timeout":
            self.phase = "done"
            self.needs_reset = False
            return
        self.needs_reset = True

    def complete_discard(self) -> None:
        self.phase = "idle"
        self._record_t0 = None
        self._warmup_t0 = None
        if self.done:
            self.needs_reset = False
            return
        self.needs_reset = True

    def mark_done(self) -> None:
        self.phase = "done"
        self.needs_reset = False

    def _enter_recording(self, now: float, reason: str) -> EpisodeEvent:
        self.phase = "recording"
        self._record_t0 = now
        return self._event("record_start", reason)

    def _event(self, kind: str, reason: str = "", end_session: bool = False) -> EpisodeEvent:
        return EpisodeEvent(
            kind=kind,
            reason=reason,
            end_session=end_session,
            episode_index=self.saved_episodes,
            attempt_index=self.attempt_index,
            saved_episodes=self.saved_episodes,
        )
