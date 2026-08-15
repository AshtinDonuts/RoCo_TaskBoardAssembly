from teleop.episode import EpisodeSession


def test_warmup_then_record_then_timeout_save():
    sess = EpisodeSession(episode_time_s=10.0, warmup_time_s=2.0, num_episodes=1)
    ev = sess.start(0.0)
    assert ev.kind == "warmup_start"
    assert sess.is_warmup
    assert sess.step("save_episode", 0.5).is_noop
    assert sess.tick(1.9).is_noop
    ev = sess.tick(2.0)
    assert ev.kind == "record_start"
    assert sess.is_recording
    ev = sess.tick(12.0)
    assert ev.kind == "save"
    assert ev.reason == "timeout"
    sess.complete_save(frames=8, end_session=False, reason="timeout")
    assert sess.done
    assert sess.saved_episodes == 1


def test_early_save_and_rerecord():
    sess = EpisodeSession(episode_time_s=60.0, warmup_time_s=1.0, num_episodes=2)
    sess.start(0.0)
    sess.tick(1.0)
    ev = sess.step("save_episode", 3.0)
    assert ev.kind == "save"
    sess.complete_save(4, end_session=False, reason="save_episode")
    assert not sess.done
    assert sess.needs_reset
    assert sess.saved_episodes == 1

    ev = sess.start(10.0)
    assert ev.kind == "warmup_start"
    sess.tick(11.0)
    ev = sess.step("rerecord_episode", 12.0)
    assert ev.kind == "discard"
    sess.complete_discard()
    assert sess.needs_reset
    assert sess.saved_episodes == 1


def test_stop_during_warmup_does_not_save():
    sess = EpisodeSession(episode_time_s=60.0, warmup_time_s=5.0, num_episodes=3)
    sess.start(0.0)
    ev = sess.step("stop_recording", 1.0)
    assert ev.kind == "session_end"
    assert sess.done
    assert sess.saved_episodes == 0


def test_stop_during_recording_requests_save():
    sess = EpisodeSession(episode_time_s=60.0, warmup_time_s=0.0, num_episodes=5)
    ev = sess.start(0.0)
    assert ev.kind == "record_start"
    ev = sess.step("stop_recording", 2.0)
    assert ev.kind == "save"
    assert ev.end_session
    sess.complete_save(3, end_session=True, reason="stop_recording")
    assert sess.done
    assert sess.saved_episodes == 1


def test_task_success_is_not_part_of_session():
    sess = EpisodeSession(episode_time_s=5.0, warmup_time_s=0.0, num_episodes=1)
    sess.start(0.0)
    sess.step("part_done", 0.1)
    sess.step("abort", 0.2)
    ev = sess.step("save_episode", 0.3)
    assert ev.kind == "save"
    sess.complete_save(1, False, "save_episode")
    assert sess.saved_episodes == 1
