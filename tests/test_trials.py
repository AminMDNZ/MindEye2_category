"""Tests for the all-sessions trial pool.

Torch-free: the trial pool decides which brain responses the category / repeat
experiments train on, so its grouping and its leakage guard must be checkable anywhere.
"""
import io
import tarfile

import numpy as np
import pytest

from mindeye_lora import trials as T
from mindeye_lora.assets import (
    BEHAV_COCO_IDX,
    BEHAV_GLOBAL_TRIAL,
    BEHAV_SESSION,
    read_behav_tar,
)
from mindeye_lora.utils import read_json, write_json


def behav(rows):
    """(coco_id, session, global_trial) tuples -> an (N, 17) behav table, -1 elsewhere."""
    out = np.full((len(rows), 17), -1, dtype=np.float32)
    for i, (coco, session, trial) in enumerate(rows):
        out[i, BEHAV_COCO_IDX] = coco
        out[i, BEHAV_SESSION] = session
        out[i, BEHAV_GLOBAL_TRIAL] = trial
    return out


def fixture():
    # image 10: once in each of sessions 1-3      image 20: twice in session 1
    # image 30: once                              image 40: three times in session 3
    # image 99: a test image that leaked into a train shard -> must be dropped
    s1 = behav([(10, 1, 0), (20, 1, 1), (20, 1, 2), (99, 1, 3)])
    s2 = behav([(10, 2, 100), (30, 2, 101)])
    s3 = behav([(40, 3, 200), (10, 3, 201), (40, 3, 202), (40, 3, 203)])
    test = behav([(99, 1, 500), (98, 2, 501)])
    return [s1, s2, s3], test


def test_trials_are_grouped_by_image_in_time_order():
    pool, _ = T.build_trial_pool(*fixture())
    assert pool.image_ids.tolist() == [10, 20, 30, 40]
    assert pool.image_counts.tolist() == [3, 2, 1, 3]
    idx = pool.trials_of(10)
    assert pool.shards[idx].tolist() == [0, 1, 2]
    assert pool.global_trials[idx].tolist() == [0, 100, 201]
    assert pool.shards[pool.trials_of(20)].tolist() == [0, 0]
    assert pool.images_with(3).tolist() == [10, 40]


def test_test_images_never_enter_the_pool():
    pool, report = T.build_trial_pool(*fixture())
    assert 99 not in pool.image_ids
    assert report["dropped_test_overlap_trials"] == 1
    assert report["dropped_test_overlap_images"] == 1
    assert report["n_test_images"] == 2
    assert pool.test_image_ids.tolist() == [98, 99]


def test_missing_values_are_dropped():
    shards, test = fixture()
    shards[1] = np.concatenate([shards[1], behav([(-1, 2, 102), (31, 2, -1)])])
    pool, report = T.build_trial_pool(shards, test)
    assert report["dropped_invalid_trials"] == 2
    assert 31 not in pool.image_ids


def test_reading_a_shard_twice_is_an_error():
    shards, test = fixture()
    with pytest.raises(ValueError, match="more than once"):
        T.build_trial_pool([shards[0], shards[0]], test)


def test_unknown_image_raises():
    pool, _ = T.build_trial_pool(*fixture())
    with pytest.raises(KeyError):
        pool.trials_of(12345)


def test_summary_reports_what_the_design_depends_on():
    pool, report = T.build_trial_pool(*fixture())
    s = T.summarise_pool(pool, report)
    assert s["n_trials"] == 9 and s["n_images"] == 4 and s["n_sessions"] == 3
    assert s["trials_per_image"] == {1: 1, 2: 1, 3: 2}
    assert s["trials_per_session"]["first_session"] == 3   # image 99 removed
    assert s["session_column"] == "consistent: session = shard + 1"
    assert s["repeats"]["images_seen_more_than_once"] == 3          # 10, 20, 40
    assert s["repeats"]["all_repeats_in_one_session"] == 2          # 20, 40
    assert s["repeats"]["median_sessions_spanned"] == 1.0           # median(3, 1, 1)
    text = T.format_summary(s)
    assert "3x: 2 images" in text and "1 images dropped" in text


def test_summary_flags_a_session_column_that_disagrees_with_shard_order():
    shards, test = fixture()
    shards[1][:, BEHAV_SESSION] = 7
    pool, report = T.build_trial_pool(shards, test)
    assert T.summarise_pool(pool, report)["session_column"].startswith("INCONSISTENT")


def test_summary_still_formats_after_a_json_round_trip(tmp_path):
    # JSON turns the integer keys of trials_per_image into strings; the cached path
    # of `cli trials` prints the summary from JSON.
    pool, report = T.build_trial_pool(*fixture())
    write_json(tmp_path / "s.json", {"subj": 1, **T.summarise_pool(pool, report)})
    assert "subj01" in T.format_summary(read_json(tmp_path / "s.json"))


def test_save_and_load_round_trip(tmp_path):
    pool, _ = T.build_trial_pool(*fixture())
    path = pool.save(tmp_path / "pool.npz")
    back = T.TrialPool.load(path)
    for name in ("coco_ids", "shards", "sessions", "global_trials", "image_ids",
                 "image_counts", "image_offsets", "test_image_ids"):
        assert np.array_equal(getattr(back, name), getattr(pool, name)), name
    assert not (tmp_path / "pool.npz.tmp").exists()


def test_load_rejects_an_incomplete_file(tmp_path):
    np.savez(tmp_path / "bad.npz", coco_ids=np.arange(3))
    with pytest.raises(ValueError, match="missing"):
        T.TrialPool.load(tmp_path / "bad.npz")


def test_read_behav_tar_ignores_neighbouring_timepoints(tmp_path):
    """Shards also carry past/future/olds behav arrays; only `behav` is the trial."""
    path = tmp_path / "0.tar"
    arrays = {
        "sample000000.behav.npy": behav([(10, 1, 0)]),
        "sample000000.past_behav.npy": behav([(11, 1, 7)]),
        "sample000000.future_behav.npy": behav([(12, 1, 8)]),
        "sample000000.olds_behav.npy": behav([(10, 2, 100)]),
        "sample000001.behav.npy": behav([(20, 1, 1)]),
    }
    with tarfile.open(path, "w") as tf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            info = tarfile.TarInfo(name)
            info.size = buf.tell()
            buf.seek(0)
            tf.addfile(info, buf)
    out = read_behav_tar(path)
    assert out.shape == (2, 17)
    assert out[:, BEHAV_COCO_IDX].tolist() == [10, 20]


# -- orchestration, with the hub replaced by the fixture --------------------------------
class FakeWorkspace:
    def __init__(self, root):
        self.paths = {"data": root / "data", "assets": root / "assets"}
        for p in self.paths.values():
            p.mkdir(parents=True)
        self.done = {}

    def __getitem__(self, key):
        return self.paths[key]

    def mark_done(self, stage, **info):
        self.done[stage] = info


@pytest.fixture
def fake_hub(monkeypatch):
    shards, test = fixture()
    tables = {T.WDS_TRAIN_TPL.format(subj=1, shard=k): t for k, t in enumerate(shards)}
    tables["wds/subj01/new_test/0.tar"] = test
    downloads = []

    def hf_download(name, local_dir):
        downloads.append(name)
        return name

    monkeypatch.setattr(T, "NSESSIONS_PER_SUBJ", {1: 3})
    monkeypatch.setattr(T, "repo_files", lambda: list(tables))
    monkeypatch.setattr(T, "hf_download", hf_download)
    monkeypatch.setattr(T, "read_behav_tar", lambda name: tables[name])
    return downloads


def test_prepare_builds_once_then_reuses_the_cache(tmp_path, fake_hub):
    ws = FakeWorkspace(tmp_path)
    pool, summary = T.prepare_trial_pool(ws, subj=1)
    assert len(pool) == 9 and summary["subj"] == 1 and summary["n_sessions"] == 3
    assert "trials_subj01_3sess" in ws.done
    pool_path, summary_path = T.pool_paths(ws, 1, 3)
    assert pool_path.exists() and summary_path.exists()

    n_downloads = len(fake_hub)
    again, _ = T.prepare_trial_pool(ws, subj=1)
    assert len(fake_hub) == n_downloads, "a cached pool should not touch the hub"
    assert np.array_equal(again.global_trials, pool.global_trials)


def test_prepare_can_limit_sessions(tmp_path, fake_hub):
    pool, summary = T.prepare_trial_pool(FakeWorkspace(tmp_path), subj=1, max_sessions=2)
    assert summary["n_sessions"] == 2
    assert set(pool.shards.tolist()) == {0, 1}


def test_prepare_rejects_more_sessions_than_the_subject_completed(tmp_path, fake_hub):
    with pytest.raises(ValueError, match="completed 3 sessions"):
        T.prepare_trial_pool(FakeWorkspace(tmp_path), subj=1, max_sessions=4)
