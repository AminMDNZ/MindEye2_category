"""Tests for step 3, the category-vs-session-1 comparison.

Two synthetic runs score the same 60 test images, but each lists them in its own row
order (as real runs do: each training set has its own image cache). The category run is
built to be better on the 20 category images and worse on the other 40, so the right
answer is known. The within-category re-scoring needs torch and is skipped without it.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mindeye_lora import compare_category as CC

N, N_CAT = 60, 20
TEST_IDS = np.arange(1000, 1000 + N)            # NSD ids of the test images
CAT_IDS = TEST_IDS[:N_CAT]
TAG, BASE = "cat-animal-only-700-first-s0", "1sess"


class FakeWorkspace:
    def __init__(self, root):
        self.paths = {k: root / k for k in ("data", "runs", "results")}
        for p in self.paths.values():
            p.mkdir(parents=True)

    def __getitem__(self, key):
        return self.paths[key]


def cfg():
    return SimpleNamespace(
        subj=1, data_tag=TAG, arms=[SimpleNamespace(name="lora_r16")],
        train_set={"kind": "category", "category": "animal", "rule": "only", "n_images": 700})


def write_run(ws, tag, seed, two_way_by_id, cos_by_id, rng):
    """A run whose image cache lists train images first, then the test images shuffled."""
    cache = np.concatenate([np.arange(5) + (0 if tag == BASE else 50), rng.permutation(TEST_IDS)])
    np.save(ws["data"] / f"subj01_coco_idx_{tag}.npy", cache)
    rows = np.arange(5, len(cache))                         # test rows, cache order
    rows = rng.permutation(rows)                             # scored in yet another order
    ids = cache[rows]
    d = ws["runs"] / f"subj01_{tag}_lora_r16_seed{seed}"
    d.mkdir(parents=True)
    np.savez(d / "per_sample_metrics.npz", rows=rows,
             two_way_clip=np.array([two_way_by_id[i] for i in ids]),
             cosine=np.array([cos_by_id[i] for i in ids]))
    return d


@pytest.fixture
def workspace(tmp_path):
    ws = FakeWorkspace(tmp_path)
    rng = np.random.default_rng(0)
    noise = {int(i): rng.normal(0, 0.01) for i in TEST_IDS}
    base = {int(i): 0.90 + noise[int(i)] for i in TEST_IDS}
    cat = {int(i): base[int(i)] + (0.05 if i in CAT_IDS else -0.10) + rng.normal(0, 0.005)
           for i in TEST_IDS}
    cos = {int(i): 0.3 for i in TEST_IDS}
    write_run(ws, TAG, 0, cat, cos, rng)
    write_run(ws, BASE, 0, base, cos, rng)
    for s in (1, 2):
        write_run_extra = {int(i): base[int(i)] + 0.001 * s for i in TEST_IDS}
        d = ws["runs"] / f"subj01_{BASE}_lora_r16_seed{s}"
        d.mkdir()
        cache = np.load(ws["data"] / f"subj01_coco_idx_{BASE}.npy")
        rows = np.arange(5, len(cache))
        np.savez(d / "per_sample_metrics.npz", rows=rows,
                 two_way_clip=np.array([write_run_extra[int(i)] for i in cache[rows]]),
                 cosine=np.full(len(rows), 0.3))
    (ws["data"] / f"subj01_{TAG}_selection.json").write_text(
        json.dumps({"category_test_image_ids": CAT_IDS.tolist()}))
    return ws


def test_align_pairs_by_image_not_by_position():
    ia, ib, ids = CC.align(np.array([7, 3, 5]), np.array([5, 7, 3, 9]))
    assert ids.tolist() == [3, 5, 7]
    assert np.array([7, 3, 5])[ia].tolist() == [3, 5, 7]
    assert np.array([5, 7, 3, 9])[ib].tolist() == [3, 5, 7]
    with pytest.raises(ValueError, match="twice"):
        CC.align(np.array([1, 1]), np.array([1]))


def test_comparison_finds_the_planted_effect(workspace):
    res = CC.run_comparison(workspace, cfg(), within=False)
    rows = {(r["group"], r["metric"]): r for r in res["rows"]}
    cat = rows[("animal (only) test images", "two_way_clip")]
    other = rows[("other test images", "two_way_clip")]
    allr = rows[("all test images", "two_way_clip")]
    assert cat["n"] == N_CAT and other["n"] == N - N_CAT and allr["n"] == N
    assert cat["mean_diff"] == pytest.approx(0.05, abs=0.01) and cat["ci_low"] > 0
    assert other["mean_diff"] == pytest.approx(-0.10, abs=0.01) and other["ci_high"] < 0
    assert res["primary"] is res["rows"][0]
    assert set(res["baseline_two_way_by_seed"]) == {0, 1, 2}
    text = CC.format_result(res)
    assert "category training is better" in text


def test_missing_scores_are_reported_clearly(workspace):
    (workspace["runs"] / f"subj01_{BASE}_lora_r16_seed0" / "per_sample_metrics.npz").unlink()
    with pytest.raises(FileNotFoundError, match="predict and evaluate"):
        CC.run_comparison(workspace, cfg(), within=False)


def test_needs_a_category_config(workspace):
    c = cfg()
    c.train_set = None
    with pytest.raises(ValueError, match="train_set"):
        CC.run_comparison(workspace, c, within=False)


def test_within_category_rescoring(workspace):
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    for tag in (TAG, BASE):
        d = workspace["runs"] / f"subj01_{tag}_lora_r16_seed0" / "predictions"
        d.mkdir()
        cache = np.load(workspace["data"] / f"subj01_coco_idx_{tag}.npy")
        rows = np.arange(5, len(cache))
        target = rng.normal(size=(len(rows), 4, 8)).astype(np.float16)
        noise = 0.1 if tag == TAG else 2.0               # category run predicts far better
        np.save(d / "target.npy", target)
        np.save(d / "prior.npy", (target + noise * rng.normal(size=target.shape)).astype(np.float16))
        np.save(d / "rows.npy", rows)
        (d / "meta.json").write_text(json.dumps({"fields": ["prior", "target"]}))
    res = CC.run_comparison(workspace, cfg(), within=True)
    w = res["rows"][-1]
    assert w["group"].endswith("vs each other only") and w["n"] == N_CAT
    assert w["mean_diff"] > 0


# -- reconstructions (the decoding itself needs a GPU; these test the bookkeeping) ------
def test_category_positions_pick_the_same_images_in_every_run(workspace):
    for tag in (TAG, BASE):
        d = workspace["runs"] / f"subj01_{tag}_lora_r16_seed0" / "predictions"
        d.mkdir()
        cache = np.load(workspace["data"] / f"subj01_coco_idx_{tag}.npy")
        rows = np.random.default_rng(3).permutation(np.arange(5, len(cache)))
        np.save(d / "rows.npy", rows)
    got = {}
    for tag in (TAG, BASE):
        d = workspace["runs"] / f"subj01_{tag}_lora_r16_seed0"
        pos, ids = CC.category_positions(workspace, 1, tag, d, CAT_IDS, n=8)
        rows = np.load(d / "predictions" / "rows.npy")
        assert CC.nsd_ids_for_rows(workspace, 1, tag, rows[pos]).tolist() == ids.tolist()
        got[tag] = ids
    assert got[TAG].tolist() == got[BASE].tolist() and len(got[TAG]) == 8
    assert set(got[TAG]) <= set(CAT_IDS)
    with pytest.raises(ValueError, match="missing"):
        CC.category_positions(workspace, 1, TAG, workspace["runs"] / f"subj01_{TAG}_lora_r16_seed0",
                              np.array([5]))


def test_image_metric_rows_respect_metric_direction():
    rng = np.random.default_rng(0)
    base = {"ssim": rng.uniform(0.3, 0.4, 50), "swav": rng.uniform(0.5, 0.6, 50)}
    cat = {"ssim": base["ssim"] + 0.05 + rng.normal(0, 0.005, 50),
           "swav": base["swav"] - 0.05 + rng.normal(0, 0.005, 50)}   # better on both
    rows = {r["metric"]: r for r in CC.image_metric_rows(cat, base)}
    assert rows["ssim"]["higher_is_better"] and not rows["swav"]["higher_is_better"]
    text = CC.format_recon_result({"n_images": 50, "category_run": "c", "baseline_run": "b",
                                   "rows": list(rows.values()), "figure": "f.png"})
    assert text.count("category better") == 2
