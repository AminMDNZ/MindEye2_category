"""Tests for choosing the category training set.

The trial selection is torch-free and tested here end to end on a small synthetic
subject. The last two tests touch the config and CLI, which import torch, so they are
skipped where torch is missing (they run on Colab).
"""
import numpy as np
import pytest

from mindeye_lora import assets as A
from mindeye_lora import trainsets as TS
from mindeye_lora import trials as T
from mindeye_lora.categories import ObjectTable

N_IMAGES = 50
TEST_IDS = [5, 6, 25, 35]


def object_table():
    """0-19 animal only; 20-29 animal + a bigger person; 30-39 person only; 40-49 nothing."""
    area = np.zeros((N_IMAGES, 2), dtype=np.float32)        # columns: person, dog
    area[0:20, 1] = 0.2
    area[20:30, 0], area[20:30, 1] = 0.3, 0.1
    area[30:40, 0] = 0.25
    return ObjectTable(area=area, category_names=["person", "dog"],
                       supercategory_of=np.array([0, 1]),
                       supercategory_names=["person", "animal"],
                       n_annotated=(area > 0).sum(1), thresholds=(0.5, 0.005))


def pool():
    """Every non-test image seen 3 times, in sessions (i % 5)+1, +2, +3."""
    shards = [[] for _ in range(8)]
    trial = 0
    for img in range(N_IMAGES):
        if img in TEST_IDS:
            continue
        for rep in range(3):
            k = img % 5 + rep
            shards[k].append((img, k + 1, 1000 * k + trial))
            trial += 1

    def behav(rows):
        b = np.full((len(rows), 17), -1, dtype=np.float32)
        for i, (img, sess, g) in enumerate(rows):
            b[i, 0], b[i, 2], b[i, 5] = img, sess, g
        return b

    test = behav([(t, 1, 90000 + i) for i, t in enumerate(TEST_IDS)])
    return T.build_trial_pool([behav(r) for r in shards], test)[0]


SPEC = TS.TrainSetSpec(category="animal", rule="only", n_images=10)


# -- the spec ---------------------------------------------------------------------------
def test_tag_and_description():
    assert SPEC.tag == "cat-animal-only-10-first-s0"
    assert TS.TrainSetSpec(category="Food & Drink").tag.startswith("cat-fooddrink-")
    assert "10 different 'animal' images" in SPEC.describe()


@pytest.mark.parametrize("bad", [
    {"category": "animal", "colour": "red"},
    {"kind": "session"},
    {"rule": "biggest"},
    {"viewing": "last"},
    {"n_images": 0},
])
def test_spec_rejects_bad_settings(bad):
    with pytest.raises(ValueError):
        TS.TrainSetSpec.from_dict(bad)


# -- selection --------------------------------------------------------------------------
def test_selects_different_category_images_one_first_viewing_each():
    p = pool()
    behav, info = TS.select_category_trials(p, object_table(), SPEC, subj=1)
    chosen = behav[:, 0].astype(int)
    assert behav.shape == (10, 17)
    assert len(set(chosen)) == 10 and chosen.tolist() == sorted(chosen.tolist())
    assert set(chosen) <= set(range(20)) - set(TEST_IDS)           # animal-only, not test
    assert info["n_candidates"] == 18 and info["test_images_in_category"] == 2
    for img, g in zip(chosen, behav[:, 5].astype(int)):
        first = p.global_trials[p.trials_of(int(img))[0]]
        assert g == first                                            # the first viewing
    assert (behav[:, 1] == 1).all()
    assert sum(info["trials_per_session"].values()) == 10
    assert info["category_test_image_ids"] == [5, 6]


def test_selection_is_reproducible_and_depends_on_the_seed():
    p, t = pool(), object_table()
    a, _ = TS.select_category_trials(p, t, SPEC)
    b, _ = TS.select_category_trials(p, t, SPEC)
    assert np.array_equal(a, b)
    c, _ = TS.select_category_trials(p, t, TS.TrainSetSpec(n_images=10, selection_seed=1))
    assert not np.array_equal(a[:, 0], c[:, 0])


def test_rule_changes_the_candidates():
    _, info = TS.select_category_trials(pool(), object_table(),
                                        TS.TrainSetSpec(rule="contains", n_images=10))
    assert info["n_candidates"] == 30 - 3                 # images 0-29 minus tests 5, 6, 25


def test_asking_for_more_images_than_exist_is_an_error():
    with pytest.raises(ValueError, match="Only 18 different 'animal' images"):
        TS.select_category_trials(pool(), object_table(), TS.TrainSetSpec(n_images=19))


def test_unknown_category_is_an_error():
    with pytest.raises(KeyError, match="animal"):
        TS.select_category_trials(pool(), object_table(), TS.TrainSetSpec(category="sports"))


class FakeWorkspace:
    def __init__(self, root):
        self.paths = {"data": root / "data", "assets": root / "assets"}
        for p in self.paths.values():
            p.mkdir(parents=True)

    def __getitem__(self, key):
        return self.paths[key]


def test_prepare_writes_a_reviewable_record(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "load_object_table", lambda ws: (object_table(), None))
    monkeypatch.setattr(TS, "prepare_trial_pool", lambda ws, subj: (pool(), {}))
    ws = FakeWorkspace(tmp_path)
    behav, info = TS.prepare_train_set(ws, 1, SPEC)
    path = TS.selection_path(ws, 1, SPEC.tag)
    assert path.exists() and path.name == "subj01_cat-animal-only-10-first-s0_selection.json"
    assert len(behav) == 10 and len(info["selected_image_ids"]) == 10


# -- asset naming -----------------------------------------------------------------------
def test_session_assets_keep_their_original_names(tmp_path):
    ws = FakeWorkspace(tmp_path)
    assert A.asset_meta_path(ws, 1, "1sess").name == "subj01_sess1_meta.json"
    assert A.asset_meta_path(ws, 1, "40sess").name == "subj01_sess40_meta.json"
    assert A.asset_meta_path(ws, 1, SPEC.tag).name == f"subj01_{SPEC.tag}_meta.json"


def test_load_asset_meta_accepts_a_session_count(tmp_path):
    ws = FakeWorkspace(tmp_path)
    A.write_json(A.asset_meta_path(ws, 1, "1sess"), {"ok": True})
    assert A.load_asset_meta(ws, 1, 1) == {"ok": True}
    assert A.load_asset_meta(ws, 1, "1sess") == {"ok": True}
    with pytest.raises(FileNotFoundError):
        A.load_asset_meta(ws, 1, SPEC.tag)


def test_a_custom_training_set_cannot_reuse_a_session_tag(tmp_path):
    with pytest.raises(ValueError, match="own tag"):
        A.prepare_assets(FakeWorkspace(tmp_path), behav_train=np.zeros((1, 17)), tag="1sess")


# -- config and CLI (need torch) ----------------------------------------------------------
def test_config_tag_and_run_names():
    pytest.importorskip("torch")
    from mindeye_lora.config import load_config

    main = load_config("configs/main.yaml")
    cat = load_config("configs/category_animal.yaml")
    assert main.data_tag == "1sess"
    assert main.run_name("lora_r16", 0) == "subj01_1sess_lora_r16_seed0"   # unchanged
    assert cat.data_tag == "cat-animal-only-700-first-s0"
    assert cat.run_name("lora_r16", 0) == "subj01_cat-animal-only-700-first-s0_lora_r16_seed0"


def test_category_results_never_land_in_the_published_results_folder(tmp_path):
    pytest.importorskip("torch")
    from mindeye_lora.cli import _scope_results
    from mindeye_lora.config import load_config

    class WS:
        def __init__(self):
            self.paths = {k: tmp_path / v for k, v in
                          {"results": "results", "figures": "results/figures",
                           "reports": "results/reports"}.items()}

    ws = WS()
    _scope_results(ws, load_config("configs/main.yaml"))
    assert ws.paths["reports"] == tmp_path / "results" / "reports"
    _scope_results(ws, load_config("configs/category_animal.yaml"))
    tag = "cat-animal-only-700-first-s0"
    assert ws.paths["reports"] == tmp_path / "results" / tag / "reports"
    assert ws.paths["results"] == tmp_path / "results" / tag
