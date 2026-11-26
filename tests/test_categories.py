"""Tests for the COCO-based image categories.

Torch-free. A six-image NSD stand-in with hand-checked answers: which objects survive
NSD's crop, which supercategory each image gets under each rule, how they are counted
for a subject, and which category is selected.
"""
import csv
import io
import json
import zipfile

import numpy as np
import pytest

from mindeye_lora import categories as C
from mindeye_lora import trials as T

N = 6

# nsd id -> (coco id, split, (W, H), cropBox, shared1000, seen by subject 1)
IMAGES = {
    0: (100, "train2017", (640, 480), "(0, 0, 0.125, 0.125)", False, True),
    1: (101, "train2017", (500, 500), "(0, 0, 0, 0)", False, True),
    2: (200, "val2017", (500, 500), "(0, 0, 0, 0)", True, True),
    3: (201, "val2017", (400, 600), "(0, 0.3333333333, 0, 0)", False, True),
    4: (102, "train2017", (500, 500), "(0, 0, 0, 0)", True, True),
    5: (103, "train2017", (500, 500), "(0, 0, 0, 0)", False, True),
}
TRAIN_IMAGES, TEST_IMAGES = [0, 1, 3, 5], [2, 4]
CATEGORIES = [  # deliberately unsorted: the loader orders by id
    {"id": 18, "name": "dog", "supercategory": "animal"},
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 52, "name": "banana", "supercategory": "food"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
]
ANNOTATIONS = {  # coco id -> [(category id, bbox, segmentation area)]
    # 0: dog fully inside the crop; person mostly cut away by the left crop (20% left)
    100: [(18, [100, 100, 200, 200], 30000), (1, [0, 0, 100, 100], 8000)],
    # 1: big person, small dog -> dominant person, contains both, "only" neither
    101: [(1, [0, 0, 300, 300], 60000), (18, [300, 300, 100, 100], 5000)],
    # 2: a banana too small to count (0.12% of the image)
    200: [(52, [10, 10, 20, 20], 300)],
    # 3: portrait, top crop keeps y 0..400: the car is cut away, the dog stays
    201: [(3, [50, 450, 100, 100], 8000), (18, [50, 50, 100, 100], 9000)],
    # 4 (coco 102): no annotations at all
    # 5: big car, small person
    103: [(3, [0, 0, 400, 400], 120000), (1, [400, 400, 50, 50], 2000)],
    # not an NSD image: must be ignored
    999: [(1, [0, 0, 10, 10], 100)],
}


def trial_id(nsd: int, rep: int) -> int:
    """1-based NSD trial id of subject 1's rep-th viewing of image `nsd`."""
    return 1 + 10 * nsd + rep


def write_stim_csv(path, drop_row=None, split_override=None):
    header = ["", "cocoId", "cocoSplit", "cropBox", "loss", "nsdId", "flagged", "BOLD5000",
              "shared1000"] + [f"subject{s}" for s in range(1, 9)] + \
             [f"subject{s}_rep{r}" for s in range(1, 9) for r in range(3)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for nsd, (coco, split, _size, crop, shared, seen) in IMAGES.items():
            if nsd == drop_row:
                continue
            reps = [trial_id(nsd, r) if seen else 0 for r in range(3)]
            w.writerow([nsd, coco, split_override or split, crop, 0.0, nsd, False, False,
                        shared, int(seen), *[0] * 7, *reps, *[0] * 21])
    return path


def coco_json():
    out = {}
    for split in ("train2017", "val2017"):
        ims = [(coco, size) for coco, sp, size, *_ in IMAGES.values() if sp == split]
        if split == "train2017":
            ims.append((999, (100, 100)))
        ids = {coco for coco, _ in ims}
        out[split] = {
            "categories": CATEGORIES,
            "images": [{"id": c, "width": wh[0], "height": wh[1], "file_name": "x.jpg"}
                       for c, wh in ims],
            "annotations": [{"image_id": c, "category_id": cat, "bbox": box, "area": area,
                             "segmentation": [[0, 0, 1, 1]], "iscrowd": 0, "id": 1}
                            for c, anns in ANNOTATIONS.items() if c in ids
                            for cat, box, area in anns],
        }
    return out


def stim(tmp_path):
    return C.read_stim_info(write_stim_csv(tmp_path / "stim.csv"), n_images=N)


def table(tmp_path):
    return C.build_object_table(coco_json(), stim(tmp_path))


def pool():
    """Subject 1's trials, global trial = NSD trial id - 1 (MindEye2's convention)."""
    def behav(nsds):
        rows = [(nsd, trial_id(nsd, r) - 1) for nsd in nsds for r in range(3)]
        b = np.full((len(rows), 17), -1, dtype=np.float32)
        b[:, 0] = [n for n, _ in rows]
        b[:, 2] = 1
        b[:, 5] = [t for _, t in rows]
        return b

    return T.build_trial_pool([behav(TRAIN_IMAGES)], behav(TEST_IMAGES))[0]


# -- NSD stimulus table ---------------------------------------------------------------
def test_parse_crop_box():
    assert C.parse_crop_box("(0, 0, 0.1671875, 0.1671875)") == (0, 0, 0.1671875, 0.1671875)
    assert C.parse_crop_box("(1e-3, 0, 0, 0)")[0] == pytest.approx(0.001)
    with pytest.raises(ValueError):
        C.parse_crop_box("(0, 0, 0.1)")


def test_read_stim_info(tmp_path):
    s = stim(tmp_path)
    assert s.coco_id.tolist() == [100, 101, 200, 201, 102, 103]
    assert s.coco_split[2] == "val2017"
    assert s.crop[0].tolist() == [0, 0, 0.125, 0.125]
    assert s.shared1000.tolist() == [False, False, True, False, True, False]
    assert s.seen_by[:, 0].all() and not s.seen_by[:, 1].any()
    assert s.trial_ids[3, 0].tolist() == [31, 32, 33]


def test_read_stim_info_rejects_missing_rows_and_unknown_splits(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        C.read_stim_info(write_stim_csv(tmp_path / "a.csv", drop_row=4), n_images=N)
    with pytest.raises(ValueError, match="cocoSplit"):
        C.read_stim_info(write_stim_csv(tmp_path / "b.csv", split_override="test2014"),
                         n_images=N)


# -- crop-aware object table ----------------------------------------------------------
def test_visible_fraction_of_a_box_half_inside_the_crop():
    frac, crop_area = C.visible_fraction(np.array([[60.0, 0, 40, 10]]),
                                         np.array([[640, 480]]),
                                         np.array([[0, 0, 0.125, 0.125]]))   # x 80..560
    assert frac[0] == pytest.approx(0.5) and crop_area[0] == pytest.approx(480 * 480)


def test_objects_cut_away_or_too_small_do_not_count(tmp_path):
    t = table(tmp_path)
    col = {n: i for i, n in enumerate(t.category_names)}
    assert t.category_names == ["person", "car", "dog", "banana"]     # ordered by COCO id
    assert t.supercategory_names == ["person", "vehicle", "animal", "food"]
    assert t.area[0, col["dog"]] == pytest.approx(30000 / (480 * 480))
    assert t.area[0, col["person"]] == 0          # only 20% survives the left crop
    assert t.area[2].sum() == 0                   # banana below 0.5% of the image
    assert t.area[3, col["car"]] == 0             # below the top crop
    assert t.area[3, col["dog"]] == pytest.approx(9000 / (400 * 400))
    assert t.n_annotated.tolist() == [2, 2, 1, 2, 0, 2]   # before thresholds; 999 ignored


def test_three_labelling_rules(tmp_path):
    m = C.memberships(table(tmp_path))
    person, vehicle, animal = 0, 1, 2

    def labels(rule, i):
        return set(np.flatnonzero(m[rule][i]).tolist())

    assert [labels("dominant", i) for i in range(N)] == \
        [{animal}, {person}, set(), {animal}, set(), {vehicle}]
    assert labels("contains", 1) == {person, animal}
    assert labels("contains", 5) == {vehicle, person}
    assert [labels("only", i) for i in range(N)] == \
        [{animal}, set(), set(), {animal}, set(), set()]


def test_category_table_counts_train_and_test_images(tmp_path):
    rows = {r["category"]: r for r in C.category_table(table(tmp_path), pool())}
    assert rows["animal"]["dominant"] == {"train": 2, "test": 0}
    assert rows["person"]["contains"] == {"train": 2, "test": 0}
    assert rows["vehicle"]["only"] == {"train": 0, "test": 0}
    assert rows["(no counted object)"]["dominant"] == {"train": 0, "test": 2}
    assert list(rows)[-1] == "(no counted object)"


# -- selection ------------------------------------------------------------------------
ROWS = [
    {"category": "person", "dominant": {"train": 3000, "test": 300}},
    {"category": "animal", "dominant": {"train": 900, "test": 120}},
    {"category": "food", "dominant": {"train": 700, "test": 150}},
    {"category": "sports", "dominant": {"train": 500, "test": 400}},
    {"category": "(no counted object)", "dominant": {"train": 9000, "test": 900}},
]


def test_selects_the_eligible_category_with_most_test_images():
    sel = C.select_category(ROWS, "dominant", n_train=688, min_test=100)
    assert sel["category"] == "person"
    assert sel["eligible"] == ["person", "animal", "food"]   # sports has too few train


def test_thresholds_change_the_choice():
    sel = C.select_category(ROWS, "dominant", n_train=688, min_test=200)
    assert sel["category"] == "person" and sel["eligible"] == ["person"]
    sel = C.select_category(ROWS, "dominant", n_train=3500, min_test=10)
    assert sel["category"] is None and "no category" in sel["reason"]


def test_requested_category_is_honoured_and_flagged_if_too_small():
    sel = C.select_category(ROWS, "dominant", 688, 100, requested="Animal")
    assert sel["category"] == "animal" and sel["eligible_ok"]
    sel = C.select_category(ROWS, "dominant", 688, 100, requested="sports")
    assert sel["category"] == "sports" and not sel["eligible_ok"] and "WARNING" in sel["reason"]
    with pytest.raises(KeyError):
        C.select_category(ROWS, "dominant", 688, 100, requested="spaceship")
    with pytest.raises(ValueError):
        C.select_category(ROWS, "biggest", 688, 100)


# -- index cross-check ----------------------------------------------------------------
def test_crosscheck_agrees_with_nsd(tmp_path):
    out = C.crosscheck(stim(tmp_path), pool(), subj=1)
    assert out["train_images_seen_by_subject"] == [4, 4]
    assert out["test_images_in_shared1000"] == [2, 2]
    assert out["trial_ids_agree"] == [12, 12] and out["trial_id_offset"] == 1


def test_crosscheck_refuses_a_wrong_image_index(tmp_path):
    s = stim(tmp_path)
    s.seen_by[:, 0] = False          # as if behav column 0 meant something else
    with pytest.raises(ValueError, match="wrong images"):
        C.crosscheck(s, pool(), subj=1)


# -- files ----------------------------------------------------------------------------
def test_read_coco_zip_keeps_only_what_is_needed(tmp_path):
    path = tmp_path / "ann.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for split, member in C.COCO_MEMBERS.items():
            zf.writestr(member, json.dumps(coco_json()[split]))
    data = C.read_coco_zip(path)
    assert set(data) == {"train2017", "val2017"}
    assert set(data["train2017"]["annotations"][0]) == {"image_id", "category_id", "bbox", "area"}
    t = C.build_object_table(data, stim(tmp_path))
    assert np.allclose(t.area, table(tmp_path).area)


def test_tables_round_trip(tmp_path):
    t = table(tmp_path)
    back = C.ObjectTable.load(t.save(tmp_path / "t.npz"))
    assert np.array_equal(back.area, t.area) and back.category_names == t.category_names
    assert back.supercategory_names == t.supercategory_names
    assert back.thresholds == (C.MIN_VISIBLE_FRACTION, C.MIN_AREA_FRACTION)
    s = stim(tmp_path)
    s2 = C.StimInfo.load(s.save(tmp_path / "s.npz"))
    assert np.array_equal(s2.trial_ids, s.trial_ids) and s2.coco_split.tolist() == s.coco_split.tolist()


def test_download_falls_back_to_the_next_mirror(tmp_path, monkeypatch):
    calls = []

    def fetch(url, dest):
        calls.append(url)
        if "first" in url:
            raise ValueError("404 Not Found")      # not transient: no retry, next mirror
        dest.write_bytes(b"ok")

    monkeypatch.setattr(C, "_fetch", fetch)
    out = C.download(["https://first/x", "https://second/x"], tmp_path / "x.csv", "x")
    assert out.read_bytes() == b"ok" and calls == ["https://first/x", "https://second/x"]
    assert not (tmp_path / "x.csv.part").exists()
    C.download(["https://first/x"], tmp_path / "x.csv", "x")
    assert len(calls) == 2, "an existing file is not downloaded again"


# -- end to end -----------------------------------------------------------------------
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


def test_prepare_categories_end_to_end(tmp_path, monkeypatch):
    fetched = []

    def fake_download(urls, dest, what):
        if dest.exists():
            return dest
        fetched.append(dest.name)
        if dest.suffix == ".csv":
            write_stim_csv(dest)
        else:
            with zipfile.ZipFile(dest, "w") as zf:
                for split, member in C.COCO_MEMBERS.items():
                    zf.writestr(member, json.dumps(coco_json()[split]))
        return dest

    monkeypatch.setattr(C, "N_IMAGES", N)
    monkeypatch.setattr(C, "download", fake_download)
    monkeypatch.setattr(C, "prepare_trial_pool", lambda ws, subj: (pool(), {}))

    ws = FakeWorkspace(tmp_path)
    result = C.prepare_categories(ws, subj=1, n_train=2, min_test=0)
    assert result["selection"]["category"] == "animal"      # 2 train images, tie on test
    assert result["crosscheck"]["trial_ids_agree"] == [12, 12]
    assert C.category_summary_path(ws, 1).exists()
    assert not (tmp_path / "assets" / "annotations_trainval2017.zip").exists()
    text = C.format_category_table(result)
    assert "selected: animal" in text and "NSD index check: 4/4" in text

    again = C.prepare_categories(ws, subj=1, n_train=2, min_test=0, rule="contains")
    assert fetched == ["nsd_stim_info_merged.csv", "annotations_trainval2017.zip"], \
        "the second run must reuse the cached tables"
    assert again["selection"]["rule"] == "contains"
    # contains: person (images 1 and 5) and animal (0, 1, 3) both have >= 2 train images
    # and 0 test images; ties go to the larger training set
    assert again["selection"]["category"] == "animal"
