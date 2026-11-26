"""A category for every NSD image, from COCO's human object annotations.

Every NSD image is a square crop of an MS-COCO photograph, and COCO's annotators
outlined every object in those photographs and named it with one of 80 object
categories, grouped into 12 supercategories (person, vehicle, outdoor, animal,
accessory, sports, kitchen, food, furniture, electronic, appliance, indoor). This module
turns those annotations into a category per NSD image, using the supercategories.

Sources, both public, each fetched once:

    NSD   nsddata/experiments/nsd/nsd_stim_info_merged.csv
          NSD image (0-based, = MindEye2's behav column 0) -> COCO id, COCO split and
          the crop NSD applied, as (top, bottom, left, right) fractions cut away
    COCO  annotations_trainval2017.zip -> instances_train2017.json, instances_val2017.json
          (241 MB zip; deleted once the compact object table has been built)

What counts as being in the image. NSD showed a square crop, so an object cut away by
the crop was never seen. An annotated object counts only if

    * at least 50% of its bounding box survives NSD's crop, and
    * its visible area is at least 0.5% of the cropped image,

the same two thresholds NSD used when it chose the crops ("object loss" counts objects
losing more than half their pixels; objects under 0.5% of the image were ignored).
Visible area = COCO's segmentation area x the fraction of the bounding box inside the
crop.

Three labelling rules are computed side by side, because most photos contain several
kinds of object:

    dominant  the supercategory covering the most visible area (one label per image)
    contains  at least one counted object of the supercategory
    only      contains it, and nothing from any other supercategory

Torch-free, so the tests run anywhere.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import retry
from .trials import TrialPool, prepare_trial_pool
from .utils import human_bytes, log, progress, write_json

N_IMAGES = 73000
N_SUBJECTS = 8

NSD_STIM_INFO_URLS = [
    "https://natural-scenes-dataset.s3.us-east-2.amazonaws.com/"
    "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
    "https://natural-scenes-dataset.s3.amazonaws.com/"
    "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
]
COCO_ANNOTATION_URLS = [
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]
COCO_MEMBERS = {"train2017": "annotations/instances_train2017.json",
                "val2017": "annotations/instances_val2017.json"}

MIN_VISIBLE_FRACTION = 0.5     # of the object's bounding box, inside NSD's crop
MIN_AREA_FRACTION = 0.005      # of the cropped image
RULES = ("dominant", "contains", "only")


# --------------------------------------------------------------------------------------
# downloads
# --------------------------------------------------------------------------------------
def _fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "mindeye2-lora"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0) or None
        bar = progress(total=total, desc=f"download {dest.name[:28]}", unit="B")
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            bar.update(len(chunk))
        bar.close()
        fh.flush()
        os.fsync(fh.fileno())
    if total is not None and dest.stat().st_size != total:
        raise IOError(f"IncompleteRead: got {dest.stat().st_size} of {total} bytes")


def download(urls: list[str], dest: Path, what: str) -> Path:
    """Fetch the first URL that works into `dest` (atomically); a no-op if present."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dest) + ".part")
    last: Exception | None = None
    for url in urls:
        try:
            log.info("downloading %s from %s", what, url)
            retry(lambda: _fetch(url, tmp), what=f"download {what}")
            tmp.replace(dest)
            log.info("  -> %s (%s)", dest.name, human_bytes(dest.stat().st_size))
            return dest
        except Exception as exc:     # try the next mirror
            last = exc
            log.warning("could not download %s from %s (%s)", what, url, str(exc)[:120])
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {what} from any of {urls}: {last}")


# --------------------------------------------------------------------------------------
# NSD image -> COCO image
# --------------------------------------------------------------------------------------
@dataclass
class StimInfo:
    """One row per NSD image, indexed by the 0-based NSD id."""

    coco_id: np.ndarray         # (N,) int64
    coco_split: np.ndarray      # (N,) str: train2017 | val2017
    crop: np.ndarray            # (N, 4) float64: fractions cut from top, bottom, left, right
    shared1000: np.ndarray      # (N,) bool
    seen_by: np.ndarray         # (N, 8) bool: subject s saw the image
    trial_ids: np.ndarray       # (N, 8, 3) int64: 1-based trial ids, 0 = not shown

    def save(self, path: Path) -> Path:
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, coco_id=self.coco_id, coco_split=self.coco_split.astype(str),
                                crop=self.crop, shared1000=self.shared1000,
                                seen_by=self.seen_by, trial_ids=self.trial_ids)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "StimInfo":
        with np.load(path, allow_pickle=False) as z:
            return cls(**{k: z[k] for k in ("coco_id", "coco_split", "crop", "shared1000",
                                            "seen_by", "trial_ids")})


def parse_crop_box(text: str) -> tuple[float, float, float, float]:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if len(nums) != 4:
        raise ValueError(f"cropBox should hold 4 numbers, got {text!r}")
    return tuple(float(x) for x in nums)


def _flag(text) -> bool:
    return str(text).strip().lower() in ("1", "1.0", "true")


def read_stim_info(path: Path, n_images: int | None = None) -> StimInfo:
    """Parse nsd_stim_info_merged.csv (plain `csv`, so pandas is not a dependency)."""
    n = N_IMAGES if n_images is None else n_images
    coco_id = np.full(n, -1, dtype=np.int64)
    split = np.full(n, "", dtype=object)
    crop = np.zeros((n, 4), dtype=np.float64)
    shared = np.zeros(n, dtype=bool)
    seen = np.zeros((n, N_SUBJECTS), dtype=bool)
    trials = np.zeros((n, N_SUBJECTS, 3), dtype=np.int64)
    filled = np.zeros(n, dtype=bool)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            i = int(row["nsdId"])
            if not 0 <= i < n:
                raise ValueError(f"nsdId {i} outside 0..{n - 1}")
            coco_id[i] = int(row["cocoId"])
            split[i] = row["cocoSplit"].strip()
            crop[i] = parse_crop_box(row["cropBox"])
            shared[i] = _flag(row["shared1000"])
            for s in range(N_SUBJECTS):
                seen[i, s] = _flag(row[f"subject{s + 1}"])
                for r in range(3):
                    trials[i, s, r] = int(float(row[f"subject{s + 1}_rep{r}"]))
            filled[i] = True
    if not filled.all():
        raise ValueError(f"{path.name}: {int((~filled).sum())} of {n} NSD images missing")
    bad = sorted(set(split.tolist()) - set(COCO_MEMBERS))
    if bad:
        raise ValueError(f"{path.name}: unexpected cocoSplit values {bad}")
    return StimInfo(coco_id, split.astype(str), crop, shared, seen, trials)


# --------------------------------------------------------------------------------------
# COCO annotations -> visible objects per NSD image
# --------------------------------------------------------------------------------------
@dataclass
class ObjectTable:
    """Visible area of each COCO object category in each NSD image, as a fraction of
    the cropped image (0 = absent, or below the thresholds)."""

    area: np.ndarray                 # (N, 80) float32
    category_names: list[str]        # 80 COCO object categories
    supercategory_of: np.ndarray     # (80,) int64 index into supercategory_names
    supercategory_names: list[str]   # 12, in COCO's order
    n_annotated: np.ndarray          # (N,) int64 COCO objects before thresholds
    thresholds: tuple[float, float]  # (MIN_VISIBLE_FRACTION, MIN_AREA_FRACTION)

    def supercategory_area(self) -> np.ndarray:
        """(N, 12) visible area per supercategory."""
        out = np.zeros((len(self.area), len(self.supercategory_names)), dtype=np.float64)
        for k in range(len(self.supercategory_names)):
            out[:, k] = self.area[:, self.supercategory_of == k].sum(1)
        return out

    def save(self, path: Path) -> Path:
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh, area=self.area, category_names=np.array(self.category_names, dtype=str),
                supercategory_of=self.supercategory_of,
                supercategory_names=np.array(self.supercategory_names, dtype=str),
                n_annotated=self.n_annotated, thresholds=np.array(self.thresholds))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        ObjectTable.load(path)       # read back: Drive uploads asynchronously
        return path

    @classmethod
    def load(cls, path: Path) -> "ObjectTable":
        with np.load(path, allow_pickle=False) as z:
            return cls(area=z["area"], category_names=[str(x) for x in z["category_names"]],
                       supercategory_of=z["supercategory_of"].astype(np.int64),
                       supercategory_names=[str(x) for x in z["supercategory_names"]],
                       n_annotated=z["n_annotated"].astype(np.int64),
                       thresholds=tuple(float(x) for x in z["thresholds"]))


def visible_fraction(bbox: np.ndarray, size: np.ndarray, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of each bounding box inside NSD's crop, and the crop's area in pixels.

    bbox: (A, 4) COCO [x, y, w, h] in pixels; size: (A, 2) image [width, height];
    crop: (A, 4) fractions cut from (top, bottom, left, right).
    """
    w, h = size[:, 0].astype(np.float64), size[:, 1].astype(np.float64)
    top, bottom, left, right = (crop[:, i] for i in range(4))
    x0, x1 = left * w, w * (1.0 - right)
    y0, y1 = top * h, h * (1.0 - bottom)
    bx0, by0 = bbox[:, 0], bbox[:, 1]
    bx1, by1 = bx0 + bbox[:, 2], by0 + bbox[:, 3]
    iw = np.clip(np.minimum(bx1, x1) - np.maximum(bx0, x0), 0, None)
    ih = np.clip(np.minimum(by1, y1) - np.maximum(by0, y0), 0, None)
    box_area = bbox[:, 2] * bbox[:, 3]
    frac = np.divide(iw * ih, box_area, out=np.zeros_like(box_area), where=box_area > 0)
    return frac, (x1 - x0) * (y1 - y0)


def build_object_table(coco_json_by_split: dict[str, dict], stim: StimInfo) -> ObjectTable:
    """Combine COCO's instance annotations with NSD's crops.

    `coco_json_by_split` maps "train2017"/"val2017" to the parsed instances JSON.
    """
    n = len(stim.coco_id)
    categories = None
    nsd_of = {(sp, int(c)): i for i, (sp, c) in enumerate(zip(stim.coco_split, stim.coco_id))}
    area = n_annot = None
    for split, data in coco_json_by_split.items():
        cats = sorted(data["categories"], key=lambda c: c["id"])
        if categories is None:
            categories = cats
            col_of = {c["id"]: j for j, c in enumerate(cats)}
            area = np.zeros((n, len(cats)), dtype=np.float32)
            n_annot = np.zeros(n, dtype=np.int64)
        elif [c["id"] for c in cats] != [c["id"] for c in categories]:
            raise ValueError(f"{split}: COCO category list differs between splits")

        size_of = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        rows = [(nsd_of[(split, a["image_id"])], a["category_id"], *a["bbox"], a["area"],
                 *size_of[a["image_id"]])
                for a in data["annotations"] if (split, a["image_id"]) in nsd_of]
        if not rows:
            continue
        arr = np.array(rows, dtype=np.float64)
        nsd = arr[:, 0].astype(np.int64)
        col = np.array([col_of[int(c)] for c in arr[:, 1]], dtype=np.int64)
        frac, crop_area = visible_fraction(arr[:, 2:6], arr[:, 7:9], stim.crop[nsd])
        vis = arr[:, 6] * frac / crop_area
        keep = (frac >= MIN_VISIBLE_FRACTION) & (vis >= MIN_AREA_FRACTION)
        np.add.at(n_annot, nsd, 1)
        np.add.at(area, (nsd[keep], col[keep]), vis[keep].astype(np.float32))
        log.info("%s: %d annotations on NSD images, %d counted after the crop",
                 split, len(arr), int(keep.sum()))

    if categories is None:
        raise ValueError("No COCO annotations were given")
    supers: list[str] = []
    for c in categories:
        if c["supercategory"] not in supers:
            supers.append(c["supercategory"])
    return ObjectTable(
        area=area,
        category_names=[c["name"] for c in categories],
        supercategory_of=np.array([supers.index(c["supercategory"]) for c in categories]),
        supercategory_names=supers,
        n_annotated=n_annot,
        thresholds=(MIN_VISIBLE_FRACTION, MIN_AREA_FRACTION),
    )


def read_coco_zip(zip_path: Path) -> dict[str, dict]:
    """Parse the two instance files straight from the zip (no extraction to disk).

    instances_train2017.json is ~470 MB and needs a few GB of RAM while parsed; only the
    fields used here are kept, so the large segmentation polygons are freed at once.
    """
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for split, member in COCO_MEMBERS.items():
            log.info("parsing %s ...", member)
            with zf.open(member) as fh:
                data = json.load(fh)
            out[split] = {
                "categories": data["categories"],
                "images": [{"id": im["id"], "width": im["width"], "height": im["height"]}
                           for im in data["images"]],
                "annotations": [{"image_id": a["image_id"], "category_id": a["category_id"],
                                 "bbox": a["bbox"], "area": a["area"]}
                                for a in data["annotations"]],
            }
            del data
            gc.collect()
    return out


# --------------------------------------------------------------------------------------
# labels, counts, selection
# --------------------------------------------------------------------------------------
def memberships(table: ObjectTable) -> dict[str, np.ndarray]:
    """Rule -> (N, 12) bool: image i is in supercategory k under that rule."""
    sup = table.supercategory_area()
    contains = sup > 0
    dominant = np.zeros_like(contains)
    has_any = contains.any(1)
    dominant[np.flatnonzero(has_any), sup[has_any].argmax(1)] = True
    only = contains & (contains.sum(1) == 1)[:, None]
    return {"dominant": dominant, "contains": contains, "only": only}


def category_table(table: ObjectTable, pool: TrialPool) -> list[dict]:
    """Per supercategory and rule: different images the subject saw (train) and test
    images. Ordered by train images under the dominant rule."""
    members = memberships(table)
    rows = []
    for k, name in enumerate(table.supercategory_names):
        row = {"category": name}
        for rule, m in members.items():
            row[rule] = {"train": int(m[pool.image_ids, k].sum()),
                         "test": int(m[pool.test_image_ids, k].sum())}
        rows.append(row)
    no_obj = ~members["contains"].any(1)
    rows.append({"category": "(no counted object)",
                 **{rule: {"train": int(no_obj[pool.image_ids].sum()),
                           "test": int(no_obj[pool.test_image_ids].sum())} for rule in RULES}})
    return sorted(rows, key=lambda r: (r["category"].startswith("("),
                                       -r["dominant"]["train"], r["category"]))


def select_category(rows: list[dict], rule: str, n_train: int, min_test: int,
                    requested: str | None = None) -> dict:
    """The category to train on: `requested` if given and eligible, otherwise the
    eligible category with the most test images. Eligible = at least `n_train`
    different training images and at least `min_test` test images under `rule`."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    real = [r for r in rows if not r["category"].startswith("(")]
    eligible = [r for r in real if r[rule]["train"] >= n_train and r[rule]["test"] >= min_test]
    out = {"rule": rule, "n_train": n_train, "min_test": min_test,
           "eligible": [r["category"] for r in eligible]}
    if requested is not None:
        match = [r for r in real if r["category"].lower() == requested.strip().lower()]
        if not match:
            raise KeyError(f"No category {requested!r}. Known: {[r['category'] for r in real]}")
        r = match[0]
        ok = r in eligible
        return out | {"category": r["category"], "train": r[rule]["train"],
                      "test": r[rule]["test"], "eligible_ok": ok,
                      "reason": "requested" + ("" if ok else
                                               " (WARNING: below the train/test minimums)")}
    if not eligible:
        return out | {"category": None, "eligible_ok": False,
                      "reason": f"no category has >= {n_train} train and >= {min_test} "
                                f"test images under rule '{rule}'"}
    best = max(eligible, key=lambda r: (r[rule]["test"], r[rule]["train"]))
    return out | {"category": best["category"], "train": best[rule]["train"],
                  "test": best[rule]["test"], "eligible_ok": True,
                  "reason": "most test images among eligible categories"}


def crosscheck(stim: StimInfo, pool: TrialPool, subj: int) -> dict:
    """Do MindEye2's image and trial indices agree with NSD's own table?

    The labels are attached through MindEye2's behav column 0, assumed to be NSD's
    0-based image id. If that were wrong every label would belong to another image, so
    a mismatch here is an error, not a warning.
    """
    s = subj - 1
    seen = stim.seen_by[pool.image_ids, s]
    shared = stim.shared1000[pool.test_image_ids]
    ids = stim.trial_ids[pool.coco_ids, s, :]               # (T, 3) 1-based, 0 = none
    match = {off: int((ids == (pool.global_trials + off)[:, None]).any(1).sum())
             for off in (1, 0)}
    best = max(match, key=match.get)
    out = {
        "train_images_seen_by_subject": [int(seen.sum()), int(len(seen))],
        "test_images_in_shared1000": [int(shared.sum()), int(len(shared))],
        "trial_ids_agree": [match[best], int(len(pool))],
        "trial_id_offset": best,
    }
    if seen.mean() < 0.99 or (len(shared) and shared.mean() < 0.99):
        raise ValueError(
            f"MindEye2's image index does not match NSD's for subj{subj:02d}: {out}. "
            "Labels would be attached to the wrong images, so nothing was saved."
        )
    return out


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def object_table_path(ws) -> Path:
    return ws["data"] / "nsd73k_coco_objects.npz"


def load_object_table(ws, force: bool = False) -> tuple[ObjectTable, StimInfo]:
    stim_csv = download(NSD_STIM_INFO_URLS, ws["assets"] / "nsd_stim_info_merged.csv",
                        "NSD stimulus info")
    stim_npz = ws["data"] / "nsd_stim_info.npz"
    if stim_npz.exists() and not force:
        stim = StimInfo.load(stim_npz)
    else:
        stim = read_stim_info(stim_csv)
        stim.save(stim_npz)

    cache = object_table_path(ws)
    if cache.exists() and not force:
        try:
            table = ObjectTable.load(cache)
            if table.thresholds == (MIN_VISIBLE_FRACTION, MIN_AREA_FRACTION):
                return table, stim
            log.info("object table built with other thresholds; rebuilding")
        except Exception as exc:
            log.warning("cached object table unusable (%s); rebuilding", exc)

    zip_path = download(COCO_ANNOTATION_URLS, ws["assets"] / "annotations_trainval2017.zip",
                        "COCO annotations")
    table = build_object_table(read_coco_zip(zip_path), stim)
    table.save(cache)
    zip_path.unlink(missing_ok=True)   # the compact table is all that is needed from now on
    log.info("object table -> %s; removed the annotation zip", cache.name)
    return table, stim


def prepare_categories(ws, subj: int = 1, n_train: int = 688, min_test: int = 100,
                       rule: str = "dominant", category: str | None = None,
                       force: bool = False) -> dict:
    """Label every image, count each supercategory for `subj`, pick one. Trains nothing."""
    table, stim = load_object_table(ws, force=force)
    pool, _ = prepare_trial_pool(ws, subj=subj)
    check = crosscheck(stim, pool, subj)
    rows = category_table(table, pool)
    selection = select_category(rows, rule, n_train, min_test, requested=category)
    result = {
        "subj": subj,
        "n_sessions": int(len(np.unique(pool.shards))),
        "n_train_images": int(pool.n_images),
        "n_test_images": int(len(pool.test_image_ids)),
        "thresholds": {"min_visible_fraction": MIN_VISIBLE_FRACTION,
                       "min_area_fraction": MIN_AREA_FRACTION},
        "crosscheck": check,
        "table": rows,
        "selection": selection,
    }
    write_json(category_summary_path(ws, subj), result)
    ws.mark_done(f"categories_subj{subj:02d}", category=selection.get("category"), rule=rule)
    return result


def category_summary_path(ws, subj: int) -> Path:
    return ws["data"] / f"subj{subj:02d}_categories.json"


def format_category_table(result: dict) -> str:
    th, ck, sel = result["thresholds"], result["crosscheck"], result["selection"]
    seen, shared, agree = (ck["train_images_seen_by_subject"], ck["test_images_in_shared1000"],
                           ck["trial_ids_agree"])
    lines = [
        f"COCO categories · subj{result['subj']:02d} · {result['n_sessions']} sessions · "
        f"{result['n_train_images']:,} train images, {result['n_test_images']:,} test images",
        f"  an object counts if >= {th['min_visible_fraction']:.0%} of it survives NSD's crop "
        f"and it covers >= {th['min_area_fraction']:.1%} of the cropped image",
        f"  NSD index check: {seen[0]:,}/{seen[1]:,} train images seen by this subject, "
        f"{shared[0]:,}/{shared[1]:,} test images in shared1000, trial ids agree for "
        f"{agree[0]:,}/{agree[1]:,} trials (offset {ck['trial_id_offset']:+d})",
        "",
        f"  {'':22s} {'-- dominant --':>15s} {'-- contains --':>15s} {'--- only ---':>15s}",
        f"  {'category':22s} {'train':>7s} {'test':>7s} {'train':>7s} {'test':>7s} "
        f"{'train':>7s} {'test':>7s}",
    ]
    for r in result["table"]:
        cells = " ".join(f"{r[rule][k]:>7,}" for rule in RULES for k in ("train", "test"))
        lines.append(f"  {r['category'][:22]:22s} {cells}")
    lines.append("")
    if sel.get("category"):
        lines.append(
            f"  selected: {sel['category']}  (rule: {sel['rule']}; {sel['train']:,} train "
            f"images, need {sel['n_train']:,}; {sel['test']:,} test images, need "
            f">= {sel['min_test']:,}) -- {sel['reason']}")
    else:
        lines.append(f"  selected: none -- {sel['reason']}")
    if sel.get("eligible"):
        lines.append(f"  eligible under '{sel['rule']}': {', '.join(sel['eligible'])}")
    return "\n".join(lines)
