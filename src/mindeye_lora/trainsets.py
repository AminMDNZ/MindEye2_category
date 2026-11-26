"""Which trials to fine-tune on, when it is not simply "the first N sessions".

The main experiment trains on session 1's 688 trials. The category experiment instead
trains on `n_images` *different* images of one COCO category, drawn from every session
the subject completed, each used once. Everything downstream (assets, embeddings,
training, prediction, evaluation) is unchanged; it simply receives a different list of
trials. This module builds that list.

A training set is described by a small dict in the config:

    train_set:
      kind: category
      category: animal     # a COCO supercategory, see `cli categories`
      rule: only           # dominant | contains | only (see categories.py)
      n_images: 700        # different images, one trial each
      viewing: first       # use the subject's first viewing of each image
      selection_seed: 0    # which images, when more are available than needed

and gets a short tag (e.g. `cat-animal-only-700-first-s0`) that names its data files and
runs, so it never collides with the session-based runs (`1sess`).

Choices, and why:

* **Images are sampled at random** (fixed `selection_seed`) from all eligible ones,
  rather than taking the earliest: the earliest would also be the ones nearest session 1
  in time, mixing "category" with "when scanned".
* **First viewing.** Session 1's trials are almost all first viewings, so using first
  viewings here keeps the comparison with the session-1 baseline about the category,
  not about familiarity with the image.

Torch-free, so the tests run anywhere.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from .assets import BEHAV_COCO_IDX, BEHAV_GLOBAL_TRIAL, BEHAV_SESSION, BEHAV_SUBJECT
from .categories import RULES, ObjectTable, load_object_table, memberships
from .trials import TrialPool, prepare_trial_pool
from .utils import log, write_json

KINDS = ("category",)
VIEWINGS = ("first",)
N_BEHAV_COLS = 17


@dataclass(frozen=True)
class TrainSetSpec:
    kind: str = "category"
    category: str = "animal"
    rule: str = "only"
    n_images: int = 700
    viewing: str = "first"
    selection_seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainSetSpec":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"train_set: unknown keys {sorted(unknown)}; "
                             f"allowed: {sorted(cls.__dataclass_fields__)}")
        spec = cls(**d)
        if spec.kind not in KINDS:
            raise ValueError(f"train_set.kind must be one of {KINDS}, got {spec.kind!r}")
        if spec.rule not in RULES:
            raise ValueError(f"train_set.rule must be one of {RULES}, got {spec.rule!r}")
        if spec.viewing not in VIEWINGS:
            raise ValueError(f"train_set.viewing must be one of {VIEWINGS}, got {spec.viewing!r}")
        if int(spec.n_images) < 1:
            raise ValueError("train_set.n_images must be positive")
        return spec

    @property
    def tag(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "", self.category.lower())
        return (f"cat-{slug}-{self.rule}-{int(self.n_images)}-{self.viewing}"
                f"-s{int(self.selection_seed)}")

    def describe(self) -> str:
        return (f"{int(self.n_images)} different '{self.category}' images (rule: {self.rule}), "
                f"{self.viewing} viewing of each, from all sessions")


def category_index(table: ObjectTable, name: str) -> int:
    wanted = name.strip().lower()
    for k, n in enumerate(table.supercategory_names):
        if n.lower() == wanted:
            return k
    raise KeyError(f"No COCO supercategory {name!r}. Known: {table.supercategory_names}")


def select_category_trials(pool: TrialPool, table: ObjectTable, spec: TrainSetSpec,
                           subj: int = 1) -> tuple[np.ndarray, dict]:
    """Pick the images and the trial of each to train on.

    Returns an (n_images, 17) behav table in MindEye2's layout (only the columns the
    pipeline reads are filled; the rest are -1, MindEye2's "missing") and a record of
    exactly what was chosen.
    """
    k = category_index(table, spec.category)
    member = memberships(table)[spec.rule][:, k]
    candidates = pool.image_ids[member[pool.image_ids]]
    test_ids = pool.test_image_ids[member[pool.test_image_ids]]
    n = int(spec.n_images)
    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} different '{spec.category}' images under rule "
            f"'{spec.rule}' for subj{subj:02d}; train_set.n_images asks for {n}."
        )
    rng = np.random.default_rng(int(spec.selection_seed))
    chosen = np.sort(rng.choice(candidates, size=n, replace=False))

    picks = np.array([pool.trials_of(int(c))[0] for c in chosen], dtype=np.int64)  # first
    behav = np.full((n, N_BEHAV_COLS), -1, dtype=np.float32)
    behav[:, BEHAV_COCO_IDX] = pool.coco_ids[picks]
    behav[:, BEHAV_SUBJECT] = subj
    behav[:, BEHAV_SESSION] = pool.sessions[picks]
    behav[:, BEHAV_GLOBAL_TRIAL] = pool.global_trials[picks]

    sessions, per_session = np.unique(pool.shards[picks] + 1, return_counts=True)
    info = {
        "spec": asdict(spec),
        "tag": spec.tag,
        "description": spec.describe(),
        "n_candidates": int(len(candidates)),
        "n_selected": n,
        "trials_per_session": {int(s): int(c) for s, c in zip(sessions, per_session)},
        "test_images_in_category": int(len(test_ids)),
        "selected_image_ids": chosen.tolist(),
        "selected_global_trials": pool.global_trials[picks].tolist(),
        "category_test_image_ids": test_ids.tolist(),
    }
    return behav, info


def selection_path(ws, subj: int, tag: str):
    return ws["data"] / f"subj{subj:02d}_{tag}_selection.json"


def prepare_train_set(ws, subj: int, spec: TrainSetSpec) -> tuple[np.ndarray, dict]:
    """Build (or rebuild; it is deterministic and cheap) the training trials for `spec`."""
    table, _ = load_object_table(ws)
    pool, _ = prepare_trial_pool(ws, subj=subj)
    behav, info = select_category_trials(pool, table, spec, subj=subj)
    write_json(selection_path(ws, subj, spec.tag), info)
    log.info("training set %s: %s; %d candidates, %d test images in the category",
             spec.tag, spec.describe(), info["n_candidates"], info["test_images_in_category"])
    return behav, info
