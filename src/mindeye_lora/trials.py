"""The subject's whole trial history, grouped by image.

The main experiment fine-tunes on the 688 training trials of one session. The planned
category and repeat experiments instead choose *which* trials to train on from every
session the subject completed (for example a third of the images with all three of
their repeats, or images of one category seen once each) while keeping the trial count
at 688. Both need to know, for every image, every trial in which the subject saw it.
This module builds that table and nothing else: no trials are selected here.

Only behavioural tables are downloaded: the webdataset shards, ~9 MB per session
(~350 MB for subj01's 40 sessions). Voxels and images for whichever trials are selected
later are fetched row by row, the same way `assets.py` does for one session.

Leakage guard: the pool is built from the *train* shards only, and any image that also
appears in the test shard is dropped and counted. MindEye2's split already keeps the
shared-1000 test images out of the train shards, so the expected count is zero; a
non-zero count is logged as a warning because it means the split is not what we think.

Torch-free on purpose, so the tests run anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np

from .assets import (
    BEHAV_COCO_IDX,
    BEHAV_GLOBAL_TRIAL,
    BEHAV_SESSION,
    WDS_TRAIN_TPL,
    hf_download,
    read_behav_tar,
    repo_files,
    resolve_test_shard,
)
from .utils import log, progress, read_json, write_json

# Sessions each NSD participant completed, matching upstream Train.ipynb's
# `nsessions_allsubj`. Subjects 3, 4, 6 and 8 did not finish all 40.
NSESSIONS_PER_SUBJ = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}


@dataclass
class TrialPool:
    """Every usable training trial, sorted by (image, shard, global trial).

    Per trial:
        coco_ids       COCO-73k index of the image shown
        shards         0-based webdataset shard = session order (shard 0 is session 1)
        sessions       behav column 2, kept only as a cross-check against `shards`
        global_trials  row in `betas_all_subjXX_fp32_renorm.hdf5`
    Per image (sorted by COCO id):
        image_ids      unique COCO-73k ids
        image_counts   how many trials showed that image
        image_offsets  where its trials start, so image i's trials are
                       ``image_offsets[i] : image_offsets[i] + image_counts[i]``
    Held out:
        test_image_ids the test-shard images (sorted); never part of the pool
    """

    coco_ids: np.ndarray
    shards: np.ndarray
    sessions: np.ndarray
    global_trials: np.ndarray
    image_ids: np.ndarray
    image_counts: np.ndarray
    image_offsets: np.ndarray
    test_image_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.coco_ids)

    @property
    def n_images(self) -> int:
        return len(self.image_ids)

    def trials_of(self, coco_id: int) -> np.ndarray:
        """Indices into the per-trial arrays for every trial that showed `coco_id`."""
        i = int(np.searchsorted(self.image_ids, coco_id))
        if i >= len(self.image_ids) or int(self.image_ids[i]) != int(coco_id):
            raise KeyError(f"COCO id {coco_id} is not in the trial pool")
        start = int(self.image_offsets[i])
        return np.arange(start, start + int(self.image_counts[i]))

    def images_with(self, n_trials: int) -> np.ndarray:
        """COCO ids of the images the subject saw exactly `n_trials` times."""
        return self.image_ids[self.image_counts == n_trials]

    # -- consistency ----------------------------------------------------------------
    def validate(self) -> None:
        n = len(self.coco_ids)
        for name in ("shards", "sessions", "global_trials"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"TrialPool.{name} has {len(getattr(self, name))} rows, "
                                 f"expected {n}")
        if not (len(self.image_ids) == len(self.image_counts) == len(self.image_offsets)):
            raise ValueError("TrialPool per-image arrays disagree in length")
        if int(self.image_counts.sum()) != n:
            raise ValueError("TrialPool image_counts do not add up to the number of trials")
        if len(self.image_ids) > 1 and not np.all(np.diff(self.image_ids) > 0):
            raise ValueError("TrialPool image_ids must be strictly increasing")
        expected = np.repeat(self.image_ids, self.image_counts)
        if not np.array_equal(expected, self.coco_ids):
            raise ValueError("TrialPool trials are not grouped by image")
        if len(np.unique(self.global_trials)) != n:
            raise ValueError("TrialPool contains a global trial more than once")
        if np.isin(self.test_image_ids, self.image_ids).any():
            raise ValueError("TrialPool contains a test image")

    # -- persistence ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write atomically, then read back: Drive uploads asynchronously, so a file
        that finished writing locally can still be truncated in the cloud."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as fh:   # an open handle stops np.savez appending ".npz"
            np.savez(fh, **{f.name: getattr(self, f.name) for f in fields(self)})
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        TrialPool.load(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrialPool":
        with np.load(path) as z:
            missing = [f.name for f in fields(cls) if f.name not in z.files]
            if missing:
                raise ValueError(f"{path} is missing {missing}; rebuild it with "
                                 f"`python -m mindeye_lora.cli trials --force`")
            pool = cls(**{f.name: z[f.name].astype(np.int64) for f in fields(cls)})
        pool.validate()
        return pool


def _as_int(col: np.ndarray) -> np.ndarray:
    # behav tables are float32; ids up to ~73k and trials up to ~30k are exact in float32
    return np.rint(np.asarray(col, dtype=np.float64)).astype(np.int64)


def build_trial_pool(
    train_shards: Sequence[np.ndarray],
    test_behav: np.ndarray | None = None,
) -> tuple[TrialPool, dict]:
    """Group every trial of every train shard by the image it showed.

    `train_shards[k]` is the (N_k, 17) behav table of shard k, i.e. session k + 1.
    Returns the pool and a report of what was dropped and why.
    """
    coco, shard, sess, glob = [], [], [], []
    for k, behav in enumerate(train_shards):
        behav = np.asarray(behav)
        if behav.ndim != 2 or behav.shape[1] <= max(BEHAV_COCO_IDX, BEHAV_SESSION,
                                                    BEHAV_GLOBAL_TRIAL):
            raise ValueError(f"shard {k}: expected an (N, 17) behav table, got {behav.shape}")
        coco.append(_as_int(behav[:, BEHAV_COCO_IDX]))
        sess.append(_as_int(behav[:, BEHAV_SESSION]))
        glob.append(_as_int(behav[:, BEHAV_GLOBAL_TRIAL]))
        shard.append(np.full(len(behav), k, dtype=np.int64))

    def cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    coco, shard, sess, glob = cat(coco), cat(shard), cat(sess), cat(glob)
    report = {"n_raw_trials": int(len(coco))}

    # -1 marks a missing value in MindEye2's behav tables
    valid = (coco >= 0) & (glob >= 0)
    report["dropped_invalid_trials"] = int((~valid).sum())

    if test_behav is not None and len(test_behav):
        test_ids = np.unique(_as_int(np.asarray(test_behav)[:, BEHAV_COCO_IDX]))
        test_ids = test_ids[test_ids >= 0]
    else:
        test_ids = np.zeros(0, dtype=np.int64)
    in_test = valid & np.isin(coco, test_ids)
    report["dropped_test_overlap_trials"] = int(in_test.sum())
    report["dropped_test_overlap_images"] = int(len(np.unique(coco[in_test])))
    report["n_test_images"] = int(len(test_ids))
    if in_test.any():
        log.warning(
            "%d train trials (%d images) also appear in the test shard and were dropped. "
            "MindEye2's split should make this zero -- check which shards were read.",
            report["dropped_test_overlap_trials"], report["dropped_test_overlap_images"],
        )

    keep = valid & ~in_test
    coco, shard, sess, glob = coco[keep], shard[keep], sess[keep], glob[keep]

    uniq, counts = np.unique(glob, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"{int((counts > 1).sum())} global trial ids appear more than once "
            f"(e.g. {int(uniq[counts > 1][0])}); a shard was probably read twice."
        )

    order = np.lexsort((glob, shard, coco))     # last key is primary: image, then time
    coco, shard, sess, glob = coco[order], shard[order], sess[order], glob[order]
    image_ids, image_offsets, image_counts = np.unique(coco, return_index=True,
                                                        return_counts=True)
    pool = TrialPool(
        coco_ids=coco, shards=shard, sessions=sess, global_trials=glob,
        image_ids=image_ids.astype(np.int64),
        image_counts=image_counts.astype(np.int64),
        image_offsets=image_offsets.astype(np.int64),
        test_image_ids=test_ids.astype(np.int64),
    )
    pool.validate()
    report["n_trials"] = len(pool)
    report["n_images"] = pool.n_images
    return pool, report


def summarise_pool(pool: TrialPool, report: dict | None = None) -> dict:
    """Facts about the pool that the trial-selection design depends on."""
    hist = dict(zip(*(a.tolist() for a in np.unique(pool.image_counts, return_counts=True))))
    per_shard = np.bincount(pool.shards) if len(pool) else np.zeros(0, dtype=np.int64)

    # Does behav's own session column agree with the order the shards were read in?
    offsets = np.unique(pool.sessions - pool.shards)
    if len(offsets) == 1:
        session_column = f"consistent: session = shard + {int(offsets[0])}"
    else:
        session_column = f"INCONSISTENT: {len(offsets)} different session-shard offsets"

    # How spread out are an image's repeats? Matters for the repeat condition: three
    # repeats from three different sessions are not the same as three in one session.
    pairs = np.unique(np.stack([pool.coco_ids, pool.shards], 1), axis=0) if len(pool) \
        else np.zeros((0, 2), dtype=np.int64)
    sessions_per_image = np.bincount(np.searchsorted(pool.image_ids, pairs[:, 0]),
                                     minlength=pool.n_images)
    repeated = pool.image_counts > 1

    return {
        "n_trials": int(len(pool)),
        "n_images": int(pool.n_images),
        "n_sessions": int(len(per_shard)),
        "trials_per_image": {int(k): int(v) for k, v in hist.items()},
        "trials_per_session": {
            "first_session": int(per_shard[0]) if len(per_shard) else 0,
            "min": int(per_shard.min()) if len(per_shard) else 0,
            "max": int(per_shard.max()) if len(per_shard) else 0,
        },
        "session_column": session_column,
        "repeats": {
            "images_seen_more_than_once": int(repeated.sum()),
            "all_repeats_in_one_session": int((repeated & (sessions_per_image == 1)).sum()),
            "median_sessions_spanned": float(np.median(sessions_per_image[repeated]))
            if repeated.any() else 0.0,
        },
        "dropped": dict(report or {}),
    }


def format_summary(summary: dict) -> str:
    tpi = summary["trials_per_image"]
    tps = summary["trials_per_session"]
    rep = summary["repeats"]
    drop = summary.get("dropped", {})
    head = "trial pool"
    if "subj" in summary:
        head += f" · subj{summary['subj']:02d}"
    head += f" · {summary['n_sessions']} sessions"
    lines = [
        head,
        f"  trials             {summary['n_trials']:,}  (session 1: {tps['first_session']:,}; "
        f"per session {tps['min']:,}-{tps['max']:,})",
        f"  images             {summary['n_images']:,}",
        "  trials per image   " + ", ".join(f"{k}x: {v:,} images" for k, v in sorted(tpi.items())),
        f"  session column     {summary['session_column']}",
        f"  repeats            {rep['images_seen_more_than_once']:,} images seen more than once; "
        f"{rep['all_repeats_in_one_session']:,} with every repeat in one session; "
        f"median {rep['median_sessions_spanned']:g} sessions spanned",
        f"  test overlap       {drop.get('dropped_test_overlap_images', 0):,} images dropped "
        f"(test set: {drop.get('n_test_images', 0):,} images)",
        f"  invalid rows       {drop.get('dropped_invalid_trials', 0):,} dropped",
    ]
    return "\n".join(lines)


def pool_paths(ws, subj: int, n_sessions: int) -> tuple[Path, Path]:
    stem = f"subj{subj:02d}_trial_pool_{n_sessions}sess"
    return ws["data"] / f"{stem}.npz", ws["data"] / f"{stem}_summary.json"


def prepare_trial_pool(
    ws,
    subj: int = 1,
    max_sessions: int | None = None,
    force: bool = False,
) -> tuple[TrialPool, dict]:
    """Download every session's behav table for `subj` and build the trial pool.

    Resumable: each shard is cached under `assets/` as it downloads (session 1's shard
    is already there from `cli assets`), so an interrupted run re-reads rather than
    re-downloads.
    """
    if subj not in NSESSIONS_PER_SUBJ:
        raise ValueError(f"subj must be one of {sorted(NSESSIONS_PER_SUBJ)}, got {subj}")
    completed = NSESSIONS_PER_SUBJ[subj]
    n = completed if max_sessions is None else int(max_sessions)
    if not 1 <= n <= completed:
        raise ValueError(f"subj{subj:02d} completed {completed} sessions; "
                         f"max_sessions must be between 1 and {completed}, got {n}")

    pool_path, summary_path = pool_paths(ws, subj, n)
    if pool_path.exists() and summary_path.exists() and not force:
        try:
            pool, summary = TrialPool.load(pool_path), read_json(summary_path)
            if summary:
                log.info("Trial pool already built for subj%02d / %d sessions.", subj, n)
                return pool, summary
        except Exception as exc:
            log.warning("cached trial pool unusable (%s); rebuilding", exc)

    files = repo_files()
    names = [WDS_TRAIN_TPL.format(subj=subj, shard=k) for k in range(n)]
    missing = [name for name in names if name not in set(files)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} shard(s) not on the hub, e.g. {missing[0]}")

    shards = [read_behav_tar(hf_download(name, ws["assets"]))
              for name in progress(names, desc=f"behav subj{subj:02d}", unit="session")]
    test_behav = read_behav_tar(hf_download(resolve_test_shard(subj, files), ws["assets"]))

    pool, report = build_trial_pool(shards, test_behav)
    summary = {"subj": subj, **summarise_pool(pool, report)}
    pool.save(pool_path)
    write_json(summary_path, summary)
    ws.mark_done(f"trials_subj{subj:02d}_{n}sess", n_trials=len(pool), n_images=pool.n_images)
    log.info("trial pool -> %s", pool_path)
    return pool, summary
