"""Step 3: did training on one category help on that category?

Compares a category-trained run (e.g. 700 animal-only images) with the session-1 run of
the same arm and seed, image by image, on the same test images:

    category test images  the test images in the training category (179 animal-only)
    other test images     the rest of the 1,000
    all test images       all 1,000

Both runs already saved one score per test image (`per_sample_metrics.npz`). Each run
numbers its images by its own image cache, so rows are first mapped to NSD image ids
and paired on those. The statistics are the paper's own: paired t-test, Wilcoxon
signed-rank, paired BCa bootstrap interval and Cohen's d_z (stats.py).

One extra view, **within-category two-way**: the saved two-way score asks "can the
model tell this image from the other 999?", and most of those 999 are not animals, so
it partly measures "is it an animal". Re-scoring the category images against each other
only (animal vs animal) asks the finer question. It is recomputed from the stored
predictions; nothing is retrained.

The primary comparison, fixed before looking at results, is two-way identification on
the category test images (the first row of the table).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .stats import cohens_dz, paired_bootstrap, paired_tests
from .utils import log, read_json, write_json

METRICS = ("two_way_clip", "cosine")


def nsd_ids_for_rows(ws, subj: int, tag: str, rows: np.ndarray) -> np.ndarray:
    """Image-cache rows of one training set -> NSD image ids."""
    index = np.load(ws["data"] / f"subj{subj:02d}_coco_idx_{tag}.npy")
    return index[np.asarray(rows, dtype=np.int64)]


def align(ids_a: np.ndarray, ids_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions in a and b of their common ids, in sorted id order."""
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    if len(np.unique(ids_a)) != len(ids_a) or len(np.unique(ids_b)) != len(ids_b):
        raise ValueError("an image appears twice in one run's per-image scores")
    common = np.intersect1d(ids_a, ids_b)
    pos_a = {int(x): i for i, x in enumerate(ids_a)}
    pos_b = {int(x): i for i, x in enumerate(ids_b)}
    ia = np.array([pos_a[int(x)] for x in common], dtype=np.int64)
    ib = np.array([pos_b[int(x)] for x in common], dtype=np.int64)
    return ia, ib, common


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """a - b on the same images: means, tests, effect size and a 95% interval."""
    out = paired_tests(a, b)
    boot = paired_bootstrap(a, b, seed=seed)
    out.update(ci_low=boot["ci_low"], ci_high=boot["ci_high"], cohens_dz=cohens_dz(a, b))
    return out


def subset_rows(scores_a: dict, scores_b: dict, ids: np.ndarray, groups: dict) -> list[dict]:
    """One paired comparison per (group, metric). `scores_*` are already aligned to `ids`."""
    rows = []
    for group, wanted in groups.items():
        mask = np.isin(ids, wanted)
        if mask.sum() < 3:
            raise ValueError(f"only {int(mask.sum())} images in group '{group}'")
        for metric in METRICS:
            r = paired(scores_a[metric][mask], scores_b[metric][mask])
            rows.append({"group": group, "metric": metric, **r})
    return rows


def _load_scores(run_dir: Path) -> dict:
    with np.load(run_dir / "per_sample_metrics.npz") as z:
        return {k: z[k] for k in z.files}


def within_category_two_way(run_dir: Path, ws, subj: int, tag: str,
                            category_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-way identification among the category images only. Returns (ids, scores)."""
    from .evaluate import PredictionStore, two_way_identification   # needs torch

    preds = PredictionStore.load(run_dir / "predictions")
    ids = nsd_ids_for_rows(ws, subj, tag, preds["rows"])
    order = np.argsort(ids)
    keep = order[np.isin(ids[order], category_ids)]
    emb = PredictionStore.embedding(preds)
    scores = two_way_identification(np.asarray(emb[keep], dtype=np.float32),
                                    np.asarray(preds["target"][keep], dtype=np.float32))
    return ids[keep], scores


def run_comparison(ws, cfg, baseline_tag: str = "1sess", arm: str | None = None,
                   seed: int = 0, within: bool = True) -> dict:
    from .trainsets import TrainSetSpec, selection_path

    if not cfg.train_set:
        raise ValueError("compare-category needs a config with a train_set (e.g. "
                         "configs/category_animal.yaml)")
    arm = arm or cfg.arms[0].name
    spec = TrainSetSpec.from_dict(cfg.train_set)
    tag = cfg.data_tag
    selection = read_json(selection_path(ws, cfg.subj, tag))
    if not selection:
        raise FileNotFoundError(f"no selection record for {tag}; run `assets` first")
    category_ids = np.asarray(selection["category_test_image_ids"], dtype=np.int64)

    run_cat = ws["runs"] / f"subj{cfg.subj:02d}_{tag}_{arm}_seed{seed}"
    run_base = ws["runs"] / f"subj{cfg.subj:02d}_{baseline_tag}_{arm}_seed{seed}"
    for d in (run_cat, run_base):
        if not (d / "per_sample_metrics.npz").exists():
            raise FileNotFoundError(f"{d.name} has no per_sample_metrics.npz; run "
                                    f"predict and evaluate for it first")

    sc, sb = _load_scores(run_cat), _load_scores(run_base)
    ia, ib, ids = align(nsd_ids_for_rows(ws, cfg.subj, tag, sc["rows"]),
                        nsd_ids_for_rows(ws, cfg.subj, baseline_tag, sb["rows"]))
    if len(ids) != len(sc["rows"]) or len(ids) != len(sb["rows"]):
        raise ValueError(f"the two runs were scored on different test images "
                         f"({len(sc['rows'])} vs {len(sb['rows'])}, {len(ids)} shared)")
    missing = np.setdiff1d(category_ids, ids)
    if len(missing):
        raise ValueError(f"{len(missing)} category test images are not in the scores")
    a = {m: sc[m][ia] for m in METRICS}
    b = {m: sb[m][ib] for m in METRICS}

    label = f"{spec.category} ({spec.rule})"
    groups = {f"{label} test images": category_ids,
              "other test images": np.setdiff1d(ids, category_ids),
              "all test images": ids}
    rows = subset_rows(a, b, ids, groups)

    if within:
        ids_c, wc = within_category_two_way(run_cat, ws, cfg.subj, tag, category_ids)
        ids_b, wb = within_category_two_way(run_base, ws, cfg.subj, baseline_tag,
                                            category_ids)
        if not np.array_equal(ids_c, ids_b):
            raise ValueError("within-category images differ between the runs")
        rows.append({"group": f"{label} vs each other only", "metric": "two_way_clip",
                     **paired(wc, wb)})

    # Context: how much the baseline itself moves between seeds, per group.
    seed_spread = {}
    for s in sorted({0, 1, 2}):
        d = ws["runs"] / f"subj{cfg.subj:02d}_{baseline_tag}_{arm}_seed{s}"
        if (d / "per_sample_metrics.npz").exists():
            sx = _load_scores(d)
            _, ix, idx = align(ids, nsd_ids_for_rows(ws, cfg.subj, baseline_tag, sx["rows"]))
            two = sx["two_way_clip"][ix]
            seed_spread[s] = {g: float(two[np.isin(idx, w)].mean()) for g, w in groups.items()}

    return {
        "category_run": run_cat.name,
        "baseline_run": run_base.name,
        "train_set": spec.describe(),
        "n_category_test_images": int(len(category_ids)),
        "primary": rows[0],
        "rows": rows,
        "baseline_two_way_by_seed": seed_spread,
    }


def format_result(res: dict) -> str:
    lines = [
        f"category run : {res['category_run']}",
        f"baseline run : {res['baseline_run']}",
        f"training set : {res['train_set']}",
        "",
        "difference = category-trained minus session-1, on the same images "
        "(positive = category training is better)",
        "",
        f"  {'test images':34s} {'metric':13s} {'n':>5s} {'category':>9s} {'session1':>9s} "
        f"{'diff':>8s}  {'95% CI':>19s} {'p (t)':>9s} {'p (W)':>9s} {'d_z':>6s}",
    ]
    for r in res["rows"]:
        lines.append(
            f"  {r['group'][:34]:34s} {r['metric']:13s} {r['n']:>5d} {r['mean_a']:>9.4f} "
            f"{r['mean_b']:>9.4f} {r['mean_diff']:>+8.4f}  "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {r['t_p']:>9.2g} "
            f"{r['wilcoxon_p']:>9.2g} {r['cohens_dz']:>+6.2f}")
    p = res["primary"]
    verdict = ("better" if p["ci_low"] > 0 else "worse" if p["ci_high"] < 0
               else "not distinguishable from session-1")
    lines += ["", f"primary (fixed in advance): two-way on the {p['group']} -> category "
                  f"training is {verdict} (diff {p['mean_diff']:+.4f}, 95% CI "
                  f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}])"]
    if res["baseline_two_way_by_seed"]:
        lines += ["", "session-1 two-way by seed (how much the baseline moves on its own):"]
        for s, g in res["baseline_two_way_by_seed"].items():
            lines.append("  seed " + str(s) + ": " +
                         " · ".join(f"{k.split(' test')[0]} {v:.4f}" for k, v in g.items()))
    return "\n".join(lines)


def save_result(ws, res: dict) -> Path:
    path = ws["results"] / "category_comparison.json"
    write_json(path, res)
    log.info("comparison -> %s", path)
    return path


# --------------------------------------------------------------------------------------
# reconstructions of the category test images, both runs
# --------------------------------------------------------------------------------------
def category_positions(ws, subj: int, tag: str, run_dir: Path, category_ids: np.ndarray,
                       n: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Positions in a run's prediction store of the category test images, ordered by
    NSD id. With `n`, a fixed random subset (the same for every run, by `seed`)."""
    rows = np.load(run_dir / "predictions" / "rows.npy", mmap_mode="r")
    ids = nsd_ids_for_rows(ws, subj, tag, np.asarray(rows))
    wanted = np.sort(np.asarray(category_ids, dtype=np.int64))
    if n is not None and n < len(wanted):
        wanted = np.sort(np.random.default_rng(seed).choice(wanted, size=n, replace=False))
    pos_of = {int(x): i for i, x in enumerate(ids)}
    missing = [int(x) for x in wanted if int(x) not in pos_of]
    if missing:
        raise ValueError(f"{len(missing)} category test images missing from {run_dir.name}")
    return np.array([pos_of[int(x)] for x in wanted], dtype=np.int64), wanted


def image_metric_rows(scores_a: dict, scores_b: dict) -> list[dict]:
    """Paired comparison per image metric (a = category-trained, b = session-1)."""
    from .metric_meta import HIGHER_IS_BETTER, IMAGE_METRICS

    rows = []
    for metric in IMAGE_METRICS:
        if metric not in scores_a or metric not in scores_b:
            continue
        a, b = np.asarray(scores_a[metric], float), np.asarray(scores_b[metric], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        r = paired(a[ok], b[ok])
        r.update(metric=metric, higher_is_better=bool(HIGHER_IS_BETTER[metric]))
        rows.append(r)
    return rows


def recon_category(ws, cfg, baseline_tag: str = "1sess", arm: str | None = None,
                   seed: int = 0, n: int | None = None, num_steps: int = 38,
                   device: str = "cuda", force: bool = False, figure_n: int = 12) -> dict:
    """Decode the category test images from both runs, score and compare them."""
    import torch

    from .assets import load_asset_meta
    from .evaluate import ImageMetricSuite
    from .recon import load_ground_truth, reconstruct_for_run
    from .report import figure_qualitative
    from .trainsets import selection_path

    if not cfg.train_set:
        raise ValueError("recon-category needs a config with a train_set")
    arm = arm or cfg.arms[0].name
    tag = cfg.data_tag
    selection = read_json(selection_path(ws, cfg.subj, tag))
    if not selection:
        raise FileNotFoundError(f"no selection record for {tag}; run `assets` first")
    category_ids = np.asarray(selection["category_test_image_ids"], dtype=np.int64)

    runs = {"session-1": (baseline_tag, ws["runs"] / f"subj{cfg.subj:02d}_{baseline_tag}_{arm}_seed{seed}"),
            "category-trained": (tag, ws["runs"] / f"subj{cfg.subj:02d}_{tag}_{arm}_seed{seed}")}
    recons, ids_by_run = {}, {}
    for label, (run_tag, run_dir) in runs.items():
        if not (run_dir / "predictions" / "meta.json").exists():
            raise FileNotFoundError(f"{run_dir.name} has no predictions; run `predict` first")
        out = run_dir / f"reconstructions_{tag}.pt"
        pos, ids = category_positions(ws, cfg.subj, run_tag, run_dir, category_ids, n=n)
        if force or not out.exists():
            log.info("decoding %d %s test images for %s", len(pos), tag, run_dir.name)
            reconstruct_for_run(run_dir / "predictions", out, ws, n_images=len(pos),
                                decoder="sdxl_unclip", device=device, num_steps=num_steps,
                                batch_size=1, positions=pos)
        d = torch.load(out, map_location="cpu", weights_only=False)
        got = nsd_ids_for_rows(ws, cfg.subj, run_tag, d["rows"].numpy())
        if not np.array_equal(got, ids):
            raise ValueError(f"{out.name} holds other images than requested; rerun with --force")
        recons[label], ids_by_run[label] = d["recons"], got

    if not np.array_equal(*ids_by_run.values()):
        raise ValueError("the two runs reconstructed different images")
    ids = ids_by_run["category-trained"]

    meta = load_asset_meta(ws, cfg.subj, tag)
    cache = np.load(ws["data"] / f"subj{cfg.subj:02d}_coco_idx_{tag}.npy")
    row_of = {int(x): i for i, x in enumerate(cache)}
    gt = load_ground_truth(meta["files"]["images"], np.array([row_of[int(x)] for x in ids]),
                           size=recons["category-trained"].shape[-1])

    suite = ImageMetricSuite(device=device)
    scores = {label: suite(r, gt) for label, r in recons.items()}
    rows = image_metric_rows(scores["category-trained"], scores["session-1"])

    figdir = ws["figures"]
    figdir.mkdir(parents=True, exist_ok=True)
    k = min(figure_n, len(ids))
    pick = torch.as_tensor(np.linspace(0, len(ids) - 1, k).round().astype(np.int64))
    fig = figure_qualitative(gt[pick].numpy(),
                             {lbl: recons[lbl][pick].numpy() for lbl in recons},
                             figdir / "category_reconstructions.png", n=k)
    res = {"category_run": runs["category-trained"][1].name,
           "baseline_run": runs["session-1"][1].name,
           "n_images": int(len(ids)), "image_ids": ids.tolist(),
           "rows": rows, "figure": str(fig)}
    write_json(ws["results"] / "category_reconstructions.json", res)
    return res


def format_recon_result(res: dict) -> str:
    lines = [
        f"reconstructions of {res['n_images']} category test images, both runs",
        f"  category run : {res['category_run']}",
        f"  baseline run : {res['baseline_run']}",
        "",
        "diff = category-trained minus session-1 on the same images; 'better' accounts "
        "for metrics where lower is better",
        "",
        f"  {'metric':10s} {'category':>9s} {'session1':>9s} {'diff':>8s}  {'95% CI':>19s} "
        f"{'p (t)':>9s} {'p (W)':>9s}  verdict",
    ]
    for r in res["rows"]:
        sign = 1 if r["higher_is_better"] else -1
        lo, hi = sorted((sign * r["ci_low"], sign * r["ci_high"]))
        verdict = "category better" if lo > 0 else "session-1 better" if hi < 0 else "no clear difference"
        lines.append(
            f"  {r['metric']:10s} {r['mean_a']:>9.4f} {r['mean_b']:>9.4f} {r['mean_diff']:>+8.4f}  "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {r['t_p']:>9.2g} {r['wilcoxon_p']:>9.2g}  "
            f"{verdict}{'' if r['higher_is_better'] else '  (lower is better)'}")
    lines += ["", f"figure -> {res['figure']}"]
    return "\n".join(lines)
