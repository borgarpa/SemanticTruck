"""Scoring the bay split against a human segmentation.

Manuscript §2.4 splits a B-double's cargo into its two stacks with k-means and then repairs the
split with DBSCAN. §3.3 reports how well that worked. This module is where that number comes from.

The reference rides through the pipeline as a per-point field (`run.BAY_FIELD`) rather than being
matched back afterwards, because by the time bays exist the cloud has been deduplicated, decimated,
outlier-filtered and rotated, and there is no correspondence left to the rows of the file. What
reaches this module is the confusion matrix the pipeline counted: `ref_bay_<g>` per bay and
`ref_total_<g>` per load. Everything here is derived from those counts, so any other matching or
metric can be recomputed from a written table without re-running anything.

Three things about the score are easy to get wrong:

* **The totals are taken over the whole outlier-filtered cargo, before the split.** Points that
  `clean_bays` discards are therefore in the denominator and count as errors. Scoring only the
  points that ended up in some bay would quietly forgive exactly the failure the step can have.
* **Bay ids are matched, not compared.** `Cargo.shift_cluster_labels` numbers the low-Y stack 0,
  while a hand-segmented file starts from whichever end the annotator did -- so on some loads the
  two run in opposite directions and comparing labels as they stand would report near-zero accuracy
  on an identical split. The pairing that agrees best is used (Hungarian assignment) and
  `ids_swapped` records when it was not the identity.
* **Single-bay loads get no row at all**, by construction rather than by omission. They never reach
  k-means; their `bay` field calls the whole cargo bay 1 and is not a segmentation of anything.

One caveat when quoting the figures: where a reference was produced by hand-*correcting* this
pipeline's own output rather than drawn from scratch, the two coincide by construction wherever the
annotator agreed and left it alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from semantictruck.analysis.reference import campaign_of, config_of
from semantictruck.pipeline.loads import MULTI_BAY_TAG, is_multi_bay

# What `--bay-truth` takes to mean "every multi-bay load" and "score nothing at all"
ALL_BAY_TRUTH = 'all'
NO_BAY_TRUTH = 'none'

# Why a multi-bay load has no score. Kept on the row, so the table says what is missing and why
# rather than leaving a reader to infer it from a blank.
REASON_EXCLUDED = 'excluded by --bay-truth, so its bay field is not read as a reference'
REASON_NO_REFERENCE = 'no bay reference on the cloud'
REASON_FAILED = 'the run produced no bays'


def resolve_bay_truth(chosen, load_ids, tag: str = MULTI_BAY_TAG) -> tuple:
    """Which loads' `bay` field is read as a reference: every multi-bay one, none, or a list."""
    if list(chosen) == [NO_BAY_TRUTH]:
        return ()
    if list(chosen) == [ALL_BAY_TRUTH]:
        return tuple(name for name in load_ids if is_multi_bay(name, tag))

    unknown = [name for name in chosen if name not in load_ids]
    if unknown:
        raise SystemExit('--bay-truth names load(s) not being run: {}'.format(', '.join(unknown)))
    return tuple(chosen)


def bay_detection_table(per_bay: pd.DataFrame, truth_loads,
                        tag: str = MULTI_BAY_TAG) -> pd.DataFrame:
    """Score the bay split of every multi-bay load, one row per (arm, load)."""
    rows = []
    carried = [k for k in ('mesh_method', 'setting_id') if k in per_bay.columns]

    for (arm, load_id), frame in per_bay.groupby(['segmentation', 'load_id'], sort=True):
        if not is_multi_bay(load_id, tag):
            continue

        row = {'segmentation': arm, 'load_id': load_id,
               'campaign': campaign_of(load_id), 'config': config_of(load_id)}
        for key in carried:
            row[key] = frame.iloc[0][key]

        confusion, totals, reason = _confusion(frame)
        if load_id not in truth_loads:
            reason = REASON_EXCLUDED

        # `reference` says whether the row was scored, so it cannot claim one the row has not used
        row['reference'] = 'none' if reason else 'manual'
        row['n_bays_pred'] = 0 if confusion is None else int(confusion.shape[0])
        rows.append(dict(row, reason=reason) if reason
                    else dict(row, **score_bay_split(confusion, totals)))

    return pd.DataFrame(rows)


def _numbered(columns, prefix: str) -> list:
    """The `<prefix><id>` columns in id order -- numeric, so bay 10 does not sort before bay 2."""
    return [prefix + str(i) for i in
            sorted(int(c[len(prefix):]) for c in columns if c.startswith(prefix))]


def _confusion(frame: pd.DataFrame):
    """One load's confusion matrix and reference totals, or a reason there is none.

    `confusion[p, g]` is how many points the pipeline put in its bay `p` that the reference calls
    bay `g`, column 0 being the points the reference places in no bay -- vehicle points the arm
    took for cargo, which is where an automated arm differs from a human one.
    """
    counts = _numbered(frame.columns, 'ref_bay_')
    totals = _numbered(frame.columns, 'ref_total_')
    if not counts or not totals:
        return None, None, REASON_NO_REFERENCE

    frame = frame.sort_values('bay_id')
    confusion = frame[counts].to_numpy(dtype=float)
    reference = frame[totals].to_numpy(dtype=float)[0]
    if np.isnan(confusion).any() or np.isnan(reference).any():
        return None, None, REASON_NO_REFERENCE
    if not len(confusion):
        return None, None, REASON_FAILED

    return confusion.astype(np.int64), reference.astype(np.int64), ''


def score_bay_split(confusion: np.ndarray, totals: np.ndarray) -> dict:
    """Accuracy and per-bay IoU of one split, under the pairing of ids that agrees best.

    Restricted to the points the reference places in a bay: its column 0 has nothing to be right or
    wrong about. Those points are reported as `n_unreferenced` instead.
    """
    labelled = confusion[:, 1:]                 # predicted bay x reference bay, both 1-based
    reference = totals[1:]
    n_reference = int(reference.sum())

    # Hungarian on the negated agreement. With two bays it is simply the better of the two
    # pairings, and it stays correct if the number of bays is ever raised.
    pred_ids, ref_ids = linear_sum_assignment(-labelled)
    intersections = labelled[pred_ids, ref_ids]
    assigned = labelled.sum(axis=1)             # points of each predicted bay that have a reference

    ious = {}
    for pred_id, ref_id, intersection in zip(pred_ids, ref_ids, intersections):
        union = assigned[pred_id] + reference[ref_id] - intersection
        ious['iou_bay_{}'.format(ref_id + 1)] = float(intersection / union) if union else np.nan

    # In reference-bay order, not in the order the matching happened to pair them up
    ious = {key: ious[key] for key in _numbered(ious, 'iou_bay_')}
    values = [v for v in ious.values() if not np.isnan(v)]

    return dict(
        n_ref_points=n_reference,
        # Everything the reference placed in a bay that the split did not: `clean_bays` drops the
        # blobs it cannot attach to either stack, and they are errors like any other
        n_unassigned=int(n_reference - assigned.sum()),
        n_unreferenced=int(confusion[:, 0].sum()),
        bay_acc=float(intersections.sum() / n_reference) if n_reference else np.nan,
        bay_miou=float(np.mean(values)) if values else np.nan,
        **ious,
        ids_swapped=bool((pred_ids != ref_ids).any()),
        matching=' '.join('bay {} -> reference {}'.format(p, g + 1)
                          for p, g in zip(pred_ids, ref_ids)),
        reason='')


def attach_bay_scores(per_truck: pd.DataFrame, bays: pd.DataFrame) -> pd.DataFrame:
    """Carry the split's accuracy and mIoU onto the per-truck table, blank where there is none."""
    columns = [c for c in ('bay_acc', 'bay_miou') if c in bays.columns]
    if bays.empty or not columns:
        return per_truck
    return per_truck.merge(bays[['segmentation', 'load_id'] + columns],
                           on=['segmentation', 'load_id'], how='left')
