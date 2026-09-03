"""Running the sweep: which runs happen, in what order, and how their cost is kept down.

The work is ordered **by truckload, not by setting**. One load is read once and every setting runs
on it, which is what makes the Stage 1 cache pay: most settings differ from their campaign's BASE
only after the aligned bay clouds exist, so those are computed once per (load, arm) and reused, and
the LAZ file is read once instead of dozens of times.

**Deliberately single-process.** Open3D's outlier removal and normal orientation already thread
across every core, and running loads side by side only makes them contend -- the largest bay peaks
near 10 GB on its own. One load at a time gets the whole machine and a predictable footprint.

Per-run timeouts use `SIGALRM` rather than a process per run, so a timed-out setting is recorded and
the campaign moves on with its cache and its loaded cloud intact. Because that signal cannot
interrupt a call already inside Open3D's C++, there is also a **strike rule**: once a setting has
timed out on a few loads, the rest are recorded as timeouts without being run. Ball pivoting at
radii approaching the object scale takes over half an hour on a single load, and paying that on
every load and arm would cost more than the whole rest of the campaign.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from semantictruck.pipeline.loads import cloud_path, is_multi_bay
from semantictruck.pipeline.params import Params
from semantictruck.pipeline.run import LoadResult, run_load, stage1
from semantictruck.pipeline.status import STATUS_TIMEOUT
from semantictruck.pipeline.timeout import DEFAULT_TIMEOUT_S, Timeout, time_limit
from semantictruck.sensitivity.grid import BASE_ID, SENSITIVITY_BASE, Setting

# Loads a setting may time out on before the rest are recorded as timeouts without being run
DEFAULT_STRIKES = 3

# Settings that vary DBSCAN. They are scoped to multi-bay loads, and the single-bay path holds its
# own DBSCAN fixed (`pipeline.run.SINGLE_BAY_DBSCAN_EPS`), so a single-bay load's result under one
# of these is BASE's by construction -- copied rather than re-run.
DBSCAN_SETTINGS = ('P03', 'P04')


def stage1_groups(settings: list[Setting],
                  base: Params = SENSITIVITY_BASE) -> list[tuple[tuple, list[Setting]]]:
    """Settings grouped by the Stage 1 they share, the baseline group first.

    Stage 1 does not depend on the reconstruction, so a group spans all four campaigns. Running a
    whole group before moving on means each distinct Stage 1 is computed exactly once per
    (load, arm) -- fifteen times rather than eighty -- while still finishing every setting in the
    group across every truckload before anything else starts, which is what the per-setting
    regression needs.
    """
    baseline = base.stage1_key()
    groups: dict = {}
    for setting in settings:
        groups.setdefault(setting.params.stage1_key(), []).append(setting)

    ordered = sorted(groups, key=lambda k: (k != baseline, repr(k)))
    return [(key, sorted(groups[key], key=lambda s: (not s.is_base, s.mesh_method, s.setting_id)))
            for key in ordered]


def run_group(load_id: str, arm: str, group: list[Setting], cloud_dir: str,
              pred_dir: str | None = None, timeout_s: int = DEFAULT_TIMEOUT_S,
              wood_label: int = 0, verbose: bool = True, strikes=None,
              strike_limit: int = DEFAULT_STRIKES, base_rows=None,
              reference: dict | None = None) -> list[dict]:
    """One truckload, one arm, every setting sharing a Stage 1. Returns the raw per-bay rows.

    Stage 1 is computed once and reused by each setting in the group; only Stage 2 differs. The
    measured Stage 1 cost is added back into every setting's `runtime_s`, because each of them
    would have paid it had it run alone -- otherwise the first setting in a group would look
    expensive and the rest free.
    """
    laz_path = cloud_path(load_id, cloud_dir)
    rows: list[dict] = []
    cached = None
    stage1_s = 0.0

    for setting in group:
        if setting.skipped:
            rows.extend(_tag(_timed_out(load_id, arm, 'not run: ' + setting.skip_reason),
                             setting, reference))
            continue

        if _reuses_base(setting, load_id) and (base_rows or {}).get(setting.mesh_method):
            rows.extend(_copy_base(base_rows[setting.mesh_method], setting, reference))
            continue

        struck = _strikes_for(strikes, setting.key, arm)
        if struck >= strike_limit:
            rows.extend(_tag(_timed_out(
                load_id, arm,
                'not run: this setting timed out on {} earlier load(s)'.format(struck)),
                setting, reference))
            continue

        try:
            with time_limit(timeout_s):
                if cached is None:
                    started = time.perf_counter()
                    cached = stage1(laz_path, setting.params, arm, pred_dir, wood_label)
                    stage1_s = time.perf_counter() - started
                result = run_load(laz_path, setting.params, arm, pred_dir, wood_label,
                                  cached=cached)
                result.stage1_runtime_s = stage1_s
                result.runtime_s += stage1_s
        except Timeout:
            result = LoadResult(load_id=load_id, segmentation=arm, status=STATUS_TIMEOUT,
                                notes='exceeded {} s'.format(timeout_s))
            _add_strike(strikes, setting.key, arm)

        setting_rows = _tag(result.as_rows(), setting, reference)
        rows.extend(setting_rows)
        if setting.is_base and base_rows is not None:
            base_rows[setting.mesh_method] = setting_rows

        if verbose:
            print('  {:<14} {:<10} {:<14} {:<10} vg={:>8.3f}  {:>6.1f}s  {}'.format(
                load_id, arm, setting.mesh_method, setting.setting_id, result.vg_m3,
                result.runtime_s, result.status), flush=True)

    return rows


def _strikes_for(strikes, key, arm: str) -> int:
    return 0 if strikes is None else int(strikes.get((key, arm), 0))


def _add_strike(strikes, key, arm: str) -> None:
    if strikes is not None:
        strikes[(key, arm)] = _strikes_for(strikes, key, arm) + 1


def _timed_out(load_id: str, arm: str, note: str) -> list[dict]:
    return LoadResult(load_id=load_id, segmentation=arm,
                      status=STATUS_TIMEOUT, notes=note).as_rows()


def _reuses_base(setting: Setting, load_id: str) -> bool:
    """A DBSCAN setting on a single-bay load is BASE by construction -- see `DBSCAN_SETTINGS`."""
    return setting.setting_id.split('_')[0] in DBSCAN_SETTINGS and not is_multi_bay(load_id)


def _copy_base(base_rows: list[dict], setting: Setting, reference) -> list[dict]:
    copied = _tag([dict(row) for row in base_rows], setting, reference)
    for row in copied:
        row['notes'] = 'identical to BASE: this parameter is scoped to multi-bay loads'
    return copied


def _tag(rows: list[dict], setting: Setting, reference: dict | None) -> list[dict]:
    """Stamp the setting's identity on its rows, and the reference volume the row can carry.

    `vs_ref_m3` is the measured volume, a per-load constant, put on the row so that a checkpoint
    file can be read on its own while the campaign is still running -- the per-setting refit needs
    every load to have finished, and this does not.
    """
    reference = reference or {}
    for row in rows:
        row.update(setting_id=setting.setting_id, mesh_method=setting.mesh_method,
                   stage=setting.stage, method=setting.method,
                   parameter=setting.parameter, value=setting.value)
        plane = row.pop('plane', None)
        row['plane_a'], row['plane_b'], row['plane_c'] = (
            plane if plane is not None else (np.nan, np.nan, np.nan))
        row['vs_ref_m3'] = reference.get(row['load_id'], np.nan)

    return rows


def normalise_rows(rows) -> pd.DataFrame:
    """The raw per-bay frame, with the load-level and bay-level volume columns disentangled."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if 'vg_m3_bay' not in frame.columns:
        frame['vg_m3_bay'] = np.nan
    return frame


def noise_floor_settings(seeds=(1, 2, 3, 4),
                         base: Params = SENSITIVITY_BASE) -> list[Setting]:
    """BASE at other k-means seeds: the clustering jitter a parameter effect has to clear.

    A `d_vg` smaller than the spread these produce is not a parameter effect at all.
    """
    return [Setting('{}_seed{}'.format(BASE_ID, seed), 0, 'All', 'kmeans seed', str(seed),
                    replace(base, kmeans_seed=seed))
            for seed in seeds]

