"""What the method costs per truckload, repeated so the number has a spread.

    semantictruck-timing --repeats 5
    semantictruck-timing --repeats 5 --external-timings inference_per_load.csv
    semantictruck-timing --merge-only

Times the geometric pipeline and the solid-volume stage on every truckload, N times over, and
writes one row per (pass, truckload). A dedicated command rather than a flag on
`semantictruck-pipeline`, because repeating would multiply that command's rows and break the
statistical contract of its tables -- one fit per arm, one bay-detection row per load.

**Every pass is a cold run.** `run_load` is called directly, so no Stage 1 cache is in play and a
drawn k-means seed is refused: it would make each pass do different work.

**Two standard deviations are reported, deliberately kept apart.** `sd_within_load_s` is a
truckload's own passes pooled -- machine jitter, which is what repeating measures.
`sd_between_loads_s` is the spread of the per-load means -- cargo size, which repeating does not
touch. Pooling them would give one number for two different questions.

**A total already contains its components.** `t_process_total_s` contains the four step columns,
and an external stage's total contains its own parts. Only the unindented totals sum to
`t_end_to_end_s`; adding a component to its own total double-counts.

Segmentation is not timed here -- the model is not part of this package. `--external-timings` joins
a CSV of per-load segmentation times measured wherever that model runs, keyed on `load_id`, so the
end-to-end figure can be composed. Without it the geometric half stands alone.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from semantictruck import __version__
from semantictruck.analysis.reference import (
    campaign_of, config_of, load_reference, load_reference_per_bay)
from semantictruck.analysis.stats import apply_errors, fit_setting, loo_fits
from semantictruck.arms import ARMS, REFERENCE
from semantictruck.cli.common import add_data_arguments, resolve
from semantictruck.cli.pipeline import analyse
from semantictruck.pipeline.loads import cloud_path, discover_loads
from semantictruck.pipeline.params import config_to_mapping, load_config
from semantictruck.pipeline.run import LoadResult, run_load
from semantictruck.pipeline.status import STATUS_TIMEOUT
from semantictruck.pipeline.timeout import DEFAULT_TIMEOUT_S, Timeout, time_limit

DEFAULT_OUT = 'results/timing'

PER_TRUCK_CSV = 'timing_per_truck.csv'
SUMMARY_CSV = 'timing_summary.csv'
STAGES_CSV = 'timing_stages.csv'
ENV_JSON = 'timing_env.json'

# The stages a per-truck row breaks into, in the order they run. Indentation marks a component of
# the total above it -- see the module docstring on why they must never be added to it.
STAGES = (
    ('t_external_s', 'segmentation (external)'),
    ('t_process_total_s', 'geometry (total)'),
    ('t_read_s', '  read + cargo mask'),
    ('t_preprocess_s', '  dedup + decimate + align'),
    ('t_bays_s', '  outlier removal + bay split + bay align'),
    ('t_mesh_s', '  voxelize + mesh + integrate'),
    ('t_solid_s', 'solid volume (per truck)'),
    ('t_end_to_end_s', 'end to end'),
)

TOTALS = ('t_external_s', 't_process_total_s', 't_solid_s')

# What a per-truck row carries from the geometric run
LOAD_TIMERS = ('t_read_s', 't_preprocess_s', 't_bays_s', 't_mesh_s')

ORDER = ['rep', 'load_id', 'campaign', 'config', 'arm', 'n_cargo_points', 'n_bays']

EXTERNAL_KEY = 'load_id'
EXTERNAL_TIME = 't_external_s'


# ------------------------------------------------------------------------------------ one pass

def run_pass(rep: int, load_ids, params, arm: str, paths, args, reference: pd.DataFrame,
             reference_per_bay: pd.DataFrame, quiet: bool = False):
    """One complete sweep of every truckload, then the solid-volume stage over the result."""
    solid = dict(zip(reference['load_id'], reference['vs_ref_m3']))
    rows: list[dict] = []
    bay_rows: list[dict] = []
    started = time.perf_counter()

    for load_id in load_ids:
        try:
            with time_limit(args.timeout):
                result = run_load(cloud_path(load_id, paths['clouds']), params, arm,
                                  paths['predictions'], args.pred_wood_label)
        except Timeout:
            result = LoadResult(load_id=load_id, segmentation=arm, status=STATUS_TIMEOUT,
                                notes='exceeded {} s'.format(args.timeout))

        bay_rows.extend(_tag(result.as_rows(), solid, params))
        rows.append(_load_row(rep, result))

        if not quiet:
            print('  {:<14} {:>6.1f} s  (read {:>4.1f} | prep {:>4.1f} | bays {:>5.1f} | '
                  'mesh {:>4.1f})  {}'.format(
                      load_id, result.runtime_s, result.t_read_s, result.t_preprocess_s,
                      result.t_bays_s, result.t_mesh_s, result.status), flush=True)

    t_loads = time.perf_counter() - started

    per_truck, timings = _solid_stage(pd.DataFrame(bay_rows), reference, reference_per_bay,
                                      arm, len(load_ids))

    # Amortised over the loads it was computed for. The calibration is a set-level operation, so
    # there is no honest per-truck cost for it; `t_apply_s` is the one part that scales with the
    # number of trucks, and it is what a per-truck row carries. The rest lands in the stages table.
    per_load_solid = timings['t_apply_s'] / max(len(load_ids), 1)
    predictions = dict(zip(per_truck.get('load_id', []), per_truck.get('vs_hat_m3', [])))
    for row in rows:
        row['t_solid_s'] = per_load_solid
        row['vs_hat_m3'] = predictions.get(row['load_id'], np.nan)

    timings.update(rep=rep, arm=arm, n_loads=len(load_ids), t_all_loads_s=t_loads,
                   t_pass_total_s=time.perf_counter() - started)
    return rows, timings


def _tag(rows, solid, params):
    for row in rows:
        row.update(setting_id='BASE', mesh_method=params.method, stage=0,
                   method='All', parameter='All at configured value', value='-')
        plane = row.pop('plane', None)
        row['plane_a'], row['plane_b'], row['plane_c'] = (
            plane if plane is not None else (np.nan, np.nan, np.nan))
        row['vs_ref_m3'] = solid.get(row['load_id'], np.nan)
        if 'vg_m3_bay' not in row:
            row['vg_m3_bay'] = np.nan
    return rows


def _load_row(rep: int, result: LoadResult) -> dict:
    row = {'rep': rep, 'load_id': result.load_id, 'arm': result.segmentation,
           'campaign': campaign_of(result.load_id), 'config': config_of(result.load_id),
           # What outlier removal saw: the cargo after the mask, the duplicate drop and the
           # decimation -- not the raw cloud
           'n_cargo_points': result.points_in, 'n_bays': result.n_bays, 'vg_m3': result.vg_m3,
           'status': result.status, 'notes': result.notes}
    row.update({name: getattr(result, name) for name in LOAD_TIMERS})
    row['t_stage1_s'] = result.stage1_runtime_s
    row['t_process_total_s'] = result.runtime_s
    return row


def _solid_stage(per_bay: pd.DataFrame, reference: pd.DataFrame,
                 reference_per_bay: pd.DataFrame, arm: str, n_loads: int):
    """Time the gross-to-solid stage, and break it into calibration and application.

    The headline total calls `cli.pipeline.analyse`, the production function, rather than an
    imitation of it. The three components beneath re-call the same statistics on the finished table
    to say where that time went: the OLS calibration, the leave-one-out refits, and the part that
    scales with the number of trucks.
    """
    if arm != REFERENCE:
        # `build_per_load` calibrates from the rows labelled `reference`, because the design is
        # that the reference arm's fit scores every arm. A run of another arm alone has none, so
        # the label is set here and the fit comes from that arm's own gross volumes. The work is
        # identical -- the same OLS and the same leave-one-out refits, which is all a timing study
        # measures -- but the coefficients are *not* the published calibration. `fit_arm` on the
        # stages row says so, and the true arm is untouched on every per-truck row.
        per_bay = per_bay.assign(segmentation=REFERENCE)

    started = time.perf_counter()
    _, per_truck, _ = analyse(per_bay, reference, reference_per_bay)
    t_total = time.perf_counter() - started

    fitted = per_truck[per_truck['segmentation'] == REFERENCE]

    started = time.perf_counter()
    fit = fit_setting(fitted)
    t_fit = time.perf_counter() - started

    started = time.perf_counter()
    loo_fits(fitted)
    t_loo = time.perf_counter() - started

    started = time.perf_counter()
    apply_errors(fitted, fit.get('a', np.nan), fit.get('b', np.nan))
    t_apply = time.perf_counter() - started

    return per_truck, dict(t_solid_total_s=t_total, t_fit_s=t_fit, t_loo_s=t_loo,
                           t_apply_s=t_apply, fit_arm=arm)


# ------------------------------------------------------------------------------------- merging

def merge_external(per_truck: pd.DataFrame, path: str) -> tuple[pd.DataFrame, str]:
    """Join per-load segmentation times on, and say how the passes were matched.

    Matched pass for pass when both halves ran the same number of them; otherwise each load's mean
    external time is joined to every pass, and the merge mode says so. That distinction matters
    when reading the within-load spread, which a mean-joined column cannot contribute to.
    """
    external = pd.read_csv(path)
    if EXTERNAL_KEY not in external.columns or EXTERNAL_TIME not in external.columns:
        raise SystemExit('{} needs at least the columns {} and {}'.format(
            path, EXTERNAL_KEY, EXTERNAL_TIME))

    per_truck = per_truck.drop(columns=[EXTERNAL_TIME], errors='ignore')
    paired = ('rep' in external.columns
              and set(external['rep'].unique()) == set(per_truck['rep'].unique()))

    if paired:
        merged = per_truck.merge(external[[EXTERNAL_KEY, 'rep', EXTERNAL_TIME]],
                                 on=[EXTERNAL_KEY, 'rep'], how='left')
        return merged, 'paired'

    means = external.groupby(EXTERNAL_KEY, as_index=False)[EXTERNAL_TIME].mean()
    return per_truck.merge(means, on=EXTERNAL_KEY, how='left'), 'mean'


def add_end_to_end(per_truck: pd.DataFrame) -> pd.DataFrame:
    """The composed per-truck cost: only the totals, never their components."""
    present = [c for c in TOTALS if c in per_truck.columns]
    per_truck['t_end_to_end_s'] = per_truck[present].sum(axis=1, min_count=1)
    return per_truck


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    head = [c for c in ORDER if c in frame.columns]
    return frame[head + [c for c in frame.columns if c not in head]].sort_values(
        ['rep', 'load_id'], ignore_index=True)


# ------------------------------------------------------------------------------------- summary

def summarise(per_truck: pd.DataFrame) -> pd.DataFrame:
    """Per stage: overall, per campaign, and per truck configuration."""
    rows = _summary_rows(per_truck, 'all', 'all')
    for campaign, frame in per_truck.groupby('campaign', dropna=False):
        rows += _summary_rows(frame, 'campaign', campaign)
    for configuration, frame in per_truck.groupby('config', dropna=False):
        rows += _summary_rows(frame, 'config', configuration)
    return pd.DataFrame(rows)


def _summary_rows(frame: pd.DataFrame, group: str, value) -> list[dict]:
    total = np.nan
    rows = []
    for stage, label in STAGES:
        if stage not in frame.columns:
            continue
        per_load = frame.groupby('load_id')[stage]
        means = per_load.mean()
        if means.dropna().empty:
            continue

        # Pooled within-load SD: the root of the mean of the per-load variances, which is how to
        # combine spreads that each estimate the same jitter. Averaging the per-load SDs instead
        # would understate it.
        variances = per_load.var(ddof=1).dropna()
        within = float(np.sqrt(variances.mean())) if not variances.empty else np.nan

        mean = float(means.mean())
        if stage == 't_end_to_end_s':
            total = mean
        rows.append({'group': group, 'value': value, 'stage': stage, 'label': label,
                     'n_loads': int(means.notna().sum()), 'n_reps': int(frame['rep'].nunique()),
                     'mean_s': mean, 'sd_within_load_s': within,
                     'sd_between_loads_s': float(means.std(ddof=1)),
                     'min_s': float(means.min()), 'max_s': float(means.max())})

    for row in rows:
        row['pct_of_total'] = 100.0 * row['mean_s'] / total if total == total else np.nan
    return rows


# --------------------------------------------------------------------------------- environment

def environment(params, args, arm: str, merge: str) -> dict:
    """The machine and the library versions, so a number can still be read years later."""
    import open3d
    import scipy
    import sklearn

    return {
        'semantictruck_version': __version__,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'cpu': _cpu_model(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
        'sklearn': sklearn.__version__, 'open3d': open3d.__version__,
        # Recorded, never set: changing them would make the numbers incomparable with earlier runs
        'threads': {name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS')},
        'arm': arm,
        'repeats': args.repeats,
        'warmup_pass': args.warmup_pass,
        'external_merge': merge,
        'params': config_to_mapping(params),
        'git_head': _git_head(os.path.dirname(os.path.abspath(__file__))),
    }


def _cpu_model() -> str:
    try:
        with open('/proc/cpuinfo') as handle:
            for line in handle:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or 'unknown'


def _git_head(path: str) -> str:
    try:
        return subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=5).stdout.strip() or 'n/a'
    except Exception:                                                   # noqa: BLE001
        return 'n/a'


# ------------------------------------------------------------------------------------ reporting

def report(summary: pd.DataFrame, per_truck: pd.DataFrame, merge: str) -> None:
    overall = summary[(summary['group'] == 'all')]
    print('\nper truckload, over {} pass(es) x {} load(s):\n'.format(
        per_truck['rep'].nunique(), per_truck['load_id'].nunique()))
    columns = ['label', 'mean_s', 'sd_within_load_s', 'sd_between_loads_s', 'min_s', 'max_s',
               'pct_of_total']
    print(overall[columns].to_string(index=False, float_format=lambda v: '{:.3f}'.format(v)))
    print('\n  An indented row is a component of the total above it and is already inside it; only'
          '\n  the unindented totals sum to the end-to-end figure.')
    print('  sd_within_load_s is a truckload\'s own passes pooled -- machine jitter, what '
          'repeating\n  measures. sd_between_loads_s is the spread of the per-load means -- cargo '
          'size, which\n  repeating does not touch.')
    if merge:
        print('  Segmentation time was joined from an external table, matched {}.'.format(merge))


# ------------------------------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-timing', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    add_data_arguments(parser)
    parser.add_argument('--repeats', type=int, default=5, help='measured passes over every load')
    parser.add_argument('--warmup-pass', action='store_true',
                        help='discard one pass first, so the file cache is warm for every '
                             'measured one')
    parser.add_argument('--arm', default=REFERENCE, choices=list(ARMS),
                        help='which segmentation the geometry is measured on')
    parser.add_argument('--config', default=None, help='JSON of pipeline parameters')
    parser.add_argument('--loads', nargs='+', default=None, help='load names to time')
    parser.add_argument('--out', default=DEFAULT_OUT, help='directory for the tables')
    parser.add_argument('--external-timings', default=None,
                        help='CSV of per-load segmentation times to join on; needs at least '
                             '{} and {} columns'.format(EXTERNAL_KEY, EXTERNAL_TIME))
    parser.add_argument('--merge-only', action='store_true',
                        help='redo just the join and the summary from an existing table')
    parser.add_argument('--reference-method', default='Manual',
                        help='value of the reference CSV\'s Method column to keep, if it has one')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT_S,
                        help='per-load wall-clock limit in seconds; 0 disables')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve(args)
    os.makedirs(args.out, exist_ok=True)
    per_truck_path = os.path.join(args.out, PER_TRUCK_CSV)

    if args.merge_only:
        if not os.path.exists(per_truck_path):
            raise SystemExit('no {} to merge into'.format(per_truck_path))
        per_truck = pd.read_csv(per_truck_path)
        merge = ''
        if args.external_timings:
            per_truck, merge = merge_external(per_truck, args.external_timings)
        per_truck = ordered(add_end_to_end(per_truck))
        per_truck.to_csv(per_truck_path, index=False)
        summary = summarise(per_truck)
        summary.to_csv(os.path.join(args.out, SUMMARY_CSV), index=False)
        report(summary, per_truck, merge)
        return 0

    params, drew_seed = load_config(args.config)
    if drew_seed:
        raise SystemExit('a drawn k-means seed would make each pass do different work; fix it '
                         'before timing anything')

    reference = load_reference(paths['reference'], args.reference_method)
    reference_per_bay = load_reference_per_bay(paths['reference'], args.reference_method)
    load_ids = args.loads or discover_loads(paths['clouds'])
    if not load_ids:
        raise SystemExit('no .laz clouds in {}'.format(paths['clouds']))

    print('{} load(s) x {} pass(es) on the {} arm, {} reconstruction\n'.format(
        len(load_ids), args.repeats, args.arm, params.method))

    if args.warmup_pass:
        print('warm-up pass (discarded)...', flush=True)
        run_pass(-1, load_ids, params, args.arm, paths, args, reference, reference_per_bay,
                 quiet=True)

    rows, stages = [], []
    for rep in range(args.repeats):
        print('pass {}/{}'.format(rep + 1, args.repeats), flush=True)
        pass_rows, pass_stages = run_pass(rep, load_ids, params, args.arm, paths, args,
                                          reference, reference_per_bay)
        rows.extend(pass_rows)
        stages.append(pass_stages)

    per_truck = pd.DataFrame(rows)
    merge = ''
    if args.external_timings:
        per_truck, merge = merge_external(per_truck, args.external_timings)

    per_truck = ordered(add_end_to_end(per_truck))
    per_truck.to_csv(per_truck_path, index=False)

    summary = summarise(per_truck)
    summary.to_csv(os.path.join(args.out, SUMMARY_CSV), index=False)
    pd.DataFrame(stages).to_csv(os.path.join(args.out, STAGES_CSV), index=False)

    with open(os.path.join(args.out, ENV_JSON), 'w') as handle:
        json.dump(environment(params, args, args.arm, merge), handle, indent=2)
        handle.write('\n')

    report(summary, per_truck, merge)
    print('\ntables in {}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
