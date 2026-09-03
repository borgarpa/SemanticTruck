"""Run the one-at-a-time sensitivity analysis and build its tables and figures.

    semantictruck-sensitivity
    semantictruck-sensitivity --resume             # continue an interrupted campaign
    semantictruck-sensitivity --noise-floor
    semantictruck-sensitivity --analyse-only       # rebuild tables from existing runs

One campaign per reconstruction method -- Poisson, alpha shape, ball pivoting, convex hull -- each
a complete one-at-a-time sweep of the shared Stage 1 parameters against that method's own BASE,
which sits at the middle of its own grid. A method's parameters are swept only in its own campaign,
so nothing is confounded by a reconstruction it does not use, and reading the same parameter across
campaigns shows whether its effect depends on the reconstruction at all.

The regression is refitted per (method, setting) on the reference arm and the same calibration
scores every other arm, so a second arm measures what segmentation costs rather than what a
recalibration could hide.

The campaign is ordered **by Stage 1 group, then arm, then load**: a parameter combination is
finished across every truckload before the next one starts, so its regression is always fitted on
the full sample. Settings sharing a Stage 1 run as a group, because Stage 1 is independent of the
reconstruction. It runs in a single process and each (group, arm, load) is checkpointed, so
`--resume` picks up where an interruption left off, and completed settings are fitted and printed
as the campaign goes.

`--noise-floor` re-runs BASE at other k-means seeds: a parameter effect smaller than the spread
that produces is clustering jitter, not an effect.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from glob import glob

import pandas as pd

from semantictruck.analysis import figures
from semantictruck.analysis.reference import load_reference
from semantictruck.analysis.stats import (
    build_per_load, build_summary, check_invariants, selection_loo)
from semantictruck.arms import ARMS, REFERENCE
from semantictruck.cli.common import add_data_arguments, resolve
from semantictruck.cloud.io import check_predictions
from semantictruck.geometry.mesh import METHODS
from semantictruck.pipeline.loads import discover_loads, is_multi_bay
from semantictruck.pipeline.timeout import DEFAULT_TIMEOUT_S
from semantictruck.sensitivity.grid import load_settings
from semantictruck.sensitivity.sweep import (
    DEFAULT_STRIKES, noise_floor_settings, normalise_rows, run_group, stage1_groups)
from semantictruck import config

DEFAULT_OUT = 'results/sensitivity'
DEFAULT_GRID = config.repo_file('configs', 'sensitivity_grid.csv')

PER_BAY_CSV = 'results_per_bay.csv'
PER_LOAD_CSV = 'results_per_load.csv'
SUMMARY_CSV = 'summary_per_setting.csv'
NOISE_CSV = 'kmeans_noise_floor.csv'
SELECTION_CSV = 'selection_loo.csv'



def _shard_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, 'partial')
    os.makedirs(path, exist_ok=True)
    return path


def _collect_shards(shard_dir: str) -> pd.DataFrame:
    shards = sorted(glob(os.path.join(shard_dir, '*.csv')))
    if not shards:
        raise SystemExit('no per-load results in {}'.format(shard_dir))
    return normalise_rows(pd.concat([pd.read_csv(p) for p in shards], ignore_index=True))


def _group_tag(key, group) -> str:
    """Short stable identifier for a Stage 1 group and the settings it carries.

    Named by what the group *is*, not by its position, so resuming after changing `--settings` or
    `--mesh-methods` reuses only the shards that really match.
    """
    signature = repr(key) + '|' + ','.join(
        sorted('{}/{}'.format(s.mesh_method, s.setting_id) for s in group))
    return 'g' + hashlib.sha1(signature.encode()).hexdigest()[:10]


def execute(load_ids, settings, arms, paths, args, reference: dict) -> pd.DataFrame:
    """Run the campaign group-first, so a parameter combination becomes fittable as it lands.

    **Deliberately single-process.** Open3D's outlier removal and normal orientation already thread
    across every core, and running loads side by side only makes them contend: the largest bay
    peaks near 10 GB on its own. One load at a time gets the whole machine and a predictable
    footprint.
    """
    shard_dir = _shard_dir(args.out)
    groups = stage1_groups(settings)
    strikes: dict = {}
    started = time.perf_counter()
    done_settings = 0

    print('{} setting(s) in {} Stage 1 group(s) x {} load(s) x {} arm(s), single process, '
          '{} s timeout, {} strikes\n'.format(
              len(settings), len(groups), len(load_ids), len(arms), args.timeout, args.strikes))

    for index, (key, group) in enumerate(groups):
        names = ', '.join(sorted({s.setting_id for s in group}))
        print('=== group {}/{}: {} setting(s) [{}]'.format(
            index + 1, len(groups), len(group), names), flush=True)

        tag = _group_tag(key, group)
        for arm in arms:
            base_rows: dict = {}
            for load_id in load_ids:
                shard = os.path.join(shard_dir, '{}_{}_{}.csv'.format(tag, arm, load_id))
                if args.resume and os.path.exists(shard):
                    continue
                rows = run_group(load_id, arm, group, paths['clouds'], paths['predictions'],
                                 args.timeout, args.pred_wood_label, strikes=strikes,
                                 strike_limit=args.strikes, base_rows=base_rows,
                                 reference=reference)
                pd.DataFrame(rows).to_csv(shard, index=False)

            print('--- group {} [{}] complete over {} load(s), {:.1f} min elapsed'.format(
                index + 1, arm, len(load_ids), (time.perf_counter() - started) / 60), flush=True)

        done_settings += len(group)
        _report_progress(shard_dir, group, done_settings, len(settings), paths['reference'],
                         args.reference_method)

    print('\ncampaign finished in {:.1f} min'.format((time.perf_counter() - started) / 60))
    return _collect_shards(shard_dir)


def _report_progress(shard_dir, group, done_settings, total_settings, reference_csv,
                     method) -> None:
    """Fit and print the settings that have just become complete across every truckload."""
    try:
        per_load = build_per_load(_collect_shards(shard_dir),
                                  load_reference(reference_csv, method))
        summary = build_summary(per_load)
    except Exception as exc:                                            # noqa: BLE001
        print('    (fit not available yet: {}: {})'.format(type(exc).__name__, exc), flush=True)
        return

    keys = {(s.mesh_method, s.setting_id) for s in group}
    just_done = summary[summary.apply(
        lambda r: (r['mesh_method'], r['setting_id']) in keys, axis=1)]
    columns = ['mesh_method', 'setting_id', 'segmentation', 'n_fit', 'a', 'b', 'adj_r2',
               'rmse_m3', 'rmse_loo_m3', 'mape_pct', 'mape_loo_pct']
    print(just_done[columns].to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))
    print('    {}/{} settings complete  (rmse_m3 = in-sample, rmse_loo_m3 = leave-one-out)\n'
          .format(done_settings, total_settings), flush=True)


def analyse(per_bay: pd.DataFrame, out_dir: str, reference: pd.DataFrame,
            expected_n: int) -> pd.DataFrame:
    """The per-load and per-setting tables, plus the invariant checks."""
    per_load = build_per_load(per_bay, reference)
    summary = build_summary(per_load)

    per_load.to_csv(os.path.join(out_dir, PER_LOAD_CSV), index=False)
    summary.to_csv(os.path.join(out_dir, SUMMARY_CSV), index=False)

    selection = selection_loo(per_load)
    selection.to_csv(os.path.join(out_dir, SELECTION_CSV), index=False)
    print('\nBest-setting selection, scored without the load it predicts:')
    print(selection.to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))

    fitted = summary[summary['segmentation'] == REFERENCE]
    if not fitted.empty:
        best = summary.loc[fitted['rmse_m3'].idxmin()]
        print('lowest in-sample RMSE: {}/{} at {:.4f} m3 (leave-one-out {:.4f} m3) -- the gap is '
              'the cost of choosing on the same loads'.format(
                  best['mesh_method'], best['setting_id'], best['rmse_m3'], best['rmse_loo_m3']))

    problems = check_invariants(summary, expected_n)
    if problems:
        print('\n{} invariant problem(s):'.format(len(problems)))
        for problem in problems:
            print('  ! {}'.format(problem))
    else:
        print('\ninvariants hold: mean_vs_ref_m3 constant, n_fit = {} everywhere, '
              'no failures'.format(expected_n))

    return per_load


def report(summary: pd.DataFrame) -> None:
    columns = ['mesh_method', 'setting_id', 'segmentation', 'n_fit', 'a', 'b', 'adj_r2',
               'rmse_m3', 'rmse_loo_m3', 'd_rmse_m3', 'd_rmse_loo_m3',
               'mape_pct', 'mape_loo_pct', 'mean_vg_m3', 'ratio_cv']
    print('\n' + summary[[c for c in columns if c in summary.columns]].to_string(
        index=False, float_format=lambda v: '{:.4f}'.format(v)))
    print('\nrmse_m3 / mape_pct are in-sample (fitted and scored on the same loads); '
          'rmse_loo_m3 / mape_loo_pct are leave-one-out.')


# ------------------------------------------------------------------------------------ the modes

def noise_floor(paths, args, reference: dict) -> int:
    """BASE at four other k-means seeds, on the multi-bay loads whose split it decides."""
    load_ids = args.loads or [n for n in discover_loads(paths['clouds'])
                              if is_multi_bay(n, args.multi_bay_tag)]
    per_bay = execute(load_ids, noise_floor_settings(), (REFERENCE,), paths, args, reference)
    per_load = per_bay.groupby(['setting_id', 'load_id'], as_index=False).agg(
        vg_m3=('vg_m3_bay', 'sum'))

    base_path = os.path.join(args.out, PER_LOAD_CSV)
    if not os.path.exists(base_path):
        raise SystemExit('no {} in {} -- run the campaign first, so there is a BASE to compare '
                         'against'.format(PER_LOAD_CSV, args.out))

    base = pd.read_csv(base_path)
    base = base[(base['setting_id'] == 'BASE') & (base['segmentation'] == REFERENCE)]
    base = base.set_index('load_id')['vg_m3']

    per_load['base_vg_m3'] = per_load['load_id'].map(base)
    per_load['d_vg_pct'] = (100 * (per_load['vg_m3'] - per_load['base_vg_m3'])
                            / per_load['base_vg_m3'])
    per_load.to_csv(os.path.join(args.out, NOISE_CSV), index=False)

    spread = per_load.groupby('load_id')['d_vg_pct'].agg(['std', lambda s: s.abs().max()])
    spread.columns = ['sd_pct', 'max_abs_pct']
    print('\nk-means jitter, other seeds against seed 0:\n')
    print(spread.to_string(float_format=lambda v: '{:.4f}'.format(v)))
    print('\nnoise floor: SD {:.4f} %, worst |d_vg| {:.4f} % -- a parameter effect below this is '
          'clustering noise'.format(spread['sd_pct'].mean(), spread['max_abs_pct'].max()))
    return 0


# ------------------------------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-sensitivity', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    add_data_arguments(parser)
    parser.add_argument('--grid', default=DEFAULT_GRID, help='the parameter grid to expand')
    parser.add_argument('--out', default=DEFAULT_OUT, help='directory for the tables and figures')
    parser.add_argument('--settings', nargs='+', default=None,
                        help='setting ids to run. Default: all of them')
    parser.add_argument('--mesh-methods', nargs='+', default=list(METHODS), choices=list(METHODS),
                        help='which reconstruction campaigns to run')
    parser.add_argument('--loads', nargs='+', default=None,
                        help='load names to run. Default: every cloud found')
    parser.add_argument('--arms', nargs='+', default=[REFERENCE], choices=list(ARMS),
                        help='segmentation arms to run')
    parser.add_argument('--multi-bay-tag', default='_2021',
                        help='substring marking a load that carries more than one bay')
    parser.add_argument('--reference-method', default='Manual',
                        help='value of the reference CSV\'s Method column to keep, if it has one')
    parser.add_argument('--resume', action='store_true',
                        help='skip loads already checkpointed under <out>/partial/')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT_S,
                        help='per (setting, load, arm) wall-clock limit in seconds; 0 disables')
    parser.add_argument('--strikes', type=int, default=DEFAULT_STRIKES,
                        help='loads a setting may time out on before the rest are recorded as '
                             'timeouts without being run')
    parser.add_argument('--check-predictions', action='store_true',
                        help='verify every prediction row-aligns with its cloud, and stop')
    parser.add_argument('--noise-floor', action='store_true',
                        help='run BASE at other k-means seeds instead of the campaign')
    parser.add_argument('--analyse-only', action='store_true',
                        help='rebuild the tables and figures from an existing per-bay table')
    parser.add_argument('--no-figures', action='store_true', help='skip the figures')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve(args)
    os.makedirs(args.out, exist_ok=True)

    if args.check_predictions:
        report_rows = pd.DataFrame(check_predictions(
            discover_loads(paths['clouds']), paths['clouds'], paths['predictions'],
            args.pred_wood_label))
        print(report_rows.to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))
        print('\n{} prediction(s) aligned; agreement with the reference {:.4f}-{:.4f} '
              '(mean {:.4f})'.format(len(report_rows), report_rows['cargo_agreement'].min(),
                                     report_rows['cargo_agreement'].max(),
                                     report_rows['cargo_agreement'].mean()))
        return 0

    reference_table = load_reference(paths['reference'], args.reference_method)
    reference = dict(zip(reference_table['load_id'], reference_table['vs_ref_m3']))

    if args.noise_floor:
        return noise_floor(paths, args, reference)

    per_bay_path = os.path.join(args.out, PER_BAY_CSV)
    if args.analyse_only:
        per_bay = (pd.read_csv(per_bay_path) if os.path.exists(per_bay_path)
                   else _collect_shards(_shard_dir(args.out)))
    else:
        settings = load_settings(args.grid, tuple(args.mesh_methods))
        if args.settings:
            wanted = set(args.settings)
            missing = wanted - {s.setting_id for s in settings}
            if missing:
                raise SystemExit('unknown setting(s): {}'.format(', '.join(sorted(missing))))
            settings = [s for s in settings if s.setting_id in wanted]

        load_ids = args.loads or discover_loads(paths['clouds'])
        per_bay = execute(load_ids, settings, tuple(args.arms), paths, args, reference)
        per_bay.to_csv(per_bay_path, index=False)

    # The count the invariants expect is how many of the loads actually run carry a reference,
    # not how many the reference table holds -- otherwise running a subset reports every setting
    # as short of its sample.
    expected_n = per_bay['load_id'].isin(reference_table['load_id']).groupby(
        per_bay['load_id']).any().sum()
    per_load = analyse(per_bay, args.out, reference_table, int(expected_n))
    summary = pd.read_csv(os.path.join(args.out, SUMMARY_CSV))
    report(summary)

    if not args.no_figures:
        figure_dir = os.path.join(args.out, 'figures')
        written, failed = figures.draw_all(per_load, summary, figure_dir)
        print('\n{} figure(s) written to {}'.format(len(written), figure_dir))
        if failed:
            # A figure that will not draw is a result about the data, not a cosmetic problem.
            # These panels once came out blank across a whole campaign because a column had been
            # read back as a float, and a printed warning scrolled past while the run succeeded.
            print('{} figure(s) FAILED:'.format(len(failed)))
            for name, reason in failed:
                print('  ! {}: {}'.format(name, reason))
            return 1

    print('\ntables in {}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
