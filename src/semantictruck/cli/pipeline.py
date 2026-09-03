"""Run the pipeline once, at one configuration, and report what it delivers.

    semantictruck-pipeline                              # the published configuration
    semantictruck-pipeline --loads Truck_1_2021 --arms reference     # a quick check
    semantictruck-pipeline --write-config my.json       # start from the defaults

This is the production entry point, and the counterpart of `semantictruck-sensitivity`: the same
geometry, the same regression, one configuration instead of eighty-four. Everything downstream of
the cargo mask is `pipeline.run.run_load` and the fit is `analysis.stats`, so the two commands
cannot drift apart.

**The model is fitted on the reference arm** and the resulting `Vs = a * Vg + b` scores both arms.
That is what makes the automated arm's numbers mean something: its extra error is segmentation, not
a recalibration that absorbed it. Every error is reported twice --

    in-data   fitted and evaluated on the same truckloads, optimistic by construction
    LOO       each load predicted by a model refitted on the other 28, which is what the method
              would give on a truck it had never seen

-- because quoting the first alone overstates the method and the second alone hides how much of the
gap is the small sample.

Outputs, all in `--out`:

    results_per_bay.csv     one row per (arm, truckload, bay): gross volume, mesh diagnostics,
                            the bay's own reference solid volume, and its meshing time
    results_per_truck.csv   one row per (arm, truckload): gross volume, reference solid volume,
                            in-data and LOO predictions with their errors, the fold's own
                            calibration, per-step timings
    model_summary.csv       one row per arm: a, b, adjusted R², Breusch-Pagan p, and every error
                            metric in both its in-data and LOO form
    bay_detection.csv       one row per (arm, multi-bay truckload): the confusion matrix of the
                            bay split against the reference bays, and the metrics derived from it
    timings.csv             per-step wall clock, per (arm, truckload) and for the run
    config_used.json        the configuration as resolved, so the run can be repeated
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from semantictruck.analysis.reference import (
    campaign_of, config_of, load_reference, load_reference_per_bay)
from semantictruck.analysis.stats import build_per_load, build_summary, loo_fits
from semantictruck.arms import ARMS, REFERENCE
from semantictruck.cli.common import add_arm_arguments, add_data_arguments, resolve
from semantictruck.pipeline.bays import (
    ALL_BAY_TRUTH, NO_BAY_TRUTH, attach_bay_scores, bay_detection_table, resolve_bay_truth)
from semantictruck.pipeline.loads import cloud_path, discover_loads, is_multi_bay
from semantictruck.pipeline.params import (
    PUBLISHED, config_to_mapping, load_config, write_config)
from semantictruck.pipeline.run import LoadResult, run_load
from semantictruck.pipeline.status import STATUS_OK, STATUS_TIMEOUT
from semantictruck.pipeline.timeout import DEFAULT_TIMEOUT_S, Timeout, time_limit

PER_BAY_CSV = 'results_per_bay.csv'
PER_TRUCK_CSV = 'results_per_truck.csv'
SUMMARY_CSV = 'model_summary.csv'
BAY_CSV = 'bay_detection.csv'
TIMINGS_CSV = 'timings.csv'
CONFIG_JSON = 'config_used.json'

DEFAULT_OUT = 'results/pipeline'

# The single setting every row carries. `analysis.stats` computes within-setting deltas against a
# setting called BASE; with one configuration there is nothing else to compare against, so naming
# it BASE makes those deltas identically zero rather than missing.
SETTING_ID = 'BASE'

# Per (arm, load) steps, in the order they happen. The first three are Stage 1, the fourth Stage 2.
STEPS = (('t_read_s', 'read + cargo mask'),
         ('t_preprocess_s', 'dedup + decimate + align'),
         ('t_bays_s', 'outlier removal + bay split + bay align'),
         ('t_mesh_s', 'voxelize + mesh + integrate'),
         ('runtime_s', 'total'))


# ------------------------------------------------------------------------------------- the run

def run_campaign(load_ids, params, arms, paths, args, reference: pd.DataFrame):
    """Every (arm, truckload) at the one configuration. Returns the per-bay rows and the timings."""
    solid = dict(zip(reference['load_id'], reference['vs_ref_m3']))
    rows: list[dict] = []
    timings: list[dict] = []
    started = time.perf_counter()

    print('{} truckload(s) x {} arm(s) at one configuration, {} reconstruction, k-means seed {}, '
          '{} s timeout\n'.format(len(load_ids), len(arms), params.method, params.kmeans_seed,
                                  args.timeout))

    for arm in arms:
        for load_id in load_ids:
            try:
                with time_limit(args.timeout):
                    result = run_load(cloud_path(load_id, paths['clouds']), params, arm,
                                      paths['predictions'], args.pred_wood_label,
                                      args.multi_bay_tag)
            except Timeout:
                result = LoadResult(load_id=load_id, segmentation=arm, status=STATUS_TIMEOUT,
                                    notes='exceeded {} s'.format(args.timeout))

            rows.extend(_tag(result.as_rows(), solid, params))
            timings.extend(_timing_rows(result))

            print('  {:<14} {:<10} vg={:>8.3f} m3   {:>6.1f} s   {}{}'.format(
                load_id, arm, result.vg_m3, result.runtime_s, result.status,
                '  [{}]'.format(result.notes) if result.notes else ''), flush=True)

    elapsed = time.perf_counter() - started
    timings.append({'scope': 'run', 'segmentation': '', 'load_id': '',
                    'step': 'all runs', 'seconds': elapsed})
    print('\n{} run(s) in {:.1f} min'.format(len(load_ids) * len(arms), elapsed / 60))

    return pd.DataFrame(rows), timings


def _tag(rows: list[dict], solid: dict, params) -> list[dict]:
    """Stamp the identity the statistics group by, and the reference volume, onto each bay row.

    The setting columns are constant here, unlike in the sweep, but they are still written so that
    the tables have the same schema as a sweep's and can be concatenated with them.
    """
    for row in rows:
        row.update(setting_id=SETTING_ID, mesh_method=params.method, stage=0,
                   method='All', parameter='All at configured value', value='-')
        plane = row.pop('plane', None)
        row['plane_a'], row['plane_b'], row['plane_c'] = (
            plane if plane is not None else (np.nan, np.nan, np.nan))
        row['vs_ref_m3'] = solid.get(row['load_id'], np.nan)
        if 'vg_m3_bay' not in row:
            row['vg_m3_bay'] = np.nan
    return rows


def _timing_rows(result: LoadResult) -> list[dict]:
    return [{'scope': 'load', 'segmentation': result.segmentation, 'load_id': result.load_id,
             'step': label, 'seconds': getattr(result, attribute, np.nan)}
            for attribute, label in STEPS]


# ------------------------------------------------------------------------------------ analysis

def analyse(per_bay: pd.DataFrame, reference: pd.DataFrame, reference_per_bay: pd.DataFrame):
    """Fit on the reference arm, score every arm in-data and leave-one-out."""
    per_truck = build_per_load(per_bay, reference)
    summary = build_summary(per_truck)

    # The LOO prediction of each load, from a model refitted on the other 28 of the reference arm.
    # The fold's own coefficients ride along on the row -- `a_loo` and `b_loo` are the calibration
    # that produced `vs_hat_loo_m3`, and `adj_r2_loo` its fit to the loads it was built from -- so
    # the prediction can be recomputed and audited without refitting anything.
    fits = loo_fits(per_truck[per_truck['segmentation'] == REFERENCE])
    predicted = []
    for _, frame in per_truck.groupby('segmentation'):
        frame = frame.copy().join(fits, on='load_id')
        frame['vs_hat_loo_m3'] = frame['a_loo'] * frame['vg_m3'] + frame['b_loo']
        frame['error_loo_m3'] = frame['vs_hat_loo_m3'] - frame['vs_ref_m3']
        frame['error_loo_pct'] = 100 * frame['error_loo_m3'] / frame['vs_ref_m3']
        # The same derived errors `stats.apply_errors` writes for the in-data calibration, so the
        # two evaluations carry the same columns and a per-truck table aggregates either way
        # without redoing the arithmetic
        frame['se_loo_m3'] = frame['error_loo_m3'] ** 2
        frame['ae_loo_m3'] = frame['error_loo_m3'].abs()
        frame['ape_loo_pct'] = frame['error_loo_pct'].abs()
        predicted.append(frame)

    per_truck = pd.concat(predicted, ignore_index=True).sort_values(
        ['segmentation', 'load_id'], ignore_index=True)

    # Every setting is BASE here, so the sweep's within-setting deltas are identically zero
    summary = summary.drop(columns=[c for c in summary.columns if c.startswith('d_')],
                           errors='ignore')
    summary = summary.drop(columns=['setting_id', 'stage', 'method', 'parameter', 'value'],
                           errors='ignore')

    if not reference_per_bay.empty:
        per_bay = per_bay.merge(reference_per_bay, on=['load_id', 'bay_id'], how='left')

    # Derived from the load name rather than merged from the reference table, so a load with no
    # measurement still carries its campaign instead of a blank
    per_bay['campaign'] = per_bay['load_id'].map(campaign_of)
    per_bay['config'] = per_bay['load_id'].map(config_of)

    return per_bay, per_truck, summary


# ------------------------------------------------------------------------------------ reporting

def _report_campaigns(per_truck: pd.DataFrame) -> None:
    """The result split by acquisition campaign.

    Printed because the two campaigns are two different truck configurations, so a model that works
    on one and not the other is worth seeing before the pooled numbers are believed.
    """
    scored = per_truck[per_truck['segmentation'] == REFERENCE].dropna(subset=['vs_ref_m3'])
    if scored.empty or 'campaign' not in scored.columns:
        return

    rows = []
    for (campaign, configuration), frame in scored.groupby(['campaign', 'config']):
        rows.append({
            'campaign': campaign,
            'configuration': configuration,
            'n_loads': len(frame),
            'mean_vg_m3': frame['vg_m3'].mean(),
            'mean_vs_ref_m3': frame['vs_ref_m3'].mean(),
            'rmse_m3': float(np.sqrt(frame['se_m3'].mean())),
            'rmse_loo_m3': float(np.sqrt((frame['error_loo_m3'] ** 2).mean())),
            'bias_m3': frame['error_m3'].mean(),
        })

    print('\nby acquisition campaign, {} arm:\n'.format(REFERENCE))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))


def _report_bay_detection(bays: pd.DataFrame, load_ids, tag: str) -> None:
    """What the bay split scored, and on how few loads -- both belong in the same breath."""
    single = sorted(name for name in load_ids if not is_multi_bay(name, tag))
    if bays.empty:
        print('\nbay detection: no multi-bay load was run, so the split was never exercised.')
        return

    scored = bays[bays['reason'] == ''] if 'reason' in bays.columns else bays
    print('\nbay detection (k-means + DBSCAN), against the reference bay segmentation:\n')
    if scored.empty:
        print('  {} multi-bay load(s) ran, none scored -- see the reason(s) below.'.format(
            len(bays)))
    else:
        columns = ['segmentation', 'load_id', 'campaign', 'n_ref_points', 'n_unassigned',
                   'n_unreferenced', 'bay_acc', 'bay_miou', 'iou_bay_1', 'iou_bay_2',
                   'ids_swapped']
        print(scored[[c for c in columns if c in scored.columns]].to_string(
            index=False, float_format=lambda v: '{:.4f}'.format(v)))

        print('\n  mean over the scored loads:')
        for arm, frame in scored.groupby('segmentation'):
            print('    {:<10} n={}   accuracy {:.4f} +/- {:.4f}   mIoU {:.4f} +/- {:.4f}'.format(
                arm, len(frame), frame['bay_acc'].mean(), frame['bay_acc'].std(),
                frame['bay_miou'].mean(), frame['bay_miou'].std()))
        print('  SD is the sample SD (ddof=1) over truckloads; ids are matched per load, so a '
              'swapped\n  numbering is not counted as error.')

    unscored = bays[bays['reason'] != ''] if 'reason' in bays.columns else bays.iloc[:0]
    for reason, frame in unscored.groupby('reason'):
        print('\n  {} (arm, load) pair(s) not scored -- {}:\n    {}'.format(
            len(frame), reason, ', '.join(sorted(frame['load_id'].unique()))))
    if single:
        print('\n  {} load(s) carry a single bay: one stack, no k-means, nothing to score:'
              '\n    {}'.format(len(single), ', '.join(single)))


def report(summary: pd.DataFrame, per_truck: pd.DataFrame, bays: pd.DataFrame,
           timings: pd.DataFrame, tag: str) -> None:
    fit = summary[summary['segmentation'] == REFERENCE]
    if not fit.empty and pd.notna(fit.iloc[0]['a']):
        row = fit.iloc[0]
        print('\nmodel, fitted on the {} arm ({} loads):  Vs = {:.6f} * Vg + {:.6f}'.format(
            REFERENCE, int(row['n_fit']), row['a'], row['b']))
        print('  adjusted R2 {:.4f}   Breusch-Pagan p {:.4f}'.format(
            row['adj_r2'], row['breusch_pagan_p']))

    columns = ['segmentation', 'n_scored', 'mean_vg_m3', 'mean_vs_ref_m3',
               'rmse_m3', 'mae_m3', 'mape_pct', 'bias_m3',
               'rmse_loo_m3', 'mae_loo_m3', 'mape_loo_pct', 'bias_loo_m3']
    print('\n' + summary[[c for c in columns if c in summary.columns]].to_string(
        index=False, float_format=lambda v: '{:.4f}'.format(v)))
    print('\n  *_loo columns are leave-one-out; the others are in-data, fitted and scored on the '
          'same loads.\n  Every arm is scored through the {} arm\'s calibration.'.format(REFERENCE))

    _report_campaigns(per_truck)
    _report_bay_detection(bays, per_truck['load_id'].unique(), tag)

    unscored = per_truck[per_truck['vs_ref_m3'].isna()]['load_id'].unique()
    if len(unscored):
        print('\n{} load(s) carry no reference solid volume and are in the tables but not the '
              'fit: {}'.format(len(unscored), ', '.join(sorted(unscored))))

    for row in per_truck[per_truck['status'] != STATUS_OK].itertuples():
        print('  ! {} [{}]: {} -- {}'.format(row.load_id, row.segmentation, row.status, row.notes))

    steps = timings[timings['scope'] == 'load']
    if steps.empty:
        return
    total = steps[steps['step'] == 'total']['seconds'].sum()
    print('\ntiming, summed over every run ({:.1f} min total):\n'.format(total / 60))
    for _, label in STEPS:
        seconds = steps[steps['step'] == label]['seconds']
        print('  {:<40} {:>8.1f} s   mean {:>6.1f} s   {:>5.1f} %'.format(
            label, seconds.sum(), seconds.mean(),
            100 * seconds.sum() / total if total else np.nan))


# ------------------------------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-pipeline', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    add_data_arguments(parser)
    add_arm_arguments(parser)
    parser.add_argument('--config', default=None,
                        help='JSON of pipeline parameters; omitted fields keep their published '
                             'value. Default: the published configuration throughout')
    parser.add_argument('--out', default=DEFAULT_OUT, help='directory for the tables')
    parser.add_argument('--loads', nargs='+', default=None,
                        help='load names to run. Default: every cloud found')
    parser.add_argument('--multi-bay-tag', default='_2021',
                        help='substring marking a load that carries more than one bay')
    parser.add_argument('--bay-truth', nargs='+', default=[ALL_BAY_TRUTH], metavar='LOAD',
                        help='loads whose `bay` field is read as a reference for the split. '
                             '{!r} means every multi-bay load, {!r} turns the scoring off, and an '
                             'explicit list restricts it'.format(ALL_BAY_TRUTH, NO_BAY_TRUTH))
    parser.add_argument('--reference-method', default='Manual',
                        help='value of the reference CSV\'s Method column to keep, if it has one')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT_S,
                        help='per (arm, load) wall-clock limit in seconds; 0 disables')
    parser.add_argument('--write-config', metavar='PATH', default=None,
                        help='write the published configuration to PATH and stop')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.write_config:
        write_config(PUBLISHED, args.write_config)
        print('published configuration written to {}'.format(args.write_config))
        return 0

    started = time.perf_counter()
    params, drew_seed = load_config(args.config)
    if drew_seed:
        print('k-means seed drawn at random: {} -- recorded in {}, which reruns this exact bay '
              'split'.format(params.kmeans_seed, CONFIG_JSON))

    paths = resolve(args)
    if not os.path.isdir(paths['clouds']):
        raise SystemExit('no cloud directory at {} -- see docs/input-data.md'.format(
            paths['clouds']))
    if not os.path.exists(paths['reference']):
        raise SystemExit('no reference table at {} -- see docs/input-data.md'.format(
            paths['reference']))

    reference = load_reference(paths['reference'], args.reference_method)
    reference_per_bay = load_reference_per_bay(paths['reference'], args.reference_method)

    load_ids = args.loads or discover_loads(paths['clouds'])
    if not load_ids:
        raise SystemExit('no .laz clouds in {}'.format(paths['clouds']))
    missing = [n for n in load_ids if not os.path.exists(cloud_path(n, paths['clouds']))]
    if missing:
        raise SystemExit('no cloud for {}'.format(', '.join(missing)))
    truth_loads = resolve_bay_truth(args.bay_truth, load_ids, args.multi_bay_tag)

    os.makedirs(args.out, exist_ok=True)
    write_config(params, os.path.join(args.out, CONFIG_JSON))
    setup_s = time.perf_counter() - started

    per_bay, timing_rows = run_campaign(load_ids, params, tuple(args.arms), paths, args, reference)

    started = time.perf_counter()
    per_bay, per_truck, summary = analyse(per_bay, reference, reference_per_bay)
    bays = bay_detection_table(per_bay, truth_loads, args.multi_bay_tag)
    per_truck = attach_bay_scores(per_truck, bays)
    fit_s = time.perf_counter() - started

    started = time.perf_counter()
    per_bay.to_csv(os.path.join(args.out, PER_BAY_CSV), index=False)
    per_truck.to_csv(os.path.join(args.out, PER_TRUCK_CSV), index=False)
    summary.to_csv(os.path.join(args.out, SUMMARY_CSV), index=False)
    bays.to_csv(os.path.join(args.out, BAY_CSV), index=False)

    timing_rows.extend([
        {'scope': 'run', 'segmentation': '', 'load_id': '',
         'step': 'setup (config + reference)', 'seconds': setup_s},
        {'scope': 'run', 'segmentation': '', 'load_id': '',
         'step': 'fit + leave-one-out', 'seconds': fit_s},
        {'scope': 'run', 'segmentation': '', 'load_id': '',
         'step': 'write tables', 'seconds': time.perf_counter() - started},
    ])
    timings = pd.DataFrame(timing_rows)
    timings.to_csv(os.path.join(args.out, TIMINGS_CSV), index=False)

    report(summary, per_truck, bays, timings, args.multi_bay_tag)
    print('\ntables in {}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
