"""Summarise the solid-volume prediction error of a finished pipeline run.

    semantictruck-errors --table results/pipeline/results_per_truck.csv
    semantictruck-errors --by-campaign --arms reference

Reduces one row per truckload to the aggregate figures a results table quotes -- the manuscript's
Table 3, and with `--by-campaign` its Table 4. Nothing is fitted or predicted here, so the summary
cannot disagree with the run it describes.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from semantictruck.analysis.errors import (
    COLUMNS, EVALUATIONS, KEYS, REQUIRED, build_summary, long_view)
from semantictruck.arms import ARMS

DEFAULT_TABLE = 'results/pipeline/results_per_truck.csv'
DEFAULT_OUT = 'results/pipeline/error_summary.csv'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-errors', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--table', default=DEFAULT_TABLE,
                        help='results_per_truck.csv from a finished run')
    parser.add_argument('--out', default=DEFAULT_OUT, help='CSV to write')
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS),
                        help='segmentation arms to summarise')
    parser.add_argument('--by-campaign', action='store_true',
                        help='add a row per acquisition campaign beside the pooled one, to show '
                             'whether one truck configuration carries the error')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.table):
        print('no per-truck table at {} -- run semantictruck-pipeline first'.format(args.table))
        return 1

    table = pd.read_csv(args.table)
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        print('{} has no column(s) {}; this reads results_per_truck.csv'.format(
            args.table, ', '.join(missing)))
        return 1
    if not any(error in table.columns for error, _ in EVALUATIONS.values()):
        print('{} carries no error columns ({}); it is not a scored run'.format(
            args.table, ', '.join(error for error, _ in EVALUATIONS.values())))
        return 1

    # A sweep table holds every setting at once, and pooling their errors would average over
    # parameter choices rather than over truckloads
    for column in ('setting_id', 'mesh_method'):
        if column in table.columns and table[column].nunique() > 1:
            print('{} holds {} distinct {}s -- this summarises one configuration. Filter it '
                  'first.'.format(args.table, table[column].nunique(), column))
            return 1

    summary = build_summary(table, tuple(args.arms), args.by_campaign)
    if summary.empty:
        print('nothing to summarise: no row of {} has an error for arm(s) {}'.format(
            args.table, ', '.join(args.arms)))
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    summary.to_csv(args.out, index=False)

    print(long_view(summary).to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))
    print()
    print(summary[KEYS + ['rmse_m3']].to_string(
        index=False, float_format=lambda v: '{:.4f}'.format(v)))
    print('\n  in_data is fitted and scored on the same truckloads; loo predicts each truckload '
          'from a model\n  refitted without it. rmse_m3 is taken about zero and carries the bias; '
          'the SD of the error is\n  taken about the mean and removes it.')

    unscored = table[table['vs_ref_m3'].isna()]['load_id'].nunique()
    if unscored:
        print('\n{} truckload(s) carry no reference solid volume and are not in these '
              'figures'.format(unscored))

    print('\nwritten to {}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
