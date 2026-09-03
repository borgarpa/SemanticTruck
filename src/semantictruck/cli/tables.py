"""Print the manuscript's result tables from a finished run.

    semantictruck-tables --run results/pipeline
    semantictruck-tables --run results/pipeline --out results/pipeline/tables

Table 1 comes from `results_per_truck.csv`; Tables 3 and 4 from the error summary, which is built
here if it is not already present. Nothing is recomputed from the clouds.

**Manuscript Table 2 is not produced.** It reports the segmentation model's per-fold accuracy and
IoU, which belongs to the training campaign rather than to this pipeline -- the model is out of
scope. `semantictruck-plot-curves` draws that campaign's per-epoch metrics if you have its CSV.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from semantictruck.analysis import tables
from semantictruck.analysis.errors import build_summary
from semantictruck.arms import ARMS, REFERENCE

DEFAULT_RUN = 'results/pipeline'
PER_TRUCK = 'results_per_truck.csv'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-tables', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', default=DEFAULT_RUN, help='a finished run directory')
    parser.add_argument('--out', default=None,
                        help='directory to write the tables to as CSV; printed either way')
    parser.add_argument('--arm', default=REFERENCE, choices=list(ARMS),
                        help='arm for the per-campaign table, which the manuscript reports for '
                             'the reference segmentation')
    parser.add_argument('--decimals', type=int, default=2, help='rounding for the printed tables')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    per_truck_path = os.path.join(args.run, PER_TRUCK)
    if not os.path.exists(per_truck_path):
        print('no {} in {} -- run semantictruck-pipeline first'.format(PER_TRUCK, args.run))
        return 1

    per_truck = pd.read_csv(per_truck_path)
    errors = build_summary(per_truck, ARMS, by_campaign=True)

    built = {
        'table1_descriptive.csv': ('TABLE 1  Measured and reconstructed volume, by campaign',
                                   tables.descriptive_table(per_truck)),
        'table3_error.csv': ('TABLE 3  Leave-one-out solid-volume error, by segmentation',
                             tables.error_table(errors)),
        'table4_error_by_campaign.csv': (
            'TABLE 4  Leave-one-out solid-volume error, by campaign ({} arm)'.format(args.arm),
            tables.campaign_table(errors, args.arm)),
    }

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for name, (title, table) in built.items():
        print('\n======= {} =======\n'.format(title))
        if table.empty:
            print('  (not available from this run)')
            continue
        print(table.round(args.decimals).to_string())
        if args.out:
            table.round(args.decimals).to_csv(os.path.join(args.out, name))

    if args.out:
        print('\nwritten to {}'.format(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
