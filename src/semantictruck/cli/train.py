"""Train the segmentation model: five-fold cross-validation over the truckloads.

    semantictruck-train --out results/segmentation
    semantictruck-train --out results/segmentation --only-fold 3
    semantictruck-train --out results/segmentation --metrics-only

**Pointcept is a dependency of this command and is not installed by `pip install -e .`** -- it is a
research framework distributed as source, with a CUDA extension to compile, and its torch and NumPy
requirements are the opposite of the ones the geometry stage pins. It therefore needs its own
environment, and this command needs to be told where it is: `--pointcept-root`, or
`$POINTCEPT_ROOT`. See `docs/reproducing.md`.

The two stages meet through files on disk, not through a shared process. This command writes one
out-of-fold prediction per truckload into `<out>/predictions/`; the geometry stage reads that
directory with `--arms reference predicted --pred-dir <out>/predictions`.

Outputs, all in `--out`:

    fold_<i>/               one Pointcept run per fold: checkpoints, logs, predictions
    predictions/            the pooled out-of-fold prediction for every truckload
    metrics_per_epoch.csv   one row per (fold, epoch) -- what `semantictruck-plot-curves` draws
    metrics_summary.csv     one row per fold, plus the mean over folds
    folds.json              the manifest this campaign used, copied for the record
"""

from __future__ import annotations

import argparse
import os
import sys

from semantictruck import config
from semantictruck.training import backend, crossval

DEFAULT_OUT = 'results/segmentation'
DEFAULT_CONFIG = config.repo_file('configs', 'ptv3_trucksegm.py')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-train', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--data-root', default=None,
                        help='dataset root; overrides ${}'.format(config.ENV_VAR))
    parser.add_argument('--dataset', default=None,
                        help='directory of .npy clouds, holding train/ and test/ '
                             '(default: <data-root>/training)')
    parser.add_argument('--folds-file', default=None,
                        help='the cross-validation manifest (default: <data-root>/folds.json)')
    parser.add_argument('--pointcept-root', default=None,
                        help='the Pointcept checkout (default: ${})'.format(backend.ENV_VAR))
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='the Pointcept configuration to train')
    parser.add_argument('--out', default=DEFAULT_OUT, help='directory for the campaign')

    parser.add_argument('--n-folds', type=int, default=5, help='number of cross-validation folds')
    parser.add_argument('--only-fold', type=int, default=None, help='train just this fold')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=None,
                        help='override the configured epoch budget')
    parser.add_argument('--eval-epoch', type=int, default=None,
                        help='override the configured evaluation interval')

    parser.add_argument('--metrics-only', action='store_true',
                        help='rebuild the CSVs from existing fold folders, training nothing')
    parser.add_argument('--force', action='store_true',
                        help='retrain folds that already have a checkpoint and predictions')
    parser.add_argument('--skip-inference', action='store_true',
                        help='train without predicting the held-out fold')
    parser.add_argument('--options', nargs='*', default=[], metavar='KEY=VALUE',
                        help='extra Pointcept configuration overrides, passed straight through')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    args.dataset = args.dataset or config.training_dir(args.data_root)
    args.folds_file = args.folds_file or config.folds_file(args.data_root)
    args.pointcept_root = backend.pointcept_root(args.pointcept_root)

    if not args.metrics_only and not os.path.isdir(args.dataset):
        raise SystemExit('no training clouds at {} -- see docs/reproducing.md'.format(
            args.dataset))
    if not os.path.exists(args.config):
        raise SystemExit('no configuration at {}'.format(args.config))

    return crossval.run_campaign(args)


if __name__ == '__main__':
    sys.exit(main())
