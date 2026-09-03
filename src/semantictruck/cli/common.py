"""Argument groups every command shares, so they cannot drift apart.

Each command takes `--data-root` plus the specific overrides it needs. Defaults are resolved
through `semantictruck.config`, which is the only module that names a location -- so no command
carries an absolute path, and a dataset moves by setting one environment variable.
"""

from __future__ import annotations

import argparse

from semantictruck import config
from semantictruck.arms import ARMS


def add_data_arguments(parser: argparse.ArgumentParser, *, clouds: bool = True,
                       reference: bool = True, predictions: bool = True) -> None:
    """Where the data is. `--data-root` sets them all; the rest override individually."""
    parser.add_argument('--data-root', default=None,
                        help='dataset root; overrides ${}. See docs/input-data.md'.format(
                            config.ENV_VAR))
    if clouds:
        parser.add_argument('--clouds', default=None,
                            help='directory of merged LAZ truckloads '
                                 '(default: <data-root>/clouds)')
    if reference:
        parser.add_argument('--reference', default=None,
                            help='CSV of measured solid volume '
                                 '(default: <data-root>/reference.csv)')
    if predictions:
        parser.add_argument('--pred-dir', default=None,
                            help='directory of per-point segmentation labels, <load>_pred.npy '
                                 '(default: <data-root>/predictions)')
        parser.add_argument('--pred-wood-label', type=int, default=0,
                            help='the value the prediction files use for wood. Getting this '
                                 'wrong inverts the cargo mask silently, so it is checked')


def add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--arms', nargs='+', default=[ARMS[0]], choices=list(ARMS),
                        help='segmentation arms to run. The model is always fitted on the '
                             'reference arm and applied to both')


def resolve(args) -> dict:
    """The paths this invocation will use, each argument falling back to the data root."""
    root = getattr(args, 'data_root', None)
    return {
        'clouds': getattr(args, 'clouds', None) or config.clouds_dir(root),
        'reference': getattr(args, 'reference', None) or config.reference_csv(root),
        'predictions': getattr(args, 'pred_dir', None) or config.predictions_dir(root),
        'results': config.results_dir(root),
        'figures': config.figures_dir(root),
    }
