"""Supplementary figure: the spread of leave-one-out RMSE across a sweep, per reconstruction.

    semantictruck-plot-sensitivity --table results/sensitivity/summary_per_setting.csv
    semantictruck-plot-sensitivity --arms reference predicted --out figures/rmse.png

One box per reconstruction method, over every setting that method's campaign ran, with its BASE
marked as a diamond. What the box shows is how much the answer moves when a parameter does: a wide
box is a method whose result depends on choices the manuscript had to make, a narrow one is a method
that would have given much the same answer under any of them.

Input is the sweep's `summary_per_setting.csv`. `semantictruck-sensitivity` draws this and the rest
of the supplementary figures itself; this command exists to redraw one of them at a different size
or with different arms without re-running anything.
"""


from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from semantictruck.arms import ARMS, REFERENCE
from semantictruck.analysis.figures import (
    box_whiskers, DEFAULT_TITLE, DEFAULT_YLABEL, MESH_LABEL)

OUT_DIR = 'results/sensitivity'
DEFAULT_SUMMARY = os.path.join(OUT_DIR, 'summary_per_setting.csv')
DEFAULT_OUT = os.path.join(OUT_DIR, 'figures', 'rmse_loo_by_method.png')

# Matplotlib's own location names, plus the two the figure adds
LEGEND_LOCS = ('best', 'upper left', 'upper right', 'lower left', 'lower right', 'upper center',
               'lower center', 'center left', 'center right', 'center', 'outside', 'none')


def limit(text: str) -> float | None:
    """One end of `--cut`: a number, or `auto` for "as far as the data goes".

    A cut that runs off the end of the axis is the common case -- drop everything above the boxes,
    or everything below them -- and `auto` spells it without making the caller find a number past
    the last value, which is a number that changes when the campaign is rerun.
    """
    if text.lower() in ('auto', 'none', '-'):
        return None
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            '{!r} is not a number or "auto"'.format(text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--summary', default=DEFAULT_SUMMARY,
                        help='summary_per_setting.csv from a finished campaign')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='image to write; the extension picks the format (.png, .pdf, .svg)')

    parser.add_argument('--figsize', nargs=2, type=float, metavar=('W', 'H'), default=None,
                        help='figure size in inches; default scales with the number of boxes')
    parser.add_argument('--font-size', type=float, default=9,
                        help='tick label size in points; axis labels, title and legend are '
                             'scaled from it')
    parser.add_argument('--dpi', type=int, default=200, help='raster resolution')

    title = parser.add_mutually_exclusive_group()
    title.add_argument('--title', default=DEFAULT_TITLE, help='figure title')
    title.add_argument('--no-title', dest='title', action='store_const', const=None,
                       help='drop the title, for a figure that will carry a caption instead')
    parser.add_argument('--ylabel', default=DEFAULT_YLABEL, help='y-axis label')

    parser.add_argument('--column', default='rmse_loo_m3',
                        help='numeric column of the summary table to plot')
    parser.add_argument('--methods', nargs='+', default=None, metavar='NAME',
                        help='mesh-method campaigns to show, in order (default: all present)')
    parser.add_argument('--arms', nargs='+', default=None, choices=list(ARMS), metavar='ARM',
                        help='segmentation arms to show (default: all present)')

    parser.add_argument('--cut', nargs=2, type=limit, metavar=('LOW', 'HIGH'), default=None,
                        help='the interval of the y-axis to remove; "auto" at either end runs the '
                             'cut off that end of the data. A band inside the data breaks the '
                             'axis in two, keeping what lies on both sides -- the boxes at a '
                             'legible scale below, the collapsed settings still on the page above '
                             '-- while a band running off one end simply drops what it covers. '
                             'Unlike --no-fliers this changes the view only: every value still '
                             'counts towards the median, the quartiles and the whiskers')
    parser.add_argument('--cut-ratio', type=float, default=0.25, metavar='F',
                        help='share of the plot height given to the panel above a cut band; '
                             'ignored when the cut runs off an end and there is only one panel')
    parser.add_argument('--no-break-mark', dest='break_mark', action='store_false',
                        help='hide the slashes drawn where --cut interrupts the y-axis, for a '
                             'figure whose caption states the cut instead')

    legend = parser.add_mutually_exclusive_group()
    legend.add_argument('--legend-loc', default='best', metavar='LOC', choices=LEGEND_LOCS,
                        help='where to put the key: any matplotlib location, plus "outside" to '
                             'move it clear of the axes and "none" to drop it. The default picks '
                             'the emptiest corner, which matters because an error column leaves '
                             'the top clear while adj_r2 fills it. Choices: ' + ', '.join(
                                 LEGEND_LOCS))
    legend.add_argument('--no-legend', dest='legend_loc', action='store_const', const='none',
                        help='drop the legend, for a figure whose caption names the series '
                             'instead. The same as --legend-loc none, spelled the way the other '
                             'suppressing flags here are')
    parser.add_argument('--no-base', dest='show_base', action='store_false',
                        help='hide the BASE marker on each box')
    parser.add_argument('--no-fliers', dest='show_fliers', action='store_false',
                        help='hide outliers; note they are the collapsed and degenerate settings, '
                             'so the plot then understates how badly a campaign can fail')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.summary):
        print('no summary table at {} -- run the campaign first'.format(args.summary))
        return 1

    summary = pd.read_csv(args.summary)
    for column in ('mesh_method', 'segmentation', 'setting_id', args.column):
        if column not in summary.columns:
            print('{} has no column {!r}'.format(args.summary, column))
            return 1

    if not 0 < args.cut_ratio < 1:
        print('--cut-ratio is a share of the height, so it lies between 0 and 1, not {}'.format(
            args.cut_ratio))
        return 1

    if args.methods:
        unknown = set(args.methods) - set(summary['mesh_method'])
        if unknown:
            print('no campaign(s) named {} in {}; present: {}'.format(
                ', '.join(sorted(unknown)), os.path.basename(args.summary),
                ', '.join(sorted(set(summary['mesh_method'])))))
            return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    try:
        path = box_whiskers(
            summary, args.out,
            figsize=tuple(args.figsize) if args.figsize else None,
            font_size=args.font_size, title=args.title, ylabel=args.ylabel,
            methods=args.methods, arms=args.arms, column=args.column,
            cut=tuple(args.cut) if args.cut is not None else None,
            cut_ratio=args.cut_ratio, break_mark=args.break_mark,
            legend_loc=args.legend_loc, show_base=args.show_base,
            show_fliers=args.show_fliers, dpi=args.dpi)
    except ValueError as bad_cut:
        print(bad_cut)
        return 1

    shown = summary.dropna(subset=[args.column])
    if args.methods:
        shown = shown[shown['mesh_method'].isin(args.methods)]
    if args.arms:
        shown = shown[shown['segmentation'].isin(args.arms)]
    dropped = len(summary) - len(summary.dropna(subset=[args.column]))

    print('{}  ({} setting-rows over {} campaign(s){})'.format(
        path, len(shown), shown['mesh_method'].nunique(),
        '; {} row(s) with no {} omitted'.format(dropped, args.column) if dropped else ''))

    # A cut hides values without dropping them, so the count belongs on stdout: it is what a
    # caption has to declare, and the figure itself cannot carry it.
    if args.cut is not None:
        low, high = args.cut
        values = shown[args.column].dropna()
        inside = ((values >= low) if low is not None else True) & \
                 ((values <= high) if high is not None else True)
        print('  y-axis cut over [{}, {}]; {} value(s) fall in the removed band, still counted '
              'in every box'.format('auto' if low is None else low,
                                    'auto' if high is None else high, int(inside.sum())))
    for mesh, group in shown.groupby('mesh_method'):
        fitted = group[group['segmentation'] == REFERENCE][args.column]
        if fitted.empty:
            continue
        print('  {:<14} median {:.3f}  IQR {:.3f}  range {:.3f}-{:.3f}'.format(
            MESH_LABEL.get(mesh, mesh), fitted.median(),
            fitted.quantile(.75) - fitted.quantile(.25), fitted.min(), fitted.max()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
