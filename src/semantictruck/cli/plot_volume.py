"""Manuscript Figure 4: solid wood volume against gross volume, coloured by campaign.

    semantictruck-plot-volume --table results/pipeline/results_per_truck.csv
    semantictruck-plot-volume --kind residual --out results/figures/residuals.png
    semantictruck-plot-volume --segmentation both --fit-line
    semantictruck-plot-volume --figsize 6 5 --font-size 11 --no-title

Two markers per truckload at the same x: **filled** is the measured solid volume, **hollow** is
what the leave-one-out calibration predicted from that load's gross volume, so the vertical gap
between a pair is that load's LOO residual. Red is the 2016 campaign (semitrailers), blue the 2021
one (B-doubles).

`--kind residual` draws the companion view: prediction error against gross volume, same colours and
the same marker meaning, with the calibration subtracted. There the filled marker is the in-data
residual and the hollow one the leave-one-out residual of the same load, so the gap within a pair
is the optimism of an in-sample error.

`--segmentation both` draws the two arms side by side sharing a y axis. The calibration is fitted
on the reference arm whatever is chosen -- the tables carry no other -- so the predicted panel shows
that same model against automatically segmented clouds, not a fit to the points under it.

Input is `results_per_truck.csv`, which is where `vs_hat_loo_m3` lives. A sweep table holds many
settings at once and has no leave-one-out column, so it is rejected rather than overplotted.

Note the two similar names: `--kind residual` chooses *what is plotted*, while `--residuals` joins
each load's pair of markers with a hair line in either plot.
"""


from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from semantictruck.arms import ARMS, REFERENCE
from semantictruck.analysis.figures import (
    CAMPAIGN_YEARS, GROSS_SOLID_TITLE, GROSS_SOLID_XLABEL, GROSS_SOLID_YLABEL, RESIDUAL_TITLE,
    RESIDUAL_YLABEL, SEGMENTATIONS, campaign_labels, gross_vs_solid, residual_vs_gross)

DEFAULT_RUN = 'results/pipeline'
DEFAULT_TABLE = os.path.join(DEFAULT_RUN, 'results_per_truck.csv')
DEFAULT_IMAGE = os.path.join(DEFAULT_RUN, 'figures', 'gross_vs_solid.png')

REQUIRED = ('vg_m3', 'campaign')

# Distinguishes "--title not given" from "--no-title", which both have to survive the same
# mutually exclusive group; the default depends on --kind and is resolved after parsing
UNSET = object()

# What each plot needs: the y of its filled marker, of its hollow one, and its default labelling.
# `--kind residual` is the companion view -- same loads, same colours, same marker meaning, with
# the calibration subtracted out.
KINDS = {
    'volume': dict(draw=gross_vs_solid, filled='vs_ref_m3', hollow='vs_hat_loo_m3',
                   title=GROSS_SOLID_TITLE, ylabel=GROSS_SOLID_YLABEL, label='solid'),
    'residual': dict(draw=residual_vs_gross, filled='error_m3', hollow='error_loo_m3',
                     title=RESIDUAL_TITLE, ylabel=RESIDUAL_YLABEL, label='error'),
}

# For the console summary only -- the figure takes its hues from `SCATTER_CAMPAIGN_COLOR`, and
# naming them here saves the reader matching a hex code to what they will see
COLOUR_NAME = {'2016': 'red', '2021': 'blue'}

# As in `plot_box_whiskers`: matplotlib's own names, plus the two the figure adds
LEGEND_LOCS = ('best', 'upper left', 'upper right', 'lower left', 'lower right', 'upper center',
               'lower center', 'center left', 'center right', 'center', 'outside', 'none')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--table', default=DEFAULT_TABLE,
                        help='results_per_truck.csv from a finished run')
    parser.add_argument('--out', default=DEFAULT_IMAGE,
                        help='image to write; the extension picks the format (.png, .pdf, .svg)')
    parser.add_argument('--kind', default='volume', choices=list(KINDS),
                        help='"volume" plots solid against gross volume; "residual" plots the '
                             'prediction error against gross volume, same colours and markers '
                             'with the calibration subtracted out')

    parser.add_argument('--segmentation', '--arm', dest='segmentation', default=REFERENCE,
                        choices=list(SEGMENTATIONS),
                        help='which segmentation supplies the gross volumes: an arm, or '
                             '"both" for the two side by side, sharing the y axis. The '
                             'calibration is fitted on the reference arm whichever is chosen, '
                             'so the predicted panel shows that same model against automatically '
                             'segmented clouds. --arm is kept as an alias')
    parser.add_argument('--panel-titles', nargs='+', default=None, metavar='TEXT',
                        help='title of each panel under --segmentation both, in order (default: '
                             '"Manual segmentation", "Model segmentation"). Always drawn bold; '
                             'pass an empty string to drop one. With a single panel the title is '
                             '--title, as before')
    parser.add_argument('--campaigns', nargs='+', default=None, metavar='YEAR',
                        choices=list(CAMPAIGN_YEARS),
                        help='campaigns to draw, by year (default: both)')

    parser.add_argument('--figsize', nargs=2, type=float, metavar=('W', 'H'), default=None,
                        help='figure size in inches; default 7x6 per panel, so --segmentation '
                             'both is 14x6')
    parser.add_argument('--font-size', type=float, default=9,
                        help='tick label size in points; axis labels, title, legend and marker '
                             'size are scaled from it')
    parser.add_argument('--dpi', type=int, default=200, help='raster resolution')

    title = parser.add_mutually_exclusive_group()
    title.add_argument('--title', default=UNSET, help='figure title (default: depends on --kind)')
    title.add_argument('--no-title', dest='title', action='store_const', const=None,
                       help='drop the title, for a figure that will carry a caption instead')
    parser.add_argument('--xlabel', default=GROSS_SOLID_XLABEL, help='x-axis label')
    parser.add_argument('--ylabel', default=None,
                        help='y-axis label (default: depends on --kind)')
    parser.add_argument('--legend-loc', default='best', metavar='LOC', choices=LEGEND_LOCS,
                        help='where to put the key: any matplotlib location, plus "outside" to '
                             'move it clear of the axes and "none" to drop it. Choices: '
                             + ', '.join(LEGEND_LOCS))

    parser.add_argument('--no-loo', dest='show_loo', action='store_false',
                        help='draw only the measured volumes, without the hollow LOO markers')
    parser.add_argument('--residuals', dest='show_residuals', action='store_true',
                        help='join each measured/predicted pair with a hair line, so the pairing '
                             'is visible where the points crowd')
    parser.add_argument('--fit-line', dest='fit_line', action='store_true',
                        help='draw the in-data calibration, taken from the table\'s vs_hat_m3')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.table):
        print('no per-truck table at {} -- run semantictruck-pipeline first'.format(args.table))
        return 1

    kind = KINDS[args.kind]
    title = kind['title'] if args.title is UNSET else args.title
    ylabel = args.ylabel or kind['ylabel']

    table = pd.read_csv(args.table)
    if 'campaign' in table.columns:
        table['campaign'] = campaign_labels(table['campaign'])
    missing = [c for c in REQUIRED + (kind['filled'],) if c not in table.columns]
    if missing:
        print('{} has no column(s) {}; this tool reads results_per_truck.csv'.format(
            args.table, ', '.join(missing)))
        return 1

    # A sweep table holds every setting at once; scattering them together would draw one truckload
    # dozens of times over and read as spread that is not there
    for column in ('setting_id', 'mesh_method'):
        if column in table.columns and table[column].nunique() > 1:
            print('{} holds {} distinct {}s -- this tool plots one configuration. Filter it '
                  'first.'.format(args.table, table[column].nunique(), column))
            return 1

    if args.show_loo and kind['hollow'] not in table.columns:
        print('{} has no {} column; pass --no-loo to draw the filled markers alone'.format(
            args.table, kind['hollow']))
        return 1

    arms = ARMS if args.segmentation == 'both' else (args.segmentation,)
    present = set(table['segmentation']) if 'segmentation' in table.columns else set(arms)
    wanted = [a for a in arms if a not in present]
    if wanted:
        print('no {} row(s) in {}; present: {}'.format(
            ', '.join(wanted), args.table, ', '.join(sorted(present))))
        return 1

    shown = (table[table['segmentation'].isin(arms)] if 'segmentation' in table.columns else table)
    shown = shown.dropna(subset=['vg_m3', kind['filled']])
    if args.campaigns:
        shown = shown[shown['campaign'].isin(args.campaigns)]
    shown = shown[shown['campaign'].isin(CAMPAIGN_YEARS)]

    # Checked before drawing, because the failure this guards against is a *blank* figure with a
    # correct-looking legend -- which is written successfully and looks like a data problem
    if shown.empty:
        print('nothing to plot: no row of {} has a campaign in {} with both a gross and a '
              'reference volume.\ncampaign values present: {}'.format(
                  args.table, ', '.join(CAMPAIGN_YEARS),
                  ', '.join(sorted(set(table['campaign'].astype(str))))))
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    options = dict(
        figsize=tuple(args.figsize) if args.figsize else None, font_size=args.font_size,
        title=title, xlabel=args.xlabel, ylabel=ylabel,
        segmentation=args.segmentation, panel_titles=args.panel_titles,
        campaigns=args.campaigns,
        show_loo=args.show_loo, show_residuals=args.show_residuals,
        legend_loc=args.legend_loc, dpi=args.dpi)
    # The residual plot has no calibration to draw: subtracting it is what the plot *is*
    if args.kind == 'volume':
        options['fit_line'] = args.fit_line
    elif args.fit_line:
        print('note: --fit-line does not apply to --kind residual; the zero line replaces it')
    path = kind['draw'](table, args.out, **options)

    print('{}  ({} truckload(s), {} segmentation)'.format(
        path, len(shown), ' + '.join(arms)))
    for arm in arms:
        panel = shown[shown['segmentation'] == arm] if 'segmentation' in shown.columns else shown
        if len(arms) > 1:
            print('  {}:'.format(arm))
        for campaign, group in panel.groupby('campaign'):
            print('  {} campaign ({:>4}):  {:>2} loads   gross {:>5.1f}-{:>5.1f} m3   '
                  '{} {:>5.1f}-{:>5.1f} m3'.format(
                      campaign, COLOUR_NAME.get(campaign, '?'), len(group),
                      group['vg_m3'].min(), group['vg_m3'].max(), kind['label'],
                      group[kind['filled']].min(), group[kind['filled']].max()))
            if args.show_loo and 'error_loo_m3' in group.columns:
                residual = group['error_loo_m3'].abs()
                print('      LOO |residual| mean {:.3f} m3, worst {:.3f} m3'.format(
                    residual.mean(), residual.max()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
