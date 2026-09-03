"""Manuscript Figure 5: computation time against the size of what it processes.

    semantictruck-plot-time --table results/time_results.xlsx
    semantictruck-plot-time --y PhotoTotal_sec ProcessTotal_sec      # the unsplit left panel
    semantictruck-plot-time --fit pooled pooled --no-fit-line

Two panels, red 2016 and blue 2021: photogrammetry time against image count on the left, geometric
processing time against point count on the right.

**The left panel plots the two photogrammetry stages as separate series** -- camera alignment as
dots, cloud generation as diamonds, which sum exactly to the total -- because that is what locates
the campaign difference instead of merely showing it. Camera alignment costs almost the same per
image in both campaigns; cloud generation differs by nearly a factor of four, and the pooled total
shows one intermediate slope in place of two very different ones.

**The two panels fit differently, because the data differs.** The left fits one line per campaign:
the slopes differ significantly and the campaigns barely overlap in image count, so a pooled line
would be fitted largely to the empty gap between the clusters and land below both of the slopes it
claims to summarise. The right pools, because there the same test does not reject over a shared x
range, and splitting would draw two lines on top of each other and imply a difference that was
looked for and not found. Colour is campaign, so **shape carries the series**, which survives
greyscale printing; fit lines are all neutral grey and all solid, and a per-campaign line is
identified by where it stops.

`--x` takes one column per panel and `--y` a comma-separated list per panel, so any timing column
in the workbook can be plotted without touching the code. Descriptive axis labels and the x scale
are inherited only while a panel's columns are still its defaults, since a label naming a stage --
or a scale chosen for point counts -- is wrong the moment the panel points elsewhere.

Reading `.xlsx` needs `openpyxl`, which is a dependency of this package; a `.csv` is read too, by
extension.
"""


from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from semantictruck.analysis.figures import (
    CAMPAIGN_YEARS, COST_MARKERS, COST_PANELS, COST_TITLE, FIT_MODES, campaign_labels,
    cost_vs_size, linear_fit, panel_series)

DEFAULT_TABLE = 'results/time_results.xlsx'
DEFAULT_IMAGE = 'results/figures/time_vs_size.png'

# The workbook capitalises its campaign column; every colour map in `figures` is keyed on the
# lower-case name, so it is renamed on read rather than special-cased in the drawing
CAMPAIGN_COLUMN = 'Campaign'

# As in `plot_gross_vs_solid`: matplotlib's own names, plus the two the figure adds
LEGEND_LOCS = ('best', 'upper left', 'upper right', 'lower left', 'lower right', 'upper center',
               'lower center', 'center left', 'center right', 'center', 'outside', 'none')

# Distinguishes "--title not given" from "--no-title", which both have to survive the same
# mutually exclusive group
UNSET = object()

# For the console summary only -- the figure takes its hues from `SCATTER_CAMPAIGN_COLOR`
COLOUR_NAME = {'2016': 'red', '2021': 'blue'}


def columns(text: str) -> list[str]:
    """One panel's y columns: a name, or several comma-separated."""
    names = [name.strip() for name in text.split(',') if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError('{!r} names no column'.format(text))
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--table', default=DEFAULT_TABLE,
                        help='workbook or CSV with one row per truckload')
    parser.add_argument('--sheet', default=0,
                        help='worksheet to read, by name or index; ignored for a .csv')
    parser.add_argument('--out', default=DEFAULT_IMAGE,
                        help='image to write; the extension picks the format (.png, .pdf, .svg)')

    parser.add_argument('--x', nargs=2, metavar=('LEFT', 'RIGHT'),
                        default=[p['x'] for p in COST_PANELS],
                        help='column on each panel\'s x axis, left then right')
    parser.add_argument('--y', nargs=2, type=columns, metavar=('LEFT', 'RIGHT'),
                        default=[[s['y'] for s in panel_series(p)] for p in COST_PANELS],
                        help='column(s) on each panel\'s y axis, left then right. Several on one '
                             'panel are comma-separated and are drawn as separate series in the '
                             'marker order {}, each fitted and reported on its own -- campaign '
                             'stays the colour, so shape is what tells two series apart'.format(
                                 ', '.join(COST_MARKERS)))
    parser.add_argument('--campaigns', nargs='+', default=None, metavar='YEAR',
                        choices=list(CAMPAIGN_YEARS),
                        help='campaigns to draw, by year (default: both)')

    parser.add_argument('--figsize', nargs=2, type=float, metavar=('W', 'H'), default=None,
                        help='figure size in inches; default 14x6, i.e. 7x6 per panel')
    parser.add_argument('--font-size', type=float, default=9,
                        help='tick label size in points; axis labels, titles, legend and marker '
                             'size are scaled from it')
    parser.add_argument('--dpi', type=int, default=200, help='raster resolution')

    title = parser.add_mutually_exclusive_group()
    title.add_argument('--title', default=UNSET, help='suptitle over both panels')
    title.add_argument('--no-title', dest='title', action='store_const', const=None,
                       help='drop the title, for a figure that will carry a caption instead')
    parser.add_argument('--panel-titles', nargs='+', default=None, metavar='TEXT',
                        help='title of each panel, in order (default: "Photogrammetric '
                             'reconstruction", "Geometric processing"). Always drawn bold; pass '
                             'an empty string to drop one')
    parser.add_argument('--xlabels', nargs=2, metavar=('LEFT', 'RIGHT'), default=None,
                        help='x-axis label of each panel (default: describes the default columns; '
                             'with --x given, the column name)')
    parser.add_argument('--x-scale', nargs=2, type=float, metavar=('LEFT', 'RIGHT'), default=None,
                        help='divide each panel\'s x tick labels by this, for an axis whose unit '
                             'belongs in its label rather than on every tick: the points axis is '
                             '1e6 by default and reads 2, 3, 4 under a label saying millions. The '
                             'default applies only while a panel still plots its default column, '
                             'since a scale is only right for the quantity it was chosen for; '
                             'pass it explicitly (with --xlabels) when repointing --x')
    parser.add_argument('--ylabels', nargs=2, metavar=('LEFT', 'RIGHT'), default=None,
                        help='y-axis label of each panel (default: as --xlabels)')
    parser.add_argument('--legend-loc', default='best', metavar='LOC', choices=LEGEND_LOCS,
                        help='where to put the key: any matplotlib location, plus "outside" to '
                             'move it clear of the axes and "none" to drop it. Choices: '
                             + ', '.join(LEGEND_LOCS))

    parser.add_argument('--fit-line', dest='fit_line', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='draw least-squares lines and report their slopes. Unlike --fit-line '
                             'on plot_gross_vs_solid, which reads a calibration back out of the '
                             'table, these are fitted here: the workbook records measured times, '
                             'not a model, so a line describes its panel and nothing more. '
                             '--no-fit-line leaves the markers alone')
    parser.add_argument('--fit', nargs=2, metavar=('LEFT', 'RIGHT'), choices=list(FIT_MODES),
                        default=[p.get('fit', 'campaign') for p in COST_PANELS],
                        help='how --fit-line fits each panel: "campaign" for one line per '
                             'campaign, stopping at the ends of its own data, or "pooled" for a '
                             'single line through every load. Either way every fit is drawn in the '
                             'same neutral grey, so hue stays campaign\'s alone. The defaults '
                             'differ by panel because the data does: against images cloud '
                             'generation costs 42.0 s/image in 2016 against 11.1 in 2021 (Chow '
                             'p=1.1e-15), and since the campaigns barely overlap in x a pooled '
                             'line would be fitted largely to the gap between them; against points '
                             'the slopes are 1.72e-05 vs 1.54e-05 (p=0.22) over a shared range, '
                             'where splitting would imply a difference that was looked for and not '
                             'found')
    return parser


def read_table(path: str, sheet) -> pd.DataFrame:
    """The workbook, or a CSV if that is what was passed. Extension decides; nothing is sniffed."""
    if os.path.splitext(path)[1].lower() in ('.csv', '.txt'):
        return pd.read_csv(path)
    try:
        return pd.read_excel(path, sheet_name=int(sheet) if str(sheet).isdigit() else sheet)
    except ImportError as missing:
        raise SystemExit('{}\nreading .xlsx needs openpyxl: pip install openpyxl (it is in '
                         'a dependency), or export the sheet to .csv'.format(missing))


def panel_specs(args) -> list[dict]:
    """The two panels, defaults filled in only where the column is still the default one.

    A label describing images is wrong the moment --x points the panel at something else, so it is
    inherited only when the column is unchanged; otherwise the column name labels its own axis and
    --xlabels/--ylabels override either way.
    """
    panels = []
    for index, default in enumerate(COST_PANELS):
        x, ys = args.x[index], args.y[index]
        default_ys = [s['y'] for s in panel_series(default)]
        panel = dict(x=x, series=[dict(y=y) for y in ys],
                     title=default['title'], fit=args.fit[index],
                     xlabel=default['xlabel'] if x == default['x'] else x,
                     # Tied to the column, exactly as the label is: a scale chosen for point
                     # counts would silently mislabel any other quantity put on that axis
                     xscale=default.get('xscale', 1.0) if x == default['x'] else 1.0,
                     # A y label naming the stage is right only while the panel still plots it;
                     # with several series it is the axis that is labelled, not any one of them
                     ylabel=default['ylabel'] if ys == default_ys else ' / '.join(ys))
        if args.xlabels:
            panel['xlabel'] = args.xlabels[index]
        if args.x_scale:
            panel['xscale'] = args.x_scale[index]
        if args.ylabels:
            panel['ylabel'] = args.ylabels[index]
        panels.append(panel)
    return panels


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.table):
        print('no table at {}'.format(args.table))
        return 1

    table = read_table(args.table, args.sheet)
    if CAMPAIGN_COLUMN in table.columns and 'campaign' not in table.columns:
        table = table.rename(columns={CAMPAIGN_COLUMN: 'campaign'})
    if 'campaign' not in table.columns:
        print('{} has no {} column; it is what colours the markers'.format(
            args.table, CAMPAIGN_COLUMN))
        return 1

    panels = panel_specs(args)
    wanted = [c for p in panels for c in [p['x']] + [s['y'] for s in panel_series(p)]]
    missing = [c for c in wanted if c not in table.columns]
    if missing:
        print('{} has no column(s) {}.\ncolumns present: {}'.format(
            args.table, ', '.join(missing), ', '.join(map(str, table.columns))))
        return 1

    table['campaign'] = campaign_labels(table['campaign'])
    shown = table[table['campaign'].isin(args.campaigns or CAMPAIGN_YEARS)]

    # Checked before drawing, because the failure this guards against is a *blank* figure with a
    # correct-looking legend -- which is written successfully and looks like a data problem
    if shown.empty:
        print('nothing to plot: no row of {} has a campaign in {}.\ncampaign values present: '
              '{}'.format(args.table, ', '.join(args.campaigns or CAMPAIGN_YEARS),
                          ', '.join(sorted(set(table['campaign'].astype(str))))))
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    path = cost_vs_size(
        table, args.out, panels=panels,
        figsize=tuple(args.figsize) if args.figsize else None,
        font_size=args.font_size,
        title=COST_TITLE if args.title is UNSET else args.title,
        panel_titles=args.panel_titles, campaigns=args.campaigns,
        fit_line=args.fit_line, legend_loc=args.legend_loc, dpi=args.dpi)

    print('{}  ({} truckload(s))'.format(path, len(shown)))
    for panel in panels:
        series = panel_series(panel)
        frame = shown.dropna(subset=[panel['x']] + [s['y'] for s in series])
        for item in series:
            print('  {} vs {}   [{}]'.format(item['y'], panel['x'], item['marker']))
            for campaign, group in frame.groupby('campaign'):
                print('    {} campaign ({:>4}):  {:>2} loads   {} {:>10.4g}-{:<10.4g} '
                      '{} {:>8.4g}-{:<8.4g}'.format(
                          campaign, COLOUR_NAME.get(campaign, '?'), len(group),
                          panel['x'], group[panel['x']].min(), group[panel['x']].max(),
                          item['y'], group[item['y']].min(), group[item['y']].max()))
            if not args.fit_line:
                continue
            # Reported per campaign or pooled to match exactly what was drawn, so a slope quoted
            # from this output is always the slope of a line that is on the figure
            fitted = ([(c, g) for c, g in frame.groupby('campaign')] if panel['fit'] == 'campaign'
                      else [('all {} loads'.format(len(frame)), frame)])
            for label, group in fitted:
                if len(group) < 3:
                    print('    {}: too few loads to fit'.format(label))
                    continue
                slope, intercept, r2 = linear_fit(group[panel['x']], group[item['y']])
                print('    least squares, {:>12}: slope {:.4g} s/{}, intercept {:.4g} s, '
                      'R2 {:.3f}'.format(label, slope, panel['x'], intercept, r2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
