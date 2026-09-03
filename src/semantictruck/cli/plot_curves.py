"""Manuscript Figure 3: per-epoch cross-validation curves of a segmentation campaign.

    semantictruck-plot-curves --metric miou --ylim 70 100
    semantictruck-plot-curves --metric miou acc loss --ylim 70 100 80 100 auto \\
        --titles "(a) IoU" "(b) Overall accuracy" "(c) Loss"

Reads a `metrics_per_epoch.csv` with one row per (fold, epoch) and draws one panel per metric:
every fold dashed and faint, the mean over folds solid, colour coding train against test.

**Model-agnostic.** The segmentation model is not part of this package; what this reads is the
shape of the numbers any k-fold training driver reports, so a campaign is plotted by dropping its
results directory where the command can find it. Column names differ between frameworks, so
`--metric` is resolved case-insensitively against the file.

**Rates are drawn in percent.** Accuracy and the IoU metrics are multiplied by 100 and their axis
label carries `(%)`; loss is left alone. `--ylim` for such a metric is therefore given in percent --
a range that is plainly a fraction is converted, with a note, rather than drawing a sliver of empty
axis.

The held-out fold is labelled **Test**, not validation: nothing is selected on it, since a k-fold
campaign keeps the last epoch's weights rather than the best.
"""


import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from semantictruck.training.curves import DEFAULT_PREFIXES, plot_cross_val_metric, pretty_metric

# Last-resort default, used only when neither --results-root nor the environment says otherwise
DEFAULT_ROOT = 'results'
CSV_NAME = "metrics_per_epoch.csv"
CAMPAIGN_GLOB = "*_results"

# Metrics that are rates in [0, 1] and are plotted as percentages. Matched on the head token, so
# `iou_wood` follows `iou` and `acc_truck` follows `acc`; `loss` is not one, and neither are the
# point counts and coverage figures these tables also carry -- which is why this is an explicit
# list and not "scale everything that is not the loss". F1 is left as it is; add it if a figure
# needs it.
PERCENT_METRICS = ("acc", "oa", "macc", "iou", "miou")
PERCENT_SCALE = 100.0

# Ticks: one decimal is the right resolution once a 0.98 has become a 98.0, two while it has not
PERCENT_DECIMALS = 1
DEFAULT_DECIMALS = 2

# Where the legend may be put, matplotlib's own names. `best` is a search for the spot that
# overlaps the curves least, which is the right default while the panels' shape is not known;
# anything else, or a pair of coordinates, pins it.
LEGEND_LOCS = ("best", "upper left", "upper center", "upper right", "center left", "center",
               "center right", "lower left", "lower center", "lower right")

# The corner of the legend box that lands on an explicit `--legend-loc X Y`
LEGEND_ANCHOR = "upper left"

# What `--legend` takes instead of a panel address
LEGEND_KEYWORDS = ("first", "all", "none")

# What `miou` is called on the axis. The panels show a two-class mean IoU, and this campaign's
# tables and manuscript call that IoU.
METRIC_LABELS = {"miou": "IoU"}


def default_root():
    """The results root: `$SEMANTICTRUCK_RESULTS`, else `$SEMANTICTRUCK_DATA`, else `DEFAULT_ROOT`.

    Two variables because training campaigns are often run from the dataset directory rather than
    from a results one.
    """
    return (os.environ.get("SEMANTICTRUCK_RESULTS")
            or os.environ.get("SEMANTICTRUCK_DATA")
            or DEFAULT_ROOT)


def find_csvs(paths, root, csv_name=CSV_NAME):
    """Resolve the `--csv` arguments to CSV files, or discover them under `root` when empty.

    Each argument may be the CSV itself or a campaign directory holding `csv_name`.

    Raises:
        SystemExit: an argument does not resolve to an existing file, or the discovery under
            `root` finds nothing.
    """
    if paths:
        resolved = []
        for path in paths:
            path = os.path.expanduser(path)
            candidate = os.path.join(path, csv_name) if os.path.isdir(path) else path
            if not os.path.isfile(candidate):
                raise SystemExit("no such metrics file: {}".format(candidate))
            resolved.append(candidate)
        return resolved

    root = os.path.expanduser(root)
    found = sorted(glob.glob(os.path.join(root, CAMPAIGN_GLOB, csv_name)))
    if not found:
        raise SystemExit(
            "no {}/{} under {}; pass --csv or --results-root".format(
                CAMPAIGN_GLOB, csv_name, root
            )
        )
    return found


def resolve_sets(df, metric, prefixes=DEFAULT_PREFIXES):
    """Build the ``{label: column}`` mapping for `metric`, tolerating the campaigns' casing.

    Returns None when neither set's column is present, so a caller can skip that file.
    """
    lookup = {c.lower(): c for c in df.columns}
    sets = {}
    for label, prefix in prefixes:
        column = lookup.get("{}_{}".format(prefix, metric).lower())
        if column is not None:
            sets[label] = column
    return sets or None


def parse_legend_panels(tokens, n_rows, n_cols):
    """`--legend` -> the set of ``(row, col)`` panels that carry the legend.

    Takes a word -- `first` the top-left panel, `all` every panel, `none` no panel -- or a panel
    address: `ROW COL`, or a single row-major index, which is the readable form for a
    single-campaign figure whose row is always 0. Both accept a negative index, counted from the
    end, so `--legend -1` is the last panel however many metrics were asked for.

    Raises:
        SystemExit: the tokens are neither a keyword nor an address inside the grid.
    """
    panels = {(r, c) for r in range(n_rows) for c in range(n_cols)}

    word = " ".join(tokens).strip().lower() if tokens else "first"
    if word == "all":
        return panels
    if word == "none":
        return set()
    if word == "first":
        return {(0, 0)}

    try:
        numbers = [int(token) for token in tokens]
    except ValueError:
        raise SystemExit("--legend '{}' is neither a panel address (ROW COL, or one row-major "
                         "index) nor one of: {}".format(word, ", ".join(LEGEND_KEYWORDS)))

    if len(numbers) == 1:
        index = numbers[0] + (n_rows * n_cols if numbers[0] < 0 else 0)
        row, col = divmod(index, n_cols)
    elif len(numbers) == 2:
        row, col = (n + size if n < 0 else n
                    for n, size in zip(numbers, (n_rows, n_cols)))
    else:
        raise SystemExit("--legend takes a keyword, one panel index or a ROW COL pair, got {} "
                         "numbers".format(len(numbers)))

    if (row, col) not in panels:
        raise SystemExit("--legend: panel ({}, {}) is outside the {} x {} grid this figure has "
                         "({} campaign(s) x {} metric(s))".format(
                             row, col, n_rows, n_cols, n_rows, n_cols))
    return {(row, col)}


def parse_legend_loc(tokens):
    """`--legend-loc` -> the ``(loc, bbox_to_anchor)`` pair the legend is drawn with.

    Either one of `LEGEND_LOCS`, or two numbers in axes coordinates -- (0, 0) the bottom left of
    the panel, (1, 1) its top right -- which put the legend box's upper-left corner there. The
    second form is what pins it to a specific spot inside the first subplot; values outside [0, 1]
    are allowed and place it outside the axes.

    Raises:
        SystemExit: the tokens are neither a known location nor a pair of numbers.
    """
    if not tokens:
        return "best", None

    # The names are two words, so they are matched on the joined tokens: `--legend-loc lower right`
    # has to work unquoted, which is how anyone will type it
    loc = " ".join(tokens).strip().lower()
    if loc in LEGEND_LOCS:
        return loc, None

    if len(tokens) == 2:
        try:
            return LEGEND_ANCHOR, (float(tokens[0]), float(tokens[1]))
        except ValueError:
            pass

    raise SystemExit("--legend-loc '{}' is neither two numbers in axes coordinates nor one of: "
                     "{}".format(" ".join(tokens), ", ".join(LEGEND_LOCS)))


def is_percent(metric):
    """Is this metric a rate to be drawn in percent? Read from its head token, so `iou_wood` is."""
    return metric.partition("_")[0].lower() in PERCENT_METRICS


def to_percent(df, metrics, prefixes=DEFAULT_PREFIXES):
    """A copy of `df` with the percent metrics' columns in percent rather than in [0, 1].

    Done once per campaign, on the columns those metrics actually resolve to, so everything
    downstream -- the shared y range, the curves, the ticks -- is already in the plotted unit. Only
    the requested metrics are touched: these tables carry columns that are neither rates nor
    requested -- a point count, say -- and blanket-scaling would corrupt them.
    """
    columns = []
    for metric in metrics:
        if is_percent(metric):
            columns.extend((resolve_sets(df, metric, prefixes) or {}).values())

    columns = list(dict.fromkeys(columns))          # a metric asked for twice must not scale twice
    if not columns:
        return df

    df = df.copy()
    df[columns] = df[columns] * PERCENT_SCALE
    return df


def axis_label(metric):
    """The y label of a metric's column, carrying `(%)` when it is drawn as a percentage."""
    label = METRIC_LABELS.get(metric) or pretty_metric(metric)
    if not is_percent(metric):
        return label
    # `iou_wood` prettifies to "IoU (wood)", where a second bracket would read badly
    return "{}, %)".format(label[:-1]) if label.endswith(")") else "{} (%)".format(label)


def ytick_decimals(metric, requested):
    """Decimals on the y ticks: what `--ytick-decimals` says, else what the metric's unit wants."""
    if requested is not None:
        return None if requested < 0 else requested
    return PERCENT_DECIMALS if is_percent(metric) else DEFAULT_DECIMALS


def shared_ylim(campaigns, metric, show_folds=True, pad=0.05):
    """The y range covering `metric` in every campaign, so a column of panels is comparable.

    Computed from the data rather than by stretching the axes afterwards, because anything a panel
    places relative to its limits -- tick formatting, the end-of-line labels' dodge -- is resolved
    when it is drawn.

    Returns None when no campaign has the metric, or when the range is degenerate.
    """
    lo, hi = None, None
    for _, df, _ in campaigns:
        sets = resolve_sets(df, metric)
        if sets is None:
            continue
        values = df[list(sets.values())]
        if not show_folds:
            values = values.groupby(df["epoch"]).mean()
        low, high = values.min().min(), values.max().max()
        lo = low if lo is None else min(lo, low)
        hi = high if hi is None else max(hi, high)
    if lo is None or not hi > lo:
        return None
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def parse_ylims(tokens, n_metrics):
    """Turn the `--ylim` tokens into one ``(lo, hi)`` -- or None -- per metric.

    Tokens are read left to right: `auto` stands for one metric's automatic range, any other token
    is one bound of a pair and is followed by the other. A single pair applies to every metric,
    which only makes sense when they share a scale -- hence `--ylim 0.7 1.0 auto 0 1.6` for three.
    Each pair is sorted, so writing it high-first cannot end up inverting the axis.

    Raises:
        SystemExit: a token is neither `auto` nor a number, a pair is left half-written, or the
            count matches neither 1 nor `n_metrics`.
    """
    if not tokens:
        return [None] * n_metrics

    limits = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() in ("auto", "none"):
            limits.append(None)
            i += 1
            continue
        try:
            # Sorted, because matplotlib reads a descending pair as "invert this axis" and would
            # draw the loss curve rising towards zero
            limits.append(tuple(sorted((float(tokens[i]), float(tokens[i + 1])))))
        except IndexError:
            raise SystemExit("--ylim: '{}' has no upper bound".format(tokens[i]))
        except ValueError:
            raise SystemExit("--ylim: '{}' is not a number or 'auto'".format(tokens[i]))
        i += 2

    if len(limits) == 1:
        return limits * n_metrics
    if len(limits) != n_metrics:
        raise SystemExit(
            "--ylim gives {} range(s) for {} metric(s); pass one, or one per metric "
            "(use 'auto' to skip a metric)".format(len(limits), n_metrics)
        )
    return limits


def scale_ylims(limits, metrics):
    """Put each `--ylim` in the unit its metric is drawn in.

    A percent metric's range is given in percent. A pair that is plainly a fraction is the old
    form, so it is converted and the conversion announced -- taking it literally would draw a
    one-percent sliver of empty axis, and a caller would have to guess why.
    """
    scaled = []
    for metric, limit in zip(metrics, limits):
        if limit is not None and is_percent(metric) and limit[1] <= 1.0:
            print("--ylim {:g} {:g} for '{}' reads as a fraction; drawing {:g} to {:g} %".format(
                limit[0], limit[1], metric, limit[0] * PERCENT_SCALE, limit[1] * PERCENT_SCALE))
            limit = (limit[0] * PERCENT_SCALE, limit[1] * PERCENT_SCALE)
        scaled.append(limit)
    return scaled


def resolve_titles(titles, n_rows, n_cols):
    """Map the `--titles` arguments onto panels, as ``{(row, col): title}``.

    Accepts one title per panel (row-major), one per column, or one per row -- the last two put
    the title on the panels of that column or on the first panel of that row.

    Raises:
        SystemExit: the count matches none of those.
    """
    if n_rows * n_cols == len(titles):
        return {(r, c): titles[r * n_cols + c] for r in range(n_rows) for c in range(n_cols)}
    if len(titles) == n_cols:
        return {(r, c): titles[c] for r in range(n_rows) for c in range(n_cols)}
    if len(titles) == n_rows:
        return {(r, 0): titles[r] for r in range(n_rows)}
    raise SystemExit(
        "--titles has {} entries; expected {} (one per panel), {} (one per metric) or {} "
        "(one per campaign)".format(len(titles), n_rows * n_cols, n_cols, n_rows)
    )


def campaign_name(csv_path):
    """`.../ptv3_results/metrics_per_epoch.csv` -> `ptv3`.

    The directory name, with the `_results` suffix removed. Deliberately not a lookup table of
    known model names: this command is model-agnostic, and `--titles` is there for a campaign whose
    directory is not called what you want the panel to say.
    """
    return os.path.basename(os.path.dirname(csv_path)).replace("_results", "")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", nargs="+", default=None,
                        help="metrics CSV files, or campaign directories holding one; "
                             "one panel each. Default: discovered under --results-root")
    parser.add_argument("--results-root", default=default_root(),
                        help="directory holding the campaign folders "
                             "(default: $SEMANTICTRUCK_RESULTS, then $SEMANTICTRUCK_DATA, "
                             "then %(default)s)")
    parser.add_argument("--csv-name", default=CSV_NAME,
                        help="file name looked for inside a campaign directory "
                             "(default: %(default)s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="campaign names, one per input, used to build the default titles. "
                             "Default: from the folder name")
    parser.add_argument("--titles", nargs="+", default=None,
                        help="bold titles drawn above the panels: one per panel (row-major), one "
                             "per metric, or one per campaign. Default: the campaign name")
    parser.add_argument("--no-title", action="store_true",
                        help="draw no title above the panels, e.g. when a single campaign goes "
                             "into a manuscript figure that is captioned instead")
    parser.add_argument("--metric", nargs="+", default=["miou"],
                        help="metric suffixes, e.g. miou acc loss iou_wood; one column each. "
                             "The metric name becomes that panel's y label")
    parser.add_argument("--ylim", nargs="+", default=None, metavar="LO HI",
                        help="one LO HI pair, applied to every metric, or one per metric in the "
                             "order given -- 'auto' leaves a metric's range automatic, e.g. "
                             "--metric miou acc loss --ylim 0.7 1.0 0.7 1.0 auto")
    parser.add_argument("--figsize", nargs=2, type=float, default=(7.0, 4.5),
                        metavar=("W", "H"),
                        help="size of one panel in inches; the figure is this times the number of "
                             "metrics and campaigns (default: %(default)s)")
    parser.add_argument("--ytick-decimals", type=int, default=None,
                        help="decimals on the y tick labels; -1 for matplotlib's own formatting "
                             "(default: {} for the percentage metrics, {} for the rest)".format(
                                 PERCENT_DECIMALS, DEFAULT_DECIMALS))
    parser.add_argument("--legend", nargs="+", default=None, metavar="WHERE",
                        help="which panel carries the legend: {}, or the panel itself as ROW COL "
                             "(campaign, metric; both 0-based, negatives count from the end) or "
                             "as one row-major index (default: first)".format(
                                 ", ".join(LEGEND_KEYWORDS)))
    parser.add_argument("--legend-loc", nargs="+", default=None, metavar="LOC",
                        help="where that legend sits: a matplotlib location -- {} -- or two "
                             "numbers X Y in axes coordinates, (0, 0) being the bottom left of "
                             "the panel and (1, 1) its top right, which pin the legend's "
                             "{} corner there (default: best)".format(
                                 ", ".join(LEGEND_LOCS), LEGEND_ANCHOR))
    parser.add_argument("--out", default=None,
                        help="save the figure here instead of only showing it")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-folds", action="store_true",
                        help="draw only the mean lines")
    args = parser.parse_args()

    legend_loc, legend_bbox = parse_legend_loc(args.legend_loc)

    paths = find_csvs(args.csv, args.results_root, args.csv_name)
    if args.labels and len(args.labels) != len(paths):
        raise SystemExit(
            "--labels has {} entries for {} inputs".format(len(args.labels), len(paths))
        )

    campaigns = []
    for i, path in enumerate(paths):
        name = args.labels[i] if args.labels else campaign_name(path)
        campaigns.append((name, pd.read_csv(path), path))

    # A metric absent from every input costs a blank column, so drop it up front
    metrics = []
    for metric in args.metric:
        if any(resolve_sets(df, metric) for _, df, _ in campaigns):
            metrics.append(metric)
        else:
            print("skipping metric '{}': no input has a column for it".format(metric))
    if not metrics:
        raise SystemExit("no input file has a column for any of {}".format(args.metric))

    # Percent before anything reads the values: the shared range and the tick labels are both
    # resolved from them
    campaigns = [(name, to_percent(df, metrics), path) for name, df, path in campaigns]

    # One row per campaign, one column per metric. The metric names the y axis of its column; the
    # title above the panel is left to the campaign, or to whatever --titles says.
    n_rows, n_cols = len(campaigns), len(metrics)
    panel_w, panel_h = args.figsize
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False
    )

    # After the grid is known, so an address can be checked against it rather than silently
    # landing on a panel that does not exist
    legend_panels = parse_legend_panels(args.legend, n_rows, n_cols)

    limits = scale_ylims(parse_ylims(args.ylim, n_cols), metrics)
    ylims = {}
    for metric, limit in zip(metrics, limits):
        # Without an explicit range, panels of the same metric share one so the campaigns can be
        # compared by eye
        ylims[metric] = limit or (
            shared_ylim(campaigns, metric, not args.no_folds) if n_rows > 1 else None
        )

    if args.no_title:
        titles = {}
    elif args.titles:
        titles = resolve_titles(args.titles, n_rows, n_cols)
    else:
        titles = {
            (r, 0): "{} - {}-fold cross-validation".format(name, df["fold"].nunique())
            for r, (name, df, _) in enumerate(campaigns)
        }

    for r, (name, df, path) in enumerate(campaigns):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            sets = resolve_sets(df, metric)
            if sets is None:
                print("skipping {}: no column for metric '{}'".format(path, metric))
                if (r, c) in legend_panels:
                    print("  ...which is the panel --legend asked for, so no legend is drawn "
                          "there")
                ax.set_visible(False)
                continue
            plot_cross_val_metric(
                df,
                metric=metric,
                sets=sets,
                ax=ax,
                title=titles.get((r, c)),
                ylabel=axis_label(metric),
                ylim=ylims[metric],
                ytick_decimals=ytick_decimals(metric, args.ytick_decimals),
                show_folds=not args.no_folds,
                legend=(r, c) in legend_panels,
                legend_loc=legend_loc,
                legend_bbox=legend_bbox,
            )

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print("wrote", args.out)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
