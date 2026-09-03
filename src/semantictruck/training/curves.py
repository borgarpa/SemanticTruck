"""Cross-validation curves for a segmentation training campaign.

**Model-agnostic by design.** This reads a `metrics_per_epoch.csv` with one row per (fold, epoch)
and one column per metric and set, named `<set prefix>_<metric>` -- for instance
`eval_train_miou` and `eval_val_miou`. Any k-fold training driver that writes such a file can be
plotted, which is what keeps this figure in the repository while the segmentation model itself is
out of scope: the model is not part of this package, only the shape of the numbers it reported.

The figure is the standard one: every fold faint and dashed, the per-epoch mean across folds solid,
and colour encoding the **set** rather than the fold, so a fold added or removed repaints nothing.

The held-out fold is labelled **Test**, not validation, even though the column prefix says `val`.
In a k-fold campaign nothing is selected on it -- the last epoch's weights are kept, not the best --
so it is a test set. Column names differ between training frameworks (one capitalises class names,
`eval_val_iou_Wood` against `eval_val_iou_wood`), which is why the set-to-prefix mapping is a
parameter and the metric is resolved case-insensitively against the file.
"""


import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# Categorical slots 1 and 2 of the reference palette: a validated adjacent pair (CVD-safe on a
# light surface). Colour encodes the *set*, so a fold added or removed never repaints anything.
SET_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# The default naming convention of both campaigns' metrics_per_epoch.csv. The column prefix stays
# `eval_val` because that is what the drivers wrote, but the label reads **Test**: in a k-fold
# campaign the held-out fold is never used to select anything -- both drivers keep `model_last`,
# not `model_best`, precisely because of that -- so it is a test set, not a validation set.
DEFAULT_PREFIXES = (("Train", "eval_train"), ("Test", "eval_val"))


def plot_cross_val_metric(
    data,
    metric="miou",
    sets=None,
    prefixes=DEFAULT_PREFIXES,
    fold_col="fold",
    epoch_col="epoch",
    colors=SET_COLORS,
    ax=None,
    figsize=(7.0, 4.5),
    title=None,
    title_weight="bold",
    xlabel="Epoch",
    ylabel=None,
    ylim=None,
    ytick_decimals=2,
    show_folds=True,
    fold_label="Folds",
    fold_alpha=0.35,
    fold_linewidth=1.0,
    mean_linewidth=2.0,
    mean_requires_all_folds=True,
    annotate_final=False,
    corner_label=None,
    corner_fontsize=12,
    legend=True,
    legend_loc="best",
    legend_bbox=None,
    grid=True,
):
    """Plot one metric per epoch over the folds of a k-fold cross-validation.

    Each fold contributes one dashed, semi-transparent line per set; the per-epoch mean over the
    folds is drawn solid in the same colour. Epochs are taken as they appear in `epoch_col`, so a
    0-based or 1-based CSV both plot correctly.

    Args:
        data: a DataFrame, or a path to a `metrics_per_epoch.csv`, holding one row per
            (fold, epoch). Extra columns are ignored.
        metric: metric suffix, appended to each prefix in `prefixes` to build the column names
            (`"miou"` -> `eval_train_miou`, `eval_val_miou`). Ignored when `sets` is given.
        sets: explicit ``{label: column}`` mapping, overriding `metric`/`prefixes`. Use it when the
            columns do not follow the convention, or to plot a single set. Order sets the colour
            and legend order.
        prefixes: ``(label, column prefix)`` pairs used to build `sets` from `metric`.
        fold_col, epoch_col: column names of the fold id and the epoch index.
        colors: one colour per set, in the order the sets are given. Cycled if too short.
        ax: axes to draw on. A new figure is created when None.
        figsize: size of that new figure. Ignored when `ax` is given.
        title: axes title, drawn above the frame. None leaves it unset.
        title_weight: font weight of that title.
        xlabel, ylabel: axis labels. `ylabel=None` falls back to a prettified `metric`.
        ylim: y limits, e.g. ``(0.5, 1.0)`` to crop the noisy first epochs out of the view.
        ytick_decimals: decimals shown on the y tick labels, so a column of panels does not mix
            `0.95` with `0.9750`. Raise it, or set None for matplotlib's own formatting, when the
            ticks are closer together than the rounding -- neighbours would print the same label.
        show_folds: draw the per-fold lines. False leaves only the means.
        fold_label: legend entry for those lines, drawn once whatever the number of folds.
        fold_alpha, fold_linewidth: styling of the per-fold lines.
        mean_linewidth: width of the mean lines.
        mean_requires_all_folds: average only over the epochs where every fold that has this
            column has a value for it. Keeps the tail of the mean from silently being the curve of
            the one fold that ran longest. Set False to average over whatever is present.
        annotate_final: label each mean line with its set and final value past its right end,
            adding a gutter to the x range to fit them. Off by default: the legend already carries
            the identity, and the numbers belong in the text rather than on the curves.
        corner_label: text drawn in black inside the top-left corner of the axes, naming what the
            panel shows. Meant for a row of panels that differ only in the metric, where a per-axes
            title would compete with the figure's own. `True` uses `pretty_metric(metric)`.
        corner_fontsize: size of that corner label.
        legend: draw the legend (sets, plus one entry for the per-fold lines).
        legend_loc: matplotlib legend location. The default keeps it off the curves, but it does
            not see the end-of-line labels -- pin it explicitly if the two collide.
        legend_bbox: `bbox_to_anchor` for that legend, in axes coordinates -- (0, 0) the bottom
            left of the panel, (1, 1) its top right. With `legend_loc` naming a corner, it is that
            corner of the legend box that lands on the point, so the pair places the legend
            anywhere inside the panel, or outside it. None leaves the placement to `legend_loc`.
        grid: draw a recessive horizontal grid.

    Returns:
        The Axes the plot was drawn on.

    Raises:
        KeyError: a requested column is missing from `data`.
        ValueError: `data` holds no rows for the requested columns.
    """
    df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data

    if sets is None:
        sets = {label: "{}_{}".format(prefix, metric) for label, prefix in prefixes}

    missing = [c for c in (fold_col, epoch_col, *sets.values()) if c not in df.columns]
    if missing:
        raise KeyError(
            "columns {} not in the metrics table; available: {}".format(
                missing, sorted(df.columns)
            )
        )

    # Counted before the rows without a single value are dropped, so the warning below can tell
    # "this fold was never run" from "this fold has no value for this metric"
    n_all_folds = df[fold_col].nunique()

    df = df.dropna(subset=list(sets.values()), how="all")
    if df.empty:
        raise ValueError("no rows with a value for any of {}".format(list(sets.values())))

    folds = sorted(df[fold_col].unique())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    finals = []
    for i, (label, column) in enumerate(sets.items()):
        color = colors[i % len(colors)]

        if show_folds:
            for fold in folds:
                sub = df[df[fold_col] == fold].sort_values(epoch_col)
                ax.plot(
                    sub[epoch_col],
                    sub[column],
                    color=color,
                    linestyle="--",
                    linewidth=fold_linewidth,
                    alpha=fold_alpha,
                    zorder=1,
                )

        # Per column, because a fold can be missing one metric and not another -- one observed
        # campaign logged mIoU every epoch but per-class IoU only sometimes. Nothing on the figure
        # says how many folds went into a mean, so say it here rather than let it pass silently.
        present = df[df[column].notna()]
        n_folds = present[fold_col].nunique()
        if n_folds < n_all_folds:
            warnings.warn(
                "{}: mean over {} of the {} folds, the rest have no value for it".format(
                    column, n_folds, n_all_folds
                )
            )
        mean = present.groupby(epoch_col)[column].mean().sort_index()
        if mean_requires_all_folds:
            per_epoch = present.groupby(epoch_col)[fold_col].nunique()
            mean = mean[per_epoch.reindex(mean.index) == n_folds]
        ax.plot(
            mean.index,
            mean.values,
            color=color,
            linestyle="-",
            linewidth=mean_linewidth,
            label=label,
            zorder=3,
        )
        if len(mean):
            finals.append((mean.index[-1], mean.values[-1], color, label))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else pretty_metric(metric))
    if title is not None:
        ax.set_title(title, fontweight=title_weight)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ytick_decimals is not None:
        # Drop matplotlib's 2.5-based tick steps, whose values (0.025, 0.075, ...) survive the
        # rounding below only as 0.03 / 0.08. What is left lands on round numbers.
        ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.{}f".format(ytick_decimals)))

    # After the limits are final: the dodge and the room made for the labels both depend on them
    if annotate_final:
        _label_line_ends(ax, finals)

    if grid:
        ax.grid(axis="y", color="#d9d8d4", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if corner_label is not None and corner_label is not False:
        ax.text(
            0.02,
            0.97,
            pretty_metric(metric) if corner_label is True else corner_label,
            transform=ax.transAxes,
            color="black",
            fontsize=corner_fontsize,
            va="top",
            ha="left",
            zorder=5,
        )

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if show_folds and len(folds) > 1:
            handles.append(
                Line2D([], [], color="#8a8981", linestyle="--", linewidth=fold_linewidth)
            )
            labels.append(fold_label)
        ax.legend(handles, labels, frameon=False, loc=legend_loc, bbox_to_anchor=legend_bbox)

    return ax


def _label_line_ends(ax, finals, min_gap=0.05, room=0.22):
    """Write `label value` past the right end of each mean line, dodged so they do not overlap.

    `finals` is a list of (x, y, color, label). `min_gap` is the smallest vertical separation
    between two labels, and `room` the share of the x range added on the right to fit them; both
    are fractions of the current axis span, so the placement is independent of the metric's units.
    """
    if not finals:
        return

    x_lo, x_hi = ax.get_xlim()
    ax.set_xlim(x_lo, x_hi + room * (x_hi - x_lo))
    # The added room is a label gutter, not data: keep the ticks on the range the curves cover
    ax.set_xticks([t for t in ax.get_xticks() if x_lo <= t <= x_hi])
    y_lo, y_hi = ax.get_ylim()
    gap = min_gap * (y_hi - y_lo)

    # Place from the bottom up, pushing each label above the previous one if they would collide
    targets = {}
    previous = None
    for i in sorted(range(len(finals)), key=lambda k: finals[k][1]):
        y = finals[i][1]
        if previous is not None and y - previous < gap:
            y = previous + gap
        targets[i] = y
        previous = y

    for i, (x, y, color, label) in enumerate(finals):
        ax.annotate(
            "{} {:.3f}".format(label, y),
            xy=(x, y),
            xytext=(x + 0.02 * (x_hi - x_lo), targets[i]),
            textcoords="data",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
            annotation_clip=False,
        )


def pretty_metric(metric):
    """`iou_wood` / `miou` -> a readable axis or panel label."""
    known = {
        "miou": "mIoU",
        "iou": "IoU",
        "acc": "Accuracy",
        "oa": "Overall accuracy",
        "macc": "Mean accuracy",
        "loss": "Loss",
        "macrof1": "Macro F1",
        "f1": "F1",
    }
    if metric in known:
        return known[metric]
    head, _, tail = metric.partition("_")
    if head in known:
        return "{} ({})".format(known[head], tail.replace("_", " "))
    return metric.replace("_", " ").capitalize()
