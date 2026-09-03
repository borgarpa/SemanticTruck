"""Every figure this package draws.

Matplotlib only: no Open3D, no geometry, nothing that needs a dataset. These functions take the CSV
tables the pipeline and the sweep write and return the path they drew to, so a figure can be
redrawn on a laptop from results computed on another machine.

Two encodings are used consistently, and are what make the figures readable side by side:

* **Hue is the campaign** (red 2016, blue 2021) wherever a truckload is drawn, and **shape is the
  series** where a panel carries more than one. Shape survives greyscale printing; a third colour
  would not, and would collide with the campaign encoding.
* **Filled means measured, hollow means predicted without that load** -- a leave-one-out
  counterpart drawn at the same x, so the vertical gap within a pair is that load's LOO residual.

**Campaign labels are normalised on read by every function that keys on them.** The column does not
survive a CSV round trip as a string: pandas reads it back as `int64`, or as `float64` (`2016.0`)
if any row is blank. Either way a comparison against the string keys of the colour maps matches
nothing and the figure comes out empty with a correct-looking legend -- written successfully, and
reading as a data problem rather than a plotting one. `campaign_labels()` is the defence; use it
for anything keying on that column.
"""



from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.transforms import ScaledTranslation                    # noqa: E402

from semantictruck.arms import ARM_LABEL, ARMS, PREDICTED, REFERENCE   # noqa: E402

# Categorical slots 1-3, in fixed order. Validated all-pairs: worst CVD dE 9.2, normal-vision
# dE 24.0, and only the aqua sits under 3:1 on this surface, which the legends relieve.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
ARM_COLOR = {REFERENCE: SERIES[0], PREDICTED: SERIES[1]}
CAMPAIGN_COLOR = {'2016': SERIES[0], '2021': SERIES[1]}

SURFACE = '#fcfcfb'
# Pure white, for a figure that will sit on a white page: the off-white above is a hair warm and
# shows as a grey rectangle once the figure is placed on a white background or a slide
WHITE = '#ffffff'
INK = '#0b0b0b'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# A dark hairline around every filled marker. Two dots of the same colour that overlap read as one
# blob without it, which is exactly what happens where a campaign's loads cluster; the edge also
# lifts a marker off the grid line it happens to sit on. Thin and near-black rather than heavy, so
# it defines the marker without darkening the hue inside it.
MARKER_EDGE = INK
MARKER_EDGE_WIDTH = 0.8

# What each arm is called in a legend lives in `semantictruck.arms.ARM_LABEL`, so a figure and
# a printed table cannot disagree about it.
MESH_LABEL = {'poisson': 'Poisson', 'alpha': 'Alpha shape',
              'ball_pivoting': 'Ball pivoting', 'hull': 'Convex hull'}


def _style(ax, title=None, xlabel=None, ylabel=None, text_color=MUTED, frame_color=AXIS,
           tick_size=9, label_size=10, title_size=12, title_loc='left', surface=SURFACE):
    """Shared axis styling.

    The colour and size arguments default to the muted house palette, so the figures that do not
    pass them are unaffected; `box_whiskers` overrides them because it is the one built for
    export at arbitrary size.

    `title_loc` defaults to left, where a title reads as a heading over the panel it labels. The
    scatter figures centre theirs instead: side by side under a suptitle, two left-set headings
    read as one column of text rather than as a label each.

    `surface` must be passed to `_save` as well, or the axes and the figure patch around them come
    out different colours.
    """
    ax.set_facecolor(surface)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(frame_color)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=text_color, labelsize=tick_size, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=title_size, fontweight='bold', loc=title_loc,
                     pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=text_color, fontsize=label_size)
    if ylabel:
        ax.set_ylabel(ylabel, color=text_color, fontsize=label_size)
    return ax


def _break_mark(ax, edge='bottom', color=AXIS, linewidth=1.0, count=2, offset=16.0,
                size=4.5, gap=5.5):
    """The slashes that say a y-axis does not run continuously through this edge.

    Without them a cut is a silent edit: the boxes look further apart than they are, a whisker
    that was severed leaves no trace, and nothing on the page says the scale jumps.

    `count` slashes are drawn on the left spine at `edge`, `offset` points inside it, `gap` points
    apart. A genuine break -- a band taken out of the middle, so two panels face each other across
    it -- wants one slash on each facing edge at `offset=0`, the two together reading as the one
    break. A cut that runs off the end of the data has no facing panel, so it takes the
    conventional double slash set in from the corner, where it clears the extreme tick label.

    Everything is in **points**, anchored to a corner of the axes through a transform whose input
    is inches from it. Axes fractions would be wrong twice over: the slashes would come out
    slanted by the figure's aspect rather than at 45 degrees, and they would grow with the figure,
    while this one is redrawn at every size a manuscript asks for.
    """
    corner = 0 if edge == 'bottom' else 1
    inward = 1 if edge == 'bottom' else -1
    inches = ax.figure.dpi_scale_trans + ScaledTranslation(0, corner, ax.transAxes)
    for step in range(count):
        centre = inward * (offset + step * gap) / 72
        ax.plot((-size / 72, size / 72), (centre - size / 72, centre + size / 72),
                transform=inches, color=color, linewidth=linewidth, clip_on=False, zorder=6)
    return ax


def _span_ylabel(fig, axes, text, fontsize=10, color=INK, pad=4.0):
    """One y-label centred over several stacked panels, set just clear of their tick labels.

    `fig.supylabel` would be the obvious call and is the wrong one: it places the text at a fixed
    figure fraction, which knows nothing about how wide the tick labels happen to be, and on this
    figure that leaves ~15 points of dead space between the label and the numbers it belongs to.
    An axes' own `ylabel` has no such gap because matplotlib measures the ticks first, so this
    does the same -- draw once, take the leftmost tick label across every panel, and hang the text
    `pad` points to its left, `pad` being matplotlib's own default label padding. Centring is on
    the panels together rather than on either one, which is the whole reason not to use
    `set_ylabel` on one of them.

    The draw is what makes the tick labels measurable; it is the same work `savefig` is about to
    do again, so it costs one render and nothing else.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [label for ax in axes for label in ax.get_yticklabels()
             if label.get_visible() and label.get_text()]
    left = (min(label.get_window_extent(renderer).x0 for label in ticks) if ticks
            else min(ax.bbox.x0 for ax in axes))
    return fig.text((left - pad * fig.dpi / 72) / fig.bbox.width,
                    (min(ax.bbox.y0 for ax in axes) + max(ax.bbox.y1 for ax in axes))
                    / 2 / fig.bbox.height,
                    text, rotation='vertical', ha='right', va='center',
                    color=color, fontsize=fontsize)


def _cut_panels(cut, lowest, highest):
    """The y-ranges that survive `cut`, top panel first, each with the edge it is broken at.

    `cut` names the interval to **remove**, either end `None` for "as far as the data goes" -- so
    `(2.1, None)` drops everything above 2.1 and `(None, 1.75)` everything below 1.75. A band
    strictly inside the data genuinely breaks the axis and two panels come back, the values on
    either side of the cut both still drawn; a band running off one end has nothing on its far
    side, and one clamped panel comes back instead.

    `ValueError` when the band would empty the figure. A cut that removes every box is a typo, and
    a blank panel is a poor way to report one.
    """
    low = -np.inf if cut[0] is None else cut[0]
    high = np.inf if cut[1] is None else cut[1]
    if not low < high:
        raise ValueError('a cut runs from LOW up to HIGH, got {} to {}'.format(cut[0], cut[1]))
    if low <= lowest and high >= highest:
        raise ValueError('cutting [{:.4g}, {:.4g}] removes every value, and they run {:.4g} to '
                         '{:.4g}'.format(low, high, lowest, highest))
    if low <= lowest:
        return [dict(ylim=(high, None), broken='bottom')]
    if high >= highest:
        return [dict(ylim=(None, low), broken='top')]
    return [dict(ylim=(high, None), broken='bottom'),
            dict(ylim=(None, low), broken='top')]


def _save(fig, path, dpi=200, surface=SURFACE):
    # `facecolor` is passed to savefig as well as set on the patch: savefig otherwise takes it from
    # `figure.facecolor` in the rcParams and quietly repaints the margin bbox_inches='tight' keeps
    fig.patch.set_facecolor(surface)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=surface)
    plt.close(fig)
    return path


def _parameter_of(setting_id: str) -> str:
    return setting_id.split('_')[0]


def tornado(summary: pd.DataFrame, path: str) -> str:
    """Settings ranked by |d_rmse_m3|, one panel per mesh method, one bar per arm.

    Each campaign is compared against its **own** BASE, so a panel answers 'how much does this
    parameter matter under this reconstruction' -- and reading across panels answers whether the
    answer depends on the reconstruction at all.
    """
    methods = _methods(summary)
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 7), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, mesh in zip(axes, methods):
        data = summary[(summary['mesh_method'] == mesh) & (summary['setting_id'] != 'BASE')]
        order = (data[data['segmentation'] == REFERENCE]
                 .assign(group=lambda d: d['setting_id'].map(_parameter_of),
                         mag=lambda d: d['d_rmse_m3'].abs())
                 .sort_values(['group', 'mag'], ascending=[True, True])['setting_id'].tolist())

        positions = np.arange(len(order))
        height = 0.38
        for offset, arm in ((-height / 2, REFERENCE), (height / 2, PREDICTED)):
            arm_data = data[data['segmentation'] == arm].set_index('setting_id')
            values = [arm_data['d_rmse_m3'].get(s, np.nan) for s in order]
            ax.barh(positions + offset, values, height=height * 0.94, color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm], zorder=3)

        ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=8)
        _style(ax, mesh, 'Change in RMSE (m³)')

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower right')
    fig.suptitle('Effect of each setting on RMSE, against that method\'s own BASE',
                 color=INK, fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()

    return _save(fig, path)


def ratio_cv(summary: pd.DataFrame, path: str) -> str:
    """Stability of vs_ref / vg by setting. Calibration-free, so it survives any rescaling."""
    order = _setting_order(summary)
    methods = _methods(summary)
    fig, ax = plt.subplots(figsize=(12, 5.5))

    fitted = summary[summary['segmentation'] == REFERENCE]
    for slot, mesh in enumerate(methods):
        data = fitted[fitted['mesh_method'] == mesh].set_index('setting_id')
        values = [data['ratio_cv'].get(s, np.nan) for s in order]
        colour = SERIES[slot % len(SERIES)]
        ax.plot(range(len(order)), values, marker='o', markersize=5, linewidth=2,
                color=colour, markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_WIDTH,
                label=mesh, zorder=3)
        ax.axhline(data['ratio_cv'].get('BASE', np.nan), color=colour, linewidth=1,
                   linestyle='--', alpha=0.55, zorder=2)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    _style(ax, 'Coefficient of variation of the solid-to-gross ratio (reference arm)',
           ylabel='ratio CV (dashed = that method\'s BASE)')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2)

    return _save(fig, path)


def paired_d_vg(per_load: pd.DataFrame, path: str, arm: str = REFERENCE,
                mesh: str = 'poisson') -> str:
    """Per-load d_vg_pct by setting, coloured by campaign.

    A parameter whose effect separates by campaign is injecting a load-dependent term the
    regression cannot absorb -- qualitatively worse than a uniform scale shift.

    The campaign column is normalised through `campaign_labels` before anything keys on it. Read
    back from `results_per_load.csv` it is float64 -- `2016.0`, not `'2016'` -- so comparing it
    against `CAMPAIGN_COLOR`'s string keys matched nothing and every one of these figures came out
    empty with a correct-looking legend, which is written successfully and reads as a data problem
    rather than a plotting one.
    """
    per_load = per_load.assign(campaign=campaign_labels(per_load['campaign']))
    per_load = per_load[per_load['mesh_method'] == mesh]
    data = per_load[(per_load['segmentation'] == arm) & (per_load['setting_id'] != 'BASE')]
    order = [s for s in _setting_order(per_load) if s != 'BASE']

    fig, ax = plt.subplots(figsize=(11, 5))
    for campaign, colour in CAMPAIGN_COLOR.items():
        subset = data[data['campaign'] == campaign]
        x = [order.index(s) for s in subset['setting_id']]
        jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(x))
        ax.scatter(np.array(x) + jitter, subset['d_vg_pct'], s=22, alpha=0.75,
                   color=colour, edgecolors=MARKER_EDGE, linewidths=MARKER_EDGE_WIDTH, zorder=3,
                   label='{} ({})'.format(campaign,
                                          'Semitrailer' if campaign == '2016' else 'B-double'))

    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    _style(ax, 'Per-load Δ gross volume against BASE — {}, {}'.format(mesh, ARM_LABEL[arm]),
           ylabel='Δ gross volume (%)')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    return _save(fig, path)


def plane_angle(summary: pd.DataFrame, path: str, mesh: str = 'poisson') -> str:
    """Plane-normal response against volume response, for the alignment parameters."""
    summary = summary[summary['mesh_method'] == mesh]
    alignment = summary[summary['setting_id'].str.startswith(('P05', 'P06', 'P07'))]

    fig, ax = plt.subplots(figsize=(7, 5))
    for slot, group in enumerate(['P05', 'P06', 'P07']):
        subset = alignment[(alignment['setting_id'].str.startswith(group))
                           & (alignment['segmentation'] == REFERENCE)]
        ax.scatter(subset['mean_plane_angle_deg'], subset['d_rmse_m3'], s=90,
                   color=SERIES[slot], edgecolors=MARKER_EDGE, linewidths=MARKER_EDGE_WIDTH,
                   zorder=3,
                   label='{} — {}'.format(group, subset['parameter'].iloc[0]
                                          if len(subset) else group))
        for row in subset.itertuples():
            ax.annotate(row.setting_id, (row.mean_plane_angle_deg, row.d_rmse_m3),
                        textcoords='offset points', xytext=(7, 4), fontsize=8, color=MUTED)

    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)
    _style(ax, 'Base-plane rotation against volume response — {}'.format(mesh),
           'Mean angle to the BASE plane normal (°)', 'Change in RMSE (m³)')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    return _save(fig, path)


def segmentation_gap(summary: pd.DataFrame, path: str) -> str:
    """What automating the segmentation costs in gross volume, setting by setting."""
    order = _setting_order(summary)
    methods = _methods(summary)
    data = summary[summary['segmentation'] == PREDICTED]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(methods), 1)
    for slot, mesh in enumerate(methods):
        values = data[data['mesh_method'] == mesh].set_index('setting_id')['mean_d_vg_seg_pct']
        offset = (slot - (len(methods) - 1) / 2) * width
        ax.bar(np.arange(len(order)) + offset, [values.get(s, np.nan) for s in order],
               width=width * 0.92, color=SERIES[slot % len(SERIES)], label=mesh, zorder=3)

    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    _style(ax, 'Gross volume from model segmentation, against the reference, at the same setting',
           ylabel='mean Δ gross volume (%)')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2)

    return _save(fig, path)


DEFAULT_TITLE = 'Leave-one-out error by reconstruction, across every setting in its campaign'
DEFAULT_YLABEL = 'RMSE of solid volume (m³)'


def _draw_boxes(ax, boxes, width, font_size, show_fliers, show_base):
    """Every box, on one axes. Called once per panel, so a broken axis draws the same artists
    twice and shows a different window on to them -- which is what makes a whisker crossing the
    cut appear on both sides of it."""
    for box in boxes:
        colour = box['colour']
        drawn = ax.boxplot([box['values']], positions=[box['position']], widths=width * 0.82,
                           patch_artist=True, manage_ticks=False, zorder=3,
                           showfliers=show_fliers,
                           flierprops=dict(marker='o', markersize=font_size * 0.45,
                                           markerfacecolor='none', markeredgecolor=colour,
                                           markeredgewidth=1.1),
                           medianprops=dict(color=INK, linewidth=1.6),
                           whiskerprops=dict(color=colour, linewidth=1.2),
                           capprops=dict(color=colour, linewidth=1.2))
        for patch in drawn['boxes']:
            patch.set(facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=1.4)

        # BASE marked on its own box: is this campaign's chosen point a good one?
        if show_base and box['base'] is not None:
            ax.plot([box['position']], [box['base']], marker='D', markersize=font_size * 0.7,
                    color=colour, markeredgecolor=MARKER_EDGE,
                    markeredgewidth=MARKER_EDGE_WIDTH, zorder=5)
    return ax


def box_whiskers(summary: pd.DataFrame, path: str, *,
                       figsize: tuple[float, float] | None = None,
                       font_size: float = 9,
                       title: str | None = DEFAULT_TITLE,
                       ylabel: str = DEFAULT_YLABEL,
                       methods: list[str] | None = None,
                       arms: list[str] | None = None,
                       column: str = 'rmse_loo_m3',
                       cut: tuple[float | None, float | None] | None = None,
                       cut_ratio: float = 0.25,
                       break_mark: bool = True,
                       legend_loc: str = 'best',
                       show_base: bool = True,
                       show_fliers: bool = True,
                       dpi: int = 200) -> str:
    """Spread of leave-one-out RMSE within each reconstruction campaign, arms side by side.

    The tornado ranks settings *within* a campaign against its own BASE. This asks the other
    question: how good is a reconstruction, and how much does its parameterisation matter at all?
    A tall box means the choice of parameters dominates; a short one means the reconstruction
    delivers the same accuracy however it is tuned, which is the more useful property.

    Leave-one-out rather than in-sample, so the boxes are comparable to a quoted accuracy: every
    setting here was fitted on 29 loads and scored on the one held out, 29 times over.

    BASE is drawn on top of its own box, because a campaign whose BASE sits at the bad end of its
    own distribution is one where a better configuration was available and not taken.

    Unlike the other figures here this one is built for export, so its geometry and type are
    arguments: `figsize` in inches, `font_size` as the tick size that the axis labels, title and
    legend are scaled from, `legend_loc` for where the key sits, and `title=None` to drop the
    title for a caption instead. Its text and frame are **black**, not the muted house palette,
    and its surface is **white** rather than the house off-white, which reads as a grey rectangle
    once the figure is placed on a page.

    `cut` names the interval of the y-axis to **remove**, `(low, high)`, either end `None` for as
    far as the data goes. It earns its place because the boxes and the outliers live at different
    scales: a campaign's collapsed settings sit whole m³ above its box, so the full range squashes
    every box into the bottom fifth of the panel and the comparison the figure exists to make
    becomes unreadable. Cutting the empty band between them keeps both -- the boxes at a legible
    scale and the outliers still on the page, at their own height -- and `_cut_panels` says which
    geometry a given band implies. Either way the statistics are untouched: every value counts
    towards its median, quartiles and whiskers wherever the window falls, unlike
    `show_fliers=False`, which takes points out of the drawing but not out of the box.

    `cut_ratio` is the upper panel's share of the plot height. The two panels deliberately do
    **not** share a scale -- at a common one the outlier panel would take three quarters of the
    figure to show four points -- and `_break_mark` declares the discontinuity; pass
    `break_mark=False` for a figure whose caption states the cut instead.

    Campaigns carry different numbers of settings -- 25 / 19 / 22 / 17 for Poisson / alpha / ball
    pivoting / hull, since each owns different parameters -- so the boxes are not like-for-like
    samples. That is a caption's job to say: annotating each box with its count invited the count
    to be read as truckloads, when every point in a box is itself a 29-load fit.
    """
    methods = [m for m in (methods or _methods(summary)) if m in set(summary['mesh_method'])]
    arms = arms or [a for a in ARMS if a in set(summary['segmentation'])]

    if figsize is None:
        figsize = (1.9 * max(len(methods), 1) * max(len(arms), 1) + 2.5, 6)

    # One knob, everything else in proportion, so a larger figure does not need six more flags
    tick_size, label_size = font_size, font_size + 1
    title_size = font_size + 3

    # Gathered before anything is drawn: where the cut falls decides how many panels there are,
    # and that cannot be known until the data range is
    width = 0.8 / max(len(arms), 1)
    boxes, ticks, labels = [], [], []
    for slot, mesh in enumerate(methods):
        for offset_idx, arm in enumerate(arms):
            data = summary[(summary['mesh_method'] == mesh) & (summary['segmentation'] == arm)]
            values = data[column].dropna()
            if values.empty:
                continue
            base = data.loc[data['setting_id'] == 'BASE', column]
            boxes.append(dict(
                position=slot + (offset_idx - (len(arms) - 1) / 2) * width,
                colour=ARM_COLOR.get(arm, SERIES[offset_idx % len(SERIES)]),
                values=values.to_numpy(),
                base=None if base.empty else float(base.iloc[0])))
        ticks.append(slot)
        labels.append(MESH_LABEL.get(mesh, mesh))

    if not boxes:
        raise ValueError('nothing to plot: no rows carry a {}'.format(column))

    panels = ([dict(ylim=None, broken=None)] if cut is None else
              _cut_panels(cut, min(b['values'].min() for b in boxes),
                          max(b['values'].max() for b in boxes)))

    fig, axes = plt.subplots(
        len(panels), 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(height_ratios=[cut_ratio, 1 - cut_ratio], hspace=0.07)
        if len(panels) == 2 else None)
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        _draw_boxes(ax, boxes, width, font_size, show_fliers, show_base)

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels, fontsize=label_size, color=INK)

    for index, (ax, panel) in enumerate(zip(axes, panels)):
        # The title heads the figure and the y-label spans it, so with two panels neither belongs
        # to an axes: the label goes to `supylabel` below, centred over both
        _style(ax, title if index == 0 else None,
               ylabel=ylabel if len(axes) == 1 else None,
               text_color=INK, frame_color=INK, tick_size=tick_size, label_size=label_size,
               title_size=title_size, surface=WHITE)

        if panel['ylim'] is not None:
            # The open side is read back rather than left to `set_ylim`, which would take it from
            # a view limit that may not have been autoscaled yet
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom if panel['ylim'][0] is None else panel['ylim'][0],
                        top if panel['ylim'][1] is None else panel['ylim'][1])

        # The upper panel of a break has no x-axis: its bottom edge is the cut, not zero
        if len(axes) == 2 and index == 0:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)

        if break_mark and panel['broken']:
            _break_mark(ax, panel['broken'], color=INK,
                        **(dict(count=1, offset=0.0) if len(axes) == 2 else {}))

    if len(axes) > 1:
        _span_ylabel(fig, axes, ylabel, fontsize=label_size, color=INK)

    handles = [plt.Line2D([], [], marker='s', linestyle='none', markersize=font_size,
                          markerfacecolor=ARM_COLOR.get(a, SERIES[i % len(SERIES)]),
                          markeredgecolor=ARM_COLOR.get(a, SERIES[i % len(SERIES)]),
                          alpha=0.6, label=ARM_LABEL.get(a, a)) for i, a in enumerate(arms)]
    if show_base:
        handles.append(plt.Line2D([], [], marker='D', linestyle='none',
                                  markersize=font_size * 0.7, color=INK, label='Baseline'))
    # On the lower panel, which is where the boxes the key names are legible
    _legend(axes[-1], handles, legend_loc, tick_size)

    return _save(fig, path, dpi=dpi, surface=WHITE)


GROSS_SOLID_TITLE = 'Solid against gross volume, by acquisition campaign'
GROSS_SOLID_XLABEL = 'Gross volume (m³)'
GROSS_SOLID_YLABEL = 'Solid wood volume (m³)'

# Red and blue, one per campaign. Deliberately **not** `CAMPAIGN_COLOR`, which is the blue/orange
# categorical pair the sweep figures use: this figure was specified in red and blue. Red/blue is
# the one pair that a red-green deficiency separates and a blue-yellow one does not, so the marker
# style carries the other distinction here -- filled for measured, hollow for predicted -- and no
# reading depends on hue alone.
CAMPAIGN_YEARS = ('2016', '2021')
SCATTER_CAMPAIGN_COLOR = {'2016': '#cf2f1e', '2021': '#2a78d6'}


def campaign_labels(campaign: pd.Series) -> pd.Series:
    """Campaign years as the string labels every colour map is keyed on.

    A year is a label, but it does not survive a CSV round trip as one: pandas reads the column
    back as int64, or as **float64** if any row is blank -- and a load with no reference row leaves
    exactly that blank, so `2016` returns as `2016.0`. Either way a string comparison matches
    nothing, every campaign filters to empty, and the figure comes out blank with a correct-looking
    legend. Normalising on read is cheap; noticing an empty plot is not.
    """
    numeric = pd.to_numeric(campaign, errors='coerce')
    labels = campaign.astype(str)
    labels[numeric.notna()] = numeric[numeric.notna()].astype(int).astype(str)
    return labels


def gross_vs_solid(per_truck: pd.DataFrame, path: str, *,
                   figsize: tuple[float, float] | None = None,
                   font_size: float = 9,
                   title: str | None = GROSS_SOLID_TITLE,
                   xlabel: str = GROSS_SOLID_XLABEL,
                   ylabel: str = GROSS_SOLID_YLABEL,
                   segmentation: str = REFERENCE,
                   panel_titles=None,
                   campaigns: list[str] | None = None,
                   colors: dict | None = None,
                   show_loo: bool = True,
                   show_residuals: bool = False,
                   fit_line: bool = False,
                   legend_loc: str = 'best',
                   dpi: int = 200) -> str:
    """Measured solid volume against gross volume, one colour per campaign.

    Two points per truckload, at the same x: a **filled** marker at its measured solid volume and
    a **hollow** one at the volume the leave-one-out calibration predicted from that same gross
    volume. Sharing x is the point -- the vertical gap between the pair is that load's LOO
    residual, read directly off the plot, and `show_residuals` draws it as a hair line when the
    pairing is hard to follow.

    Splitting by campaign is what makes the figure worth drawing rather than a plain fit plot: the
    two are different vehicle configurations, so a single line through both is only honest if the
    colours interleave. A campaign sitting consistently above or below is a load-dependent term
    the pooled regression cannot absorb.

    The hollow markers lie near-collinear by construction -- each is a linear model evaluated at
    one point -- so they read as the calibration itself, and the filled markers as what it had to
    predict. `fit_line` adds the in-data line for reference, taken from the table's `vs_hat_m3`
    rather than refitted here, so it cannot disagree with the model the table reports.
    """
    return _scatter_figure(_VOLUME_SPEC, per_truck, path, figsize=figsize, font_size=font_size,
                           title=title, xlabel=xlabel, ylabel=ylabel, segmentation=segmentation,
                           panel_titles=panel_titles, campaigns=campaigns, colors=colors,
                           show_loo=show_loo, show_residuals=show_residuals, fit_line=fit_line,
                           legend_loc=legend_loc, dpi=dpi)


RESIDUAL_TITLE = 'Prediction error against gross volume, by acquisition campaign'
RESIDUAL_YLABEL = 'Error in solid volume (m³)'


def residual_vs_gross(per_truck: pd.DataFrame, path: str, *,
                      figsize: tuple[float, float] | None = None,
                      font_size: float = 9,
                      title: str | None = RESIDUAL_TITLE,
                      xlabel: str = GROSS_SOLID_XLABEL,
                      ylabel: str = RESIDUAL_YLABEL,
                      segmentation: str = REFERENCE,
                      panel_titles=None,
                      campaigns: list[str] | None = None,
                      colors: dict | None = None,
                      show_loo: bool = True,
                      show_residuals: bool = False,
                      legend_loc: str = 'best',
                      dpi: int = 200) -> str:
    """Prediction error against gross volume, in the same colours and markers as `gross_vs_solid`.

    The companion plot, and the one that actually shows whether the calibration is sound: the fit
    plot's points hug a rising line whatever the residuals do, because both axes carry the same
    scale. Subtracting it leaves the part worth looking at. What to read here --

        a trend in x         the linear model is the wrong shape; error should not depend on load
                             size once the calibration has done its work
        a campaign offset    red sitting above blue is a per-configuration term the pooled fit
                             cannot absorb, and the case for fitting the two separately
        widening spread      heteroscedasticity, which is what `breusch_pagan_p` tests numerically

    Markers keep the meaning they have in `gross_vs_solid`: **filled** is the in-data residual,
    from the calibration fitted on all 29 loads including this one, and **hollow** is the
    leave-one-out residual, from the model refitted without it. The hollow marker is always the
    larger of the two in expectation -- that gap *is* the optimism of an in-sample error, per load
    rather than pooled into one RMSE.
    """
    return _scatter_figure(_RESIDUAL_SPEC, per_truck, path, figsize=figsize, font_size=font_size,
                           title=title, xlabel=xlabel, ylabel=ylabel, segmentation=segmentation,
                           panel_titles=panel_titles, campaigns=campaigns, colors=colors,
                           show_loo=show_loo, show_residuals=show_residuals, fit_line=False,
                           legend_loc=legend_loc, dpi=dpi)


# What separates the two figures: the columns each marker of the pair reads, how the pair is
# named, and whether the panel carries a zero line or a calibration line.
_VOLUME_SPEC = dict(filled='vs_ref_m3', hollow='vs_hat_loo_m3',
                    filled_label='Measured', hollow_label='Leave-one-out prediction',
                    zero_line=False, allow_fit_line=True)
_RESIDUAL_SPEC = dict(filled='error_m3', hollow='error_loo_m3',
                      filled_label='In-data residual', hollow_label='Leave-one-out residual',
                      zero_line=True, allow_fit_line=False)

COST_TITLE = 'Computation time against the size of what it processes, by acquisition campaign'

# The two halves of the method, each against the input that actually drives it: photogrammetric
# reconstruction against the number of images flown, geometric processing against the number of
# points that reconstruction handed it. Column names are the workbook's own, so a panel can be
# repointed at any other pair of its columns without touching the drawing.
COST_PANELS = (
    # The photogrammetry split into its two stages rather than plotted as its total, because the
    # total hides where the campaign difference comes from. Against images, camera alignment
    # costs 47.9 s/image in 2016 against 49.9 in 2021 -- the same slope, and a Chow test does not
    # separate them (p = 0.27). Cloud generation costs 42.0 against 11.1, a factor of 3.8, at
    # p = 1.1e-15. So the whole of the difference visible in `PhotoTotal_sec` is the cloud
    # generation stage, and the sum -- exactly `MatchAlignTime_sec + DenseCloudTime_sec` -- shows
    # one intermediate slope in place of two very different ones.
    dict(x='Images',
         series=(dict(y='MatchAlignTime_sec', marker='o'),
                 dict(y='DenseCloudTime_sec', marker='D')),
         title='Photogrammetric reconstruction',
         xlabel='Images in the survey',
         ylabel='Photogrammetry time (s)',
         fit='campaign'),
    dict(x='Points', y='ProcessTotal_sec',
         title='Geometric processing',
         xlabel='Points in the merged cloud (millions)',
         xscale=1e6,
         ylabel='Processing time (s)',
         fit='pooled'),
)

# Marker shape separates the series inside a panel, in this order. Hue is campaign's and cannot be
# borrowed for it.
COST_MARKERS = ('o', 'D', 's', '^')

# Every fit is solid. A dash pattern per series was tried and dropped: it made the four lines on the
# left panel four visually distinct objects competing with the markers, when the marker shape and
# the line's own position already say which series and which campaign it belongs to. A series can
# still override it by naming `linestyle` in its spec.
FIT_LINESTYLE = '-'

# Readable names for the workbook's timing columns; anything not here labels itself
SERIES_LABEL = {
    'MatchAlignTime_sec': 'Camera alignment',
    'DenseCloudTime_sec': 'Cloud generation',
    'PhotoTotal_sec': 'Photogrammetry total',
    'ProcessTotal_sec': 'Processing total',
    'Inference_sec': 'Segmentation inference',
    'VolumeTotal_sec': 'Volume estimation',
    'Preprocess_sec': 'Preprocessing',
    'Bays_sec': 'Bay split and alignment',
    'Mesh_sec': 'Meshing',
}


def panel_series(panel: dict) -> list[dict]:
    """A panel's series, whether it was written with one `y` or a list of them.

    Keeps the single-series spelling working -- the right-hand panel is one column and reads
    better as `y='ProcessTotal_sec'` than as a one-element tuple -- while letting the left carry
    two. Marker, line style and label are filled in by position where a series does not name them.
    """
    series = list(panel.get('series') or [dict(y=panel['y'])])
    return [dict(y=item['y'],
                 marker=item.get('marker', COST_MARKERS[i % len(COST_MARKERS)]),
                 linestyle=item.get('linestyle', FIT_LINESTYLE),
                 label=item.get('label', SERIES_LABEL.get(item['y'], item['y'])))
            for i, item in enumerate(series)]

# How `fit_line` fits a panel. The default differs between the two, from what the data does:
#
#   Images -> photogrammetry time     one line per campaign. The slopes are 89.9 s/image in 2016
#                                     against 60.9 in 2021, a factor of 1.5, and a Chow test
#                                     rejects the common line at p = 2.5e-4. Worse, the campaigns
#                                     barely overlap in x -- 30-66 images against 108-154 -- so a
#                                     pooled line is fitted mostly to the empty gap between the two
#                                     clusters, and comes out at 42.3 s/image, *below both* of the
#                                     slopes it claims to summarise. It would read as the cost per
#                                     image and be that for neither campaign.
#   Points -> processing time         one pooled line. Here the same test does not reject
#                                     (1.72e-05 against 1.54e-05 s/point, p = 0.22) and the two
#                                     campaigns overlap across most of the x range, so splitting
#                                     would show two lines on top of each other and imply a
#                                     difference that was measured and not found.
FIT_MODES = ('campaign', 'pooled')


def scaled_ticks(scale: float = 1.0):
    """Tick formatter dividing by `scale`, for an axis whose unit lives in its label.

    A cloud of 2 057 488 points sets a tick label eight digits wide, and four of them across the
    foot of a panel is most of its width spent on zeros. At `scale=1e6` the same axis reads
    2, 3, 4 and the label says millions -- the usual convention in print, and the reason the
    scale is not written on to each tick as a `2M` suffix.

    Deliberately not matplotlib's scientific offset either, which parks a lone `1e6` at the end
    of the axis: that is the same information, but placed where a reader has to find it rather
    than in the label they have already read. `scale=1` leaves the numbers exactly as they are,
    which is what an axis counting images wants.
    """
    return plt.FuncFormatter(lambda value, _pos=None: '{:g}'.format(value / scale))


def linear_fit(x, y) -> tuple[float, float, float]:
    """Ordinary least squares through one panel: slope, intercept, R squared.

    Unlike `gross_vs_solid`'s calibration line, which is read back out of the table so it cannot
    disagree with the model that table reports, this one has no fitted model behind it -- the
    workbook records measured times, not a regression -- so the line is fitted here and is a
    description of the panel, nothing more. It is off by default for exactly that reason.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    spread = ((y - y.mean()) ** 2).sum()
    return slope, intercept, 1.0 - (residual ** 2).sum() / spread if spread else float('nan')


def _fit_line(ax, frame: pd.DataFrame, x: str, y: str, colour: str,
              linestyle: str = '-') -> None:
    """One least-squares line across the x-range it was fitted on, and no further.

    Clipped to its own data rather than drawn across the axes, which matters when the line is one
    campaign's: extending it over the other's range would assert a cost per image out where that
    campaign flew nothing. It is also what identifies a per-campaign line, since every fit is drawn
    in the same neutral grey -- colour is the markers' to carry -- and solid, `FIT_LINESTYLE`.
    `linestyle` is still a per-series override for a panel that wants one dashed.
    """
    if len(frame) < 3:
        return
    slope, intercept, _ = linear_fit(frame[x], frame[y])
    span = np.array([frame[x].min(), frame[x].max()], dtype=float)
    ax.plot(span, slope * span + intercept, color=colour, linewidth=1.4, linestyle=linestyle,
            zorder=2)


def cost_vs_size(per_truck: pd.DataFrame, path: str, *,
                 panels=COST_PANELS,
                 figsize: tuple[float, float] | None = None,
                 font_size: float = 9,
                 title: str | None = COST_TITLE,
                 panel_titles=None,
                 campaigns: list[str] | None = None,
                 colors: dict | None = None,
                 fit_line: bool = True,
                 legend_loc: str = 'best',
                 dpi: int = 200) -> str:
    """What each half of the method costs per truckload, against the input that drives it.

    Same palette, same marker and the same page furniture as `gross_vs_solid` -- red 2016, blue
    2021, white surface, bold centred panel titles under a suptitle, the key on the first panel
    only -- so the two figures sit together in a manuscript without reading as two designs.

    One marker per truckload, not two: there is no prediction to pair a measurement with here,
    only a measured time, so the hollow marker has nothing to mean and the legend drops the
    marker-style entries with it.

    **The panels deliberately share neither axis**, which is where this departs from
    `_scatter_figure`. There the two panels hold the same quantity under two segmentations, so a
    shared y is the only honest way to compare them by eye. Here they hold different quantities
    entirely -- images against seconds on the left, points against seconds on the right, with the
    left running to some thousands of seconds and the right to about a hundred. Forcing a common
    scale would flatten the right panel into a line along its own axis and compare nothing.

    Splitting by campaign does the same work it does in the volume figures. The two are different
    surveys as well as different vehicles, and on this data they barely overlap in x on the left
    -- 30-66 images in 2016 against 108-154 in 2021.

    That is also why `fit_line` fits each panel the way its own `fit` key says: one line per
    campaign on the left, where the slopes genuinely differ and a pooled line would be fitted to
    the empty gap between the clusters, and one pooled line on the right, where they do not. See
    `FIT_MODES` for the numbers behind both defaults. Every fit is drawn in the same neutral grey
    whichever mode it came from, and a per-campaign line stops at the ends of its own data.
    """
    colors = colors or SCATTER_CAMPAIGN_COLOR
    panels = list(panels)
    if figsize is None:
        figsize = (7.0 * len(panels), 6.0)

    tick_size, label_size, title_size = font_size, font_size + 1, font_size + 3

    data = per_truck.assign(campaign=campaign_labels(per_truck['campaign']))

    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    axes = np.atleast_1d(axes)

    for index, (ax, panel) in enumerate(zip(axes, panels)):
        series = panel_series(panel)
        columns = [panel['x']] + [s['y'] for s in series]
        frame = data.dropna(subset=columns)
        present = [c for c in (campaigns or CAMPAIGN_YEARS) if c in set(frame['campaign'])]
        per_campaign = panel.get('fit', 'campaign') == 'campaign'

        # Every line before every marker, so no fit sits on top of the points it describes --
        # including another series' points, which a per-series interleaving would let it do
        if fit_line:
            for item in series:
                # Every fit in the same neutral grey, pooled or not, so a line never competes with
                # the markers for the reader's colour vocabulary: hue means campaign and only
                # campaign. Which campaign a per-campaign line belongs to is unambiguous anyway --
                # it is clipped to that campaign's own x range, and on this data the two ranges do
                # not overlap. The caption states that the lines are fitted separately.
                if per_campaign:
                    for campaign in present:
                        _fit_line(ax, frame[frame['campaign'] == campaign], panel['x'], item['y'],
                                  MUTED, item['linestyle'])
                else:
                    _fit_line(ax, frame, panel['x'], item['y'], MUTED, item['linestyle'])

        for item in series:
            _campaign_scatter(ax, frame, present, colors, panel['x'], item['y'], font_size,
                              item['marker'])

        _style(ax, _panel_title(panel_titles, index, panel['title']),
               xlabel=panel.get('xlabel', panel['x']),
               ylabel=panel.get('ylabel', series[0]['y']),
               text_color=INK, frame_color=INK, tick_size=tick_size, label_size=label_size,
               title_size=title_size, title_loc='center', surface=WHITE)
        ax.xaxis.set_major_formatter(scaled_ticks(panel.get('xscale', 1.0)))

        # The volume figures key the first panel only, because the second encodes identically.
        # Here they need not: the panels can carry different series and different fit modes, and a
        # key describing the left one would then misdescribe the right. So a panel is keyed when
        # it is the first, or when what it encodes differs from the first.
        encoding = (tuple(item['marker'] for item in series), per_campaign)
        if index == 0:
            first_encoding = encoding
        if index == 0 or encoding != first_encoding:
            # Campaign hues alone: `filled_label=None` drops the volume figures' marker-style
            # entries, which name a distinction this figure does not draw
            handles = _campaign_handles(present, colors, font_size, None, '', False)
            # A single series needs no key: its shape is the only shape on the panel
            if len(series) > 1:
                handles.extend(plt.Line2D([], [], marker=item['marker'], linestyle='none',
                                          markersize=font_size * 0.8, color=MUTED,
                                          markeredgecolor=MARKER_EDGE,
                                          markeredgewidth=MARKER_EDGE_WIDTH,
                                          label=item['label'])
                               for item in series)
            if fit_line:
                handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.4,
                                          label='Least-squares fit'))
            _legend(ax, handles, legend_loc, tick_size)

    if title:
        fig.suptitle(title, color=INK, fontsize=title_size + 1, fontweight='bold')

    return _save(fig, path, dpi=dpi, surface=WHITE)


SEGMENTATIONS = ARMS + ('both',)


def _scatter_figure(spec: dict, per_truck: pd.DataFrame, path: str, *,
                    figsize, font_size, title, xlabel, ylabel, segmentation, panel_titles,
                    campaigns, colors, show_loo, show_residuals, fit_line, legend_loc, dpi) -> str:
    """One panel per segmentation arm, side by side and sharing y when both are asked for.

    Sharing the y axis is what makes `segmentation='both'` worth more than two separate figures:
    the arms are compared by eye, and a shared scale is the only way that comparison is honest.
    The x axes are deliberately *not* shared -- gross volume is the quantity segmentation changes,
    so forcing a common x would hide the very difference the pair is drawn to show.

    On the predicted panel the calibration line is still the reference arm's fit -- the tables
    carry no other, by design -- evaluated at the model's gross volumes. It is that calibration's
    prediction for an automatically segmented cloud, not a fit to the points beneath it, and a
    caption should say so.

    The legend goes on the first panel only: it names campaigns and marker styles, both of which
    are identical in the second, and repeating it would spend a corner of the figure on nothing.
    """
    colors = colors or SCATTER_CAMPAIGN_COLOR
    arms = ARMS if segmentation == 'both' else (segmentation,)
    if figsize is None:
        figsize = (7.0 * len(arms), 6.0)

    tick_size, label_size, title_size = font_size, font_size + 1, font_size + 3

    fig, axes = plt.subplots(1, len(arms), figsize=figsize, sharey=len(arms) > 1)
    axes = np.atleast_1d(axes)

    for index, (ax, arm) in enumerate(zip(axes, arms)):
        data = _campaign_frame(per_truck, arm, ['vg_m3', spec['filled']])
        present = [c for c in (campaigns or CAMPAIGN_YEARS) if c in set(data['campaign'])]

        if spec['zero_line']:
            # Zero is the reference the whole plot is read against, so it is drawn, not implied
            ax.axhline(0, color=MUTED, linewidth=1.2, zorder=2)

        drew_fit = fit_line and spec['allow_fit_line'] and 'vs_hat_m3' in data.columns
        if drew_fit:
            line = data.dropna(subset=['vs_hat_m3']).sort_values('vg_m3')
            ax.plot(line['vg_m3'], line['vs_hat_m3'], color=MUTED, linewidth=1.2, zorder=2)

        _campaign_pairs(ax, data, present, colors, spec['filled'], spec['hollow'],
                        font_size=font_size, show_hollow=show_loo,
                        show_residuals=show_residuals)

        # A single panel keeps `title` on its own axes, as it always has. Two panels put `title`
        # above both as a suptitle and give each its own, so neither is left unnameable.
        panel = title if len(arms) == 1 else _panel_title(panel_titles, index, arm)
        _style(ax, panel, xlabel=xlabel, ylabel=ylabel if index == 0 else None,
               text_color=INK, frame_color=INK, tick_size=tick_size, label_size=label_size,
               title_size=title_size, title_loc='center', surface=WHITE)

        if index == 0:
            handles = _campaign_handles(present, colors, font_size, spec['filled_label'],
                                        spec['hollow_label'], show_loo)
            if drew_fit:
                handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.2,
                                          label='Least-squares fit'))
            _legend(ax, handles, legend_loc, tick_size)

    if len(arms) > 1 and title:
        fig.suptitle(title, color=INK, fontsize=title_size + 1, fontweight='bold')

    return _save(fig, path, dpi=dpi, surface=WHITE)


def _panel_title(panel_titles, index: int, fallback: str) -> str | None:
    """A panel's own title: as given, else what the panel is. Always bold -- `_style` sees to that.

    An empty string drops one panel's title while leaving the other's, which a figure whose
    caption already names the panels will want. `fallback` is a segmentation arm in the volume
    figures, where `ARM_LABEL` renders it, and already-written text elsewhere.
    """
    if panel_titles is not None and index < len(panel_titles):
        return panel_titles[index] or None
    return ARM_LABEL.get(fallback, fallback)


def _campaign_frame(per_truck: pd.DataFrame, arm: str, required: list[str]) -> pd.DataFrame:
    """One arm's rows, with campaign normalised to the string labels the colour maps use."""
    data = per_truck[per_truck['segmentation'] == arm] if 'segmentation' in per_truck else per_truck
    data = data.dropna(subset=[c for c in required if c in data.columns])
    return data.assign(campaign=campaign_labels(data['campaign']))


def _campaign_scatter(ax, data: pd.DataFrame, present: list[str], colors: dict,
                      x: str, y: str, font_size: float, marker: str = 'o') -> None:
    """One filled marker per truckload, campaign by hue -- the mark every scatter here is built on.

    Extracted so the marker spec lives in one place: `_campaign_pairs` draws its filled layer with
    it and `cost_vs_size` draws each of its series with it, which is what stops a dot in one figure
    from quietly ending up a different size or weight from a dot in the next.

    `marker` is what separates two series sharing a panel and a colour scheme. Hue is already
    spoken for by campaign, so shape has to carry the series -- and shape survives greyscale
    printing, which is the other reason not to reach for a third colour here.
    """
    for campaign in present:
        frame = data[data['campaign'] == campaign]
        colour = colors.get(campaign, INK)
        # Diamonds and squares read smaller than a circle of equal area, so they are nudged up to
        # match by eye rather than by number
        size = font_size * (5.6 if marker in ('D', 's') else 4.5)
        ax.scatter(frame[x], frame[y], s=size, color=colour, marker=marker,
                   edgecolors=MARKER_EDGE, linewidths=MARKER_EDGE_WIDTH, zorder=5)


def _campaign_pairs(ax, data: pd.DataFrame, present: list[str], colors: dict,
                    filled: str, hollow: str, *, font_size: float,
                    show_hollow: bool, show_residuals: bool) -> None:
    """The two-marker-per-load scatter both volume figures are built from.

    Shared so the pair keeps one meaning across them: filled is what the calibration saw, hollow
    what it predicted without this load, and the two always sit at the same x.

    The filled marker takes `MARKER_EDGE`'s dark hairline, through `_campaign_scatter`. The hollow
    one deliberately does not: its ring *is* the marker, and that ring is what carries its
    campaign, so darkening it would cost the reading it exists for.
    """
    size = font_size * 4.5

    for campaign in present:
        frame = data[data['campaign'] == campaign]
        colour = colors.get(campaign, INK)

        if show_residuals and show_hollow and hollow in frame.columns:
            for _, row in frame.dropna(subset=[hollow]).iterrows():
                ax.plot([row['vg_m3']] * 2, [row[filled], row[hollow]],
                        color=colour, linewidth=0.8, alpha=0.45, zorder=3)

    _campaign_scatter(ax, data, present, colors, 'vg_m3', filled, font_size)

    for campaign in present:
        frame = data[data['campaign'] == campaign]
        if show_hollow and hollow in frame.columns:
            open_markers = frame.dropna(subset=[hollow])
            ax.scatter(open_markers['vg_m3'], open_markers[hollow], s=size, facecolors='none',
                       edgecolors=colors.get(campaign, INK), linewidths=1.3, zorder=4)


def _campaign_handles(present: list[str], colors: dict, font_size: float,
                      filled_label: str | None, hollow_label: str, show_hollow: bool) -> list:
    """Legend entries: campaign by hue, then what each marker style means.

    A `filled_label` of None drops the marker-style entries, which is what a figure drawing one
    marker per load wants: there is no second style to contrast it with, and a key explaining that
    a dot is a dot spends a corner of the plot saying nothing.
    """
    # The swatches carry the markers' dark edge too, so the key looks like what it names. The
    # hollow entry below is the exception: its ring *is* the marker, and it stays MUTED.
    handles = [plt.Line2D([], [], marker='o', linestyle='none', markersize=font_size * 0.8,
                          color=colors.get(c, INK), markeredgecolor=MARKER_EDGE,
                          markeredgewidth=MARKER_EDGE_WIDTH, label='{} campaign'.format(c))
               for c in present]
    if filled_label is None:
        return handles
    handles.append(plt.Line2D([], [], marker='o', linestyle='none', markersize=font_size * 0.8,
                              color=MUTED, markeredgecolor=MARKER_EDGE,
                              markeredgewidth=MARKER_EDGE_WIDTH, label=filled_label))
    if show_hollow:
        handles.append(plt.Line2D([], [], marker='o', linestyle='none',
                                  markersize=font_size * 0.8, markerfacecolor='none',
                                  markeredgecolor=MUTED, markeredgewidth=1.3,
                                  label=hollow_label))
    return handles


def _legend(ax, handles, loc: str, font_size: float):
    """Place the legend, with two options matplotlib does not spell as locations.

    `'best'` is the default because which corner is free depends on the column plotted: an error
    column puts the boxes low and leaves the top clear, while `adj_r2` puts them high and the
    legend lands on top of them. `'outside'` moves it clear of the axes entirely, for a figure
    too crowded for any corner, and `'none'` drops it for a caption to carry instead.
    """
    if loc == 'none':
        return
    kwargs = dict(handles=handles, frameon=False, fontsize=font_size, labelcolor=INK)
    if loc == 'outside':
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), **kwargs)
    else:
        ax.legend(loc=loc, **kwargs)


def _methods(frame: pd.DataFrame) -> list[str]:
    """Mesh-method campaigns present, in a stable order."""
    from semantictruck.geometry.mesh import METHODS as MESH_BASE
    present = set(frame['mesh_method'])
    return [m for m in MESH_BASE if m in present]


def _setting_order(frame: pd.DataFrame) -> list[str]:
    """BASE first, then the P-numbers in their natural order."""
    ids = sorted(set(frame['setting_id']))
    return ['BASE'] + [s for s in ids if s != 'BASE']


def draw_all(per_load: pd.DataFrame, summary: pd.DataFrame,
             out_dir: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Draw every sweep figure. Returns `(written, failed)` -- see the note below."""
    os.makedirs(out_dir, exist_ok=True)
    written = []

    plots = [('tornado_d_rmse.png', lambda p: tornado(summary, p)),
             ('ratio_cv.png', lambda p: ratio_cv(summary, p)),
             ('rmse_loo_by_method.png', lambda p: box_whiskers(summary, p)),
             ('segmentation_gap.png', lambda p: segmentation_gap(summary, p))]
    for mesh in _methods(summary):
        for arm in ARMS:
            plots.append(('d_vg_pct_{}_{}.png'.format(mesh, arm),
                          lambda p, m=mesh, a=arm: paired_d_vg(per_load, p, a, m)))
        plots.append(('plane_angle_vs_rmse_{}.png'.format(mesh),
                      lambda p, m=mesh: plane_angle(summary, p, m)))

    failed = []
    for name, draw in plots:
        try:
            written.append(draw(os.path.join(out_dir, name)))
        except Exception as exc:                                       # noqa: BLE001
            # Recorded and returned, not merely printed. A figure that cannot be drawn is a result
            # about the data, and the caller exits non-zero on it -- one of these panels once came
            # out blank across a whole campaign because a column had been read back as float, and a
            # printed line scrolled past while the run reported success.
            failed.append((name, '{}: {}'.format(type(exc).__name__, exc)))
            print('  ! {} not drawn: {}: {}'.format(name, type(exc).__name__, exc))

    return written, failed
