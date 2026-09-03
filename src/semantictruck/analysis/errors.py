"""Reducing a run's per-truckload errors to the figures a results table quotes.

For the error, the percentage error, the squared error, the absolute error and the absolute
percentage error: their **mean, standard deviation, median, minimum and maximum**, plus the RMSE.
This is the manuscript's Table 3, and with `by_campaign` its Table 4.

Every figure is reported twice, as separate rows, because the difference between them is the point:

    in_data   the calibration was fitted on all the truckloads, this one included
    loo       each truckload predicted by a model refitted on the others

An in-data error is optimistic by construction and the gap to leave-one-out is the size of that
optimism. Quoting either alone misleads, in opposite directions.

**Nothing here fits or predicts anything.** The pipeline produced the calibration and the per-truck
errors; this reduces them. So it cannot disagree with the run it describes, and both evaluations
are computed by identical arithmetic from their signed error rather than read from different
columns that might have been produced differently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from semantictruck.analysis.figures import campaign_labels
from semantictruck.arms import ARMS

# Evaluation -> (signed error column, percentage error column), both written per truckload by the
# pipeline. Everything else in the summary is derived from these two here.
EVALUATIONS = {
    'in_data': ('error_m3', 'error_pct'),
    'loo': ('error_loo_m3', 'error_loo_pct'),
}

REQUIRED = ('load_id', 'vs_ref_m3')

# The five quantities summarised, and how each is named in the console view. Note the per-truck
# column `se_m3` carries a squared volume despite its suffix; the summary spells the unit right.
METRICS = {
    'error_m3': 'Error (m³)',
    'error_pct': 'Error (%)',
    'squared_error_m6': 'Squared error (m⁶)',
    'abs_error_m3': 'Absolute error (m³)',
    'abs_error_pct': 'Absolute error (%)',
}

# Reported for each of them, in this order
STATS = ('mean', 'sd', 'median', 'min', 'max')

KEYS = ['segmentation', 'evaluation', 'campaign', 'n_trucks']

# The CSV reads the way the metrics are quoted: all five statistics of one quantity before the next
COLUMNS = KEYS + ['{}_{}'.format(stat, metric) for metric in METRICS for stat in STATS] \
    + ['rmse_m3']


def summarise_errors(frame: pd.DataFrame, error_column: str, pct_column: str) -> dict:
    """The aggregate error figures of one group of truckloads.

    Standard deviations are the sample ones (`ddof=1`), which is how a spread over 29 loads is
    normally quoted. **`rmse_m3` is not `sd_error_m3`**: the RMSE is taken about zero and so carries
    the bias, while the SD is taken about the mean error and removes it. Reporting both is what
    separates a calibration that is off-centre from one that is merely noisy.
    """
    error = frame[error_column].astype(float)
    pct = (frame[pct_column].astype(float) if pct_column in frame.columns
           else pd.Series(dtype=float))

    values = {
        'error_m3': error,
        'error_pct': pct,
        'squared_error_m6': error ** 2,
        'abs_error_m3': error.abs(),
        'abs_error_pct': pct.abs(),
    }

    summary = {'n_trucks': int(error.notna().sum())}
    for metric, series in values.items():
        summary['mean_' + metric] = series.mean()
        summary['sd_' + metric] = series.std()
        # Median beside the mean because one bad load moves one and not the other, which separates
        # a calibration that is generally off from one with an outlier in it. Min and max are the
        # actual worst cases; on the signed error their two ends also show whether the misses are
        # one-sided.
        summary['median_' + metric] = series.median()
        summary['min_' + metric] = series.min()
        summary['max_' + metric] = series.max()

    summary['rmse_m3'] = float(np.sqrt((error ** 2).mean())) if error.notna().any() else np.nan
    return summary


def build_summary(table: pd.DataFrame, arms=ARMS, by_campaign: bool = False) -> pd.DataFrame:
    """One row per (arm, evaluation), or per (arm, evaluation, campaign) when asked."""
    rows = []
    for arm in arms:
        arm_frame = (table[table['segmentation'] == arm] if 'segmentation' in table.columns
                     else table)
        if arm_frame.empty:
            continue

        groups = [('all', arm_frame)]
        if by_campaign and 'campaign' in arm_frame.columns:
            labelled = arm_frame.assign(campaign=campaign_labels(arm_frame['campaign']))
            groups += [(name, frame) for name, frame in labelled.groupby('campaign')]

        for campaign, frame in groups:
            for evaluation, (error_column, pct_column) in EVALUATIONS.items():
                if error_column not in frame.columns:
                    continue
                scored = frame.dropna(subset=[error_column])
                if scored.empty:
                    continue
                rows.append(dict(segmentation=arm, evaluation=evaluation, campaign=campaign,
                                 **summarise_errors(scored, error_column, pct_column)))

    summary = pd.DataFrame(rows)
    return summary[[c for c in COLUMNS if c in summary.columns]] if len(summary) else summary


def long_view(summary: pd.DataFrame) -> pd.DataFrame:
    """The same figures with one row per metric, for reading in a terminal.

    The CSV keeps one row per (arm, evaluation) so a spreadsheet can filter it, but five statistics
    of five quantities is twenty-five columns and does not read. Nothing is recomputed here.
    """
    rows = []
    for _, row in summary.iterrows():
        for metric, label in METRICS.items():
            rows.append({
                'segmentation': row['segmentation'],
                'evaluation': row['evaluation'],
                'campaign': row['campaign'],
                'metric': label,
                **{stat: row['{}_{}'.format(stat, metric)] for stat in STATS},
            })
    return pd.DataFrame(rows)
