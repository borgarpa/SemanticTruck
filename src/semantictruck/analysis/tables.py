"""The manuscript's result tables, built from a finished run's CSVs.

    descriptive_table    Table 1 -- measured solid volume, reconstructed gross volume and their
                         ratio, per acquisition campaign
    error_table          Table 3 -- leave-one-out prediction error, per segmentation arm
    campaign_table       Table 4 -- the same error split by campaign

All three read tables the pipeline wrote and reshape them. **Nothing is recomputed**, so a table
cannot disagree with the run it describes -- which is the failure the predecessor of this module
was prone to, having recomputed its own statistics from a different file.

The layout is the manuscript's: statistics across the columns, one row per quantity, grouped by
whatever the table splits on. That is a reading layout, not a storage one; the CSVs the tables are
built from stay one row per group with the statistics as columns, which is what a spreadsheet or a
`groupby` wants.

**Manuscript Table 2 is not here.** It reports the segmentation model's per-fold accuracy and IoU,
which is an output of the training campaign and not of this pipeline -- the model is out of scope,
as the README says. `training.curves` plots the same campaign's per-epoch metrics if you have its
CSV.
"""

from __future__ import annotations

import pandas as pd

from semantictruck.analysis.figures import campaign_labels
from semantictruck.arms import ARM_LABEL, REFERENCE

# Statistics across the columns, in the manuscript's order
STATS = {'min': 'Minimum', 'max': 'Maximum', 'sd': 'SD', 'median': 'Median', 'mean': 'Mean'}

# The five error quantities, and how the manuscript names them. Note `squared_error_m6` really is
# a squared volume: cubic metres squared.
ERROR_METRICS = {
    'error_m3': 'Error (m³)',
    'error_pct': 'Error (%)',
    'squared_error_m6': 'SE (m⁶)',
    'abs_error_m3': 'AE (m³)',
    'abs_error_pct': 'APE (%)',
}

# Table 1's quantities, in its column order, and the aggregation applied to each
DESCRIPTIVE_METRICS = {
    'vs_ref_m3': 'Net vol. (m³)',
    'vg_m3': 'Gross vol. (m³ st)',
    'ratio': 'Ratio',
}
DESCRIPTIVE_STATS = {'min': 'Minimum', 'max': 'Maximum', 'mean': 'Mean',
                     'median': 'Median', 'std': 'SD'}


def descriptive_table(per_truck: pd.DataFrame, arm: str = REFERENCE) -> pd.DataFrame:
    """Manuscript Table 1: what was measured and reconstructed, per campaign.

    The ratio is solid over gross -- the stacking coefficient. Loads with no measured solid volume
    carry no ratio and are dropped, so the count is the calibration set rather than every load run.
    """
    frame = per_truck
    if 'segmentation' in frame.columns:
        frame = frame[frame['segmentation'] == arm]
    frame = frame.dropna(subset=['ratio']) if 'ratio' in frame.columns else frame
    frame = frame.assign(campaign=campaign_labels(frame['campaign']))

    campaigns = sorted(frame['campaign'].dropna().unique())
    table = (frame.groupby('campaign')[list(DESCRIPTIVE_METRICS)]
             .agg(list(DESCRIPTIVE_STATS))
             .stack(level=0, future_stack=True)
             .reindex(index=pd.MultiIndex.from_product([campaigns, list(DESCRIPTIVE_METRICS)]),
                      columns=list(DESCRIPTIVE_STATS))
             .transpose()
             .rename(index=DESCRIPTIVE_STATS)
             .rename(columns=DESCRIPTIVE_METRICS, level=1))
    table.columns.names = ['Campaign', 'Metric']
    return table


def _error_block(summary: pd.DataFrame, group_column: str, group_label: str,
                 evaluation: str = 'loo') -> pd.DataFrame:
    """Reshape the error summary into the manuscript's rows-are-metrics layout."""
    frame = summary[summary['evaluation'] == evaluation].set_index(group_column)
    groups = [g for g in frame.index.unique()]

    index = pd.MultiIndex.from_product([groups, list(ERROR_METRICS.values())],
                                       names=[group_label, 'Metric'])
    values = [[frame.at[group, '{}_{}'.format(stat, metric)] for stat in STATS]
              for group in groups for metric in ERROR_METRICS]
    return pd.DataFrame(values, index=index, columns=list(STATS.values()))


def error_table(summary: pd.DataFrame, evaluation: str = 'loo') -> pd.DataFrame:
    """Manuscript Table 3: leave-one-out prediction error, one block per segmentation arm."""
    pooled = summary[summary['campaign'] == 'all'] if 'campaign' in summary.columns else summary
    table = _error_block(pooled, 'segmentation', 'Segmentation', evaluation)
    return table.rename(index=ARM_LABEL, level='Segmentation')


def campaign_table(summary: pd.DataFrame, arm: str = REFERENCE,
                   evaluation: str = 'loo') -> pd.DataFrame:
    """Manuscript Table 4: the same error split by acquisition campaign, for one arm.

    Needs an error summary built `by_campaign`; a pooled one has only the `all` row and this
    returns nothing rather than silently reporting the pooled figures twice.
    """
    frame = summary[(summary['segmentation'] == arm) & (summary['campaign'] != 'all')]
    if frame.empty:
        return pd.DataFrame()
    return _error_block(frame, 'campaign', 'Campaign', evaluation)
