"""The measured solid volume that calibrates the model, and what each truckload is.

The reference is manual mensuration, not another model: every log on the truck was measured for
mid-diameter and length and its volume computed by Huber's formula (manuscript §2.1). It is
therefore invariant across every configuration the pipeline is run at, which is the check that no
load has been silently dropped -- `mean_vs_ref_m3` identical in every summary row.

"""

from __future__ import annotations

import pandas as pd

# Re-exported: these live in a module with no imports, because the training stage needs them
# in an environment that has no pandas.
from semantictruck.campaigns import CONFIG, campaign_of, config_of  # noqa: F401

METHOD_COLUMN = 'Method'
DEFAULT_METHOD = 'Manual'

REQUIRED_COLUMNS = ('File', 'Solid_vol')


def load_reference(path: str, method: str | None = DEFAULT_METHOD) -> pd.DataFrame:
    """Per-load reference solid volume: one row per truckload with a valid measurement.

    Bay rows are summed to the load, because the regression is per truckload. A load whose
    `Solid_vol` is blank is dropped rather than treated as zero -- one semitrailer's log
    measurements were faulty, which is how 30 clouds yield 29 calibration points.
    """
    table = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit('{}: missing column(s) {}; a reference table needs at least {}'.format(
            path, ', '.join(missing), ' and '.join(REQUIRED_COLUMNS)))

    if method and METHOD_COLUMN in table.columns:
        table = table[table[METHOD_COLUMN] == method]
    table = table.dropna(subset=['Solid_vol'])

    aggregation = {'vs_ref_m3': ('Solid_vol', 'sum')}
    if 'Bay_ID' in table.columns:
        aggregation['n_bays_ref'] = ('Bay_ID', 'size')

    per_load = table.groupby('File', as_index=False).agg(**aggregation)
    per_load['campaign'] = per_load['File'].map(campaign_of)
    per_load['config'] = per_load['File'].map(config_of)

    return per_load.rename(columns={'File': 'load_id'})


def load_reference_per_bay(path: str, method: str | None = DEFAULT_METHOD) -> pd.DataFrame:
    """The same table left per bay, so a bay can be shown against the logs measured on it.

    Bay ids line up with the pipeline's because `Cargo.shift_cluster_labels` orders bays by their
    low-Y bound, which is the order the measurements were recorded in.
    """
    table = pd.read_csv(path)
    if method and METHOD_COLUMN in table.columns:
        table = table[table[METHOD_COLUMN] == method]
    table = table.dropna(subset=['Solid_vol'])

    if 'Bay_ID' not in table.columns:
        return pd.DataFrame(columns=['load_id', 'bay_id', 'vs_ref_m3_bay'])

    return table[['File', 'Bay_ID', 'Solid_vol']].rename(columns={
        'File': 'load_id', 'Bay_ID': 'bay_id', 'Solid_vol': 'vs_ref_m3_bay'})
