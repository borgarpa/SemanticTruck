"""Calibrating solid volume from gross volume, and scoring the result honestly.

The model is the manuscript's Equation 2, an ordinary least-squares line

    Vs = a * Vg + b

fitted on the **reference** arm and used to score both arms, so the predicted arm's extra error is
attributable to its segmentation rather than to a refit that would absorb it.

**Every error is reported twice**, and quoting either alone misleads:

    in-data      fitted and evaluated on the same truckloads. Optimistic by construction.
    leave-one-out   each load predicted by a model refitted on the other 28 -- what the method
                    would give on a truck it had never seen.

`loo_fits` reports the held-out calibration per load rather than only its pooled error, because
that is what makes the number auditable: a slope that moves when one truck is dropped says the
calibration leans on that truck, and a pooled RMSE cannot show it.

Reporting `a` and `b` for every setting is the point of the sensitivity sweep, not decoration: a
parameter that merely rescales the gross volume shows up as `a` moving inversely to `mean_vg_m3`
while `mean_vs_hat_m3` holds steady. That is visible here rather than asserted.

Nothing in this module imports Open3D. It reads the tables the pipeline writes, so it runs on a
machine with neither the clouds nor a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from semantictruck.analysis.reference import campaign_of, config_of
from semantictruck.arms import PREDICTED, REFERENCE
from semantictruck.geometry.diagnostics import plane_angle_deg
from semantictruck.pipeline.status import STATUS_OK


def per_bay_to_per_load(per_bay: pd.DataFrame) -> pd.DataFrame:
    """Collapse the raw per-bay table to one row per (setting, load, arm).

    Bay-level quantities aggregate the way the underlying question does: volumes and triangle
    counts add, boundary length adds (it is the total open edge of the reconstruction), a load is
    watertight only if every bay is, and the plane angle is the worst bay's -- a load is only as
    level as the bay that levelled worst.
    """
    keys = ['mesh_method', 'setting_id', 'segmentation', 'load_id']
    grouped = per_bay.groupby(keys, as_index=False, sort=False)

    spec = dict(
        stage=('stage', 'first'), method=('method', 'first'),
        parameter=('parameter', 'first'), value=('value', 'first'),
        n_bays=('n_bays', 'first'), status=('status', 'first'), notes=('notes', 'first'),
        runtime_s=('runtime_s', 'first'), stage1_runtime_s=('stage1_runtime_s', 'first'),
        points_in=('points_in', 'first'), points_after_sor=('points_after_sor', 'first'),
        frac_removed=('frac_removed', 'first'),
        frac_removed_labelled_wood=('frac_removed_labelled_wood', 'first'),
        points_reassigned=('points_reassigned', 'first'),
        points_discarded=('points_discarded', 'first'),
        cargo_agreement=('cargo_agreement', 'first'),
        normal_radius_m=('normal_radius_m', 'first'),
        n_collapsed_bays=('n_collapsed_bays', 'first'),
        vg_m3=('vg_m3_bay', 'sum'),
        # Eq. 1 unconditionally, alongside the volume actually used -- they differ only where a
        # bay meshed watertight, which is where `get_volume` takes Open3D's closed-mesh branch
        eq1_m3=('eq1_m3', 'sum'),
        volume_closed=('volume_closed', 'all'),
        hull_vg_m3=('hull_vg_m3', 'sum'),
        min_vg_hull_ratio=('vg_hull_ratio', 'min'),
        points_after_voxel=('points_after_voxel', 'sum'),
        n_triangles=('n_triangles', 'sum'),
        boundary_edge_len_m=('boundary_edge_len_m', 'sum'),
        n_boundary_loops=('n_boundary_loops', 'sum'),
        is_watertight=('is_watertight', 'all'),
        min_points_per_slice=('min_points_per_slice', 'min'),
        slices_underfilled=('slices_underfilled', 'sum'),
        crop_span_m=('crop_span_m', 'mean'), base_span_m=('base_span_m', 'mean'),
        plane_angle_deg=('plane_angle_deg', 'max'),
    )

    # Load-level columns that are carried, not aggregated. Optional because not every producer
    # writes them: the per-step timings are absent from a per-bay table written before they
    # existed, and `vs_hat_pub_m3` -- the published Eq. 2 applied to this run, so a sweep
    # checkpoint can be read before the campaign is complete enough to refit -- is written by
    # `runner._tag` alone, not by a single-configuration run where it means nothing.
    for column in ('vs_hat_pub_m3', 't_read_s', 't_preprocess_s', 't_bays_s', 't_mesh_s'):
        if column in per_bay.columns:
            spec[column] = (column, 'first')

    per_load = grouped.agg(**spec)

    # A load that failed anywhere has no usable gross volume, however many bays did mesh
    per_load.loc[per_load['status'] != STATUS_OK, 'vg_m3'] = np.nan

    return per_load


def add_plane_angles(per_bay: pd.DataFrame) -> pd.DataFrame:
    """Angle between each run's fitted base-plane normal and the same bay's normal at BASE.

    Paired within an arm and by bay index. `shift_cluster_labels` orders bays by their low-Y
    bound, so the index is stable across settings even when the clustering moves points.
    """
    per_bay = per_bay.copy()
    planes = per_bay[['plane_a', 'plane_b', 'plane_c']].to_numpy(dtype=float)
    keys = list(zip(per_bay['mesh_method'], per_bay['segmentation'],
                    per_bay['load_id'], per_bay['bay_id']))
    is_base = (per_bay['setting_id'] == 'BASE').to_numpy()

    reference = {key: plane for key, plane, base in zip(keys, planes, is_base)
                 if base and not np.isnan(plane).any()}

    angles = np.full(len(per_bay), np.nan)
    for i, (key, plane) in enumerate(zip(keys, planes)):
        against = reference.get(key)
        if against is None or np.isnan(plane).any():
            continue
        angles[i] = plane_angle_deg(plane, against)

    per_bay['plane_angle_deg'] = angles
    return per_bay


def fit_setting(frame: pd.DataFrame) -> dict:
    """OLS `vs_ref ~ a * vg + b` on the loads of one setting that produced a volume."""
    usable = frame.dropna(subset=['vg_m3', 'vs_ref_m3'])
    if len(usable) < 3:
        return {'n_fit': len(usable), 'a': np.nan, 'b': np.nan,
                'adj_r2': np.nan, 'breusch_pagan_p': np.nan}

    exog = sm.add_constant(usable[['vg_m3']].to_numpy(dtype=float), has_constant='add')
    model = sm.OLS(usable['vs_ref_m3'].to_numpy(dtype=float), exog).fit()
    intercept, slope = model.params

    return {
        'n_fit': len(usable),
        'a': float(slope),
        'b': float(intercept),
        'adj_r2': float(model.rsquared_adj),
        'breusch_pagan_p': float(het_breuschpagan(model.resid, model.model.exog)[1]),
    }


LOO_COLUMNS = ['a_loo', 'b_loo', 'adj_r2_loo', 'n_fit_loo']


def loo_fits(calibration: pd.DataFrame) -> pd.DataFrame:
    """The calibration that predicts each truckload: refit on the others, one row per load.

    Indexed by the load *held out*, with the slope, intercept and adjusted R2 of the model fitted
    on the remaining 28. Reported per load rather than only summarised because it is what makes
    the leave-one-out error auditable: a slope that moves when one truck is dropped says the
    calibration leans on that truck, which a pooled `rmse_loo_m3` cannot show.

    The fit always comes from the reference arm, so a load's coefficients are the same whichever
    arm's row they end up on -- only the gross volume they are applied to differs.
    """
    fit = calibration.dropna(subset=['vg_m3', 'vs_ref_m3'])
    loads = fit['load_id'].to_numpy()
    x = fit['vg_m3'].to_numpy(dtype=float)
    y = fit['vs_ref_m3'].to_numpy(dtype=float)

    rows = []
    for i, load in enumerate(loads):
        keep = np.ones(len(loads), dtype=bool)
        keep[i] = False
        n = int(keep.sum())
        if n < 3:
            continue

        design = np.column_stack([x[keep], np.ones(n)])
        (slope, intercept), *_ = np.linalg.lstsq(design, y[keep], rcond=None)

        # Of the fold's own 28 loads: how well the model that predicts the held-out load fits the
        # loads it was built from. One predictor, so the adjustment is (n - 1) / (n - 2).
        residual = y[keep] - (slope * x[keep] + intercept)
        ss_tot = float(((y[keep] - y[keep].mean()) ** 2).sum())
        r2 = 1 - float((residual ** 2).sum()) / ss_tot if ss_tot else np.nan
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if ss_tot else np.nan

        rows.append({'load_id': load, 'a_loo': float(slope), 'b_loo': float(intercept),
                     'adj_r2_loo': adj_r2, 'n_fit_loo': n})

    return pd.DataFrame(rows, columns=['load_id'] + LOO_COLUMNS).set_index('load_id')


def loo_predictions(calibration: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Leave-one-load-out predicted solid volume, indexed by load.

    For each truckload the calibration is refitted on the *other* 28 and then applied to the one
    held out, so no load contributes to the model that predicts it. `calibration` supplies the fit
    -- always the reference arm -- and `target` supplies the gross volumes being predicted, which
    are the reference arm's own on a reference row and the model's on a predicted row.

    This is what makes the reported error honest. `rmse_m3` is fitted and evaluated on the same
    29 loads and is optimistic by construction; `rmse_loo_m3` is not.
    """
    fits = loo_fits(calibration)
    predicted = target.set_index('load_id')['vg_m3']

    return pd.Series({load: row.a_loo * predicted.get(load, np.nan) + row.b_loo
                      for load, row in fits.iterrows()}, dtype=float)


def loo_metrics(frame: pd.DataFrame, calibration: pd.DataFrame) -> dict:
    """Out-of-sample counterparts of the in-sample error metrics, for one (setting, arm)."""
    predicted = loo_predictions(calibration, frame)
    if predicted.empty:
        return {'n_loo': 0, 'rmse_loo_m3': np.nan, 'mae_loo_m3': np.nan,
                'mape_loo_pct': np.nan, 'bias_loo_m3': np.nan}

    reference = frame.set_index('load_id')['vs_ref_m3'].reindex(predicted.index)
    error = predicted - reference
    return {
        'n_loo': int(error.notna().sum()),
        'rmse_loo_m3': float(np.sqrt((error ** 2).mean())),
        'mae_loo_m3': float(error.abs().mean()),
        'mape_loo_pct': float((100 * error / reference).abs().mean()),
        'bias_loo_m3': float(error.mean()),
    }


def selection_loo(per_load: pd.DataFrame) -> pd.DataFrame:
    """Honest error for the *procedure* of choosing the best-performing setting.

    Picking the lowest-RMSE setting out of ~100 and quoting its in-sample error double-counts the
    same 29 loads: they choose the setting and then score it. Here the choice is made without the
    held-out load -- for each truckload, every setting is fitted and scored on the other 28, the
    best is taken, and only then is the held-out load predicted. The spread of `chosen` across
    folds says how stable that choice is; a different winner every fold means the argmax is noise.
    """
    calibration_all = per_load[per_load['segmentation'] == REFERENCE]
    loads = sorted(calibration_all.dropna(subset=['vg_m3', 'vs_ref_m3'])['load_id'].unique())
    keys = sorted({(m, s) for m, s in
                   zip(calibration_all['mesh_method'], calibration_all['setting_id'])})

    rows = []
    for arm, arm_frame in per_load.groupby('segmentation'):
        errors, chosen = {}, []
        for held_out in loads:
            best_key, best_rmse, best_fit = None, np.inf, None
            for key in keys:
                fit_frame = calibration_all[(calibration_all['mesh_method'] == key[0])
                                            & (calibration_all['setting_id'] == key[1])
                                            & (calibration_all['load_id'] != held_out)]
                fit = fit_setting(fit_frame)
                if np.isnan(fit['a']):
                    continue
                residual = (fit['a'] * fit_frame['vg_m3'] + fit['b']) - fit_frame['vs_ref_m3']
                rmse = float(np.sqrt((residual ** 2).mean()))
                if rmse < best_rmse:
                    best_key, best_rmse, best_fit = key, rmse, fit

            if best_key is None:
                continue
            row = arm_frame[(arm_frame['mesh_method'] == best_key[0])
                            & (arm_frame['setting_id'] == best_key[1])
                            & (arm_frame['load_id'] == held_out)]
            if row.empty or pd.isna(row['vg_m3'].iloc[0]):
                continue
            predicted = best_fit['a'] * row['vg_m3'].iloc[0] + best_fit['b']
            errors[held_out] = predicted - row['vs_ref_m3'].iloc[0]
            chosen.append('{}/{}'.format(*best_key))

        error = pd.Series(errors, dtype=float)
        counts = pd.Series(chosen).value_counts() if chosen else pd.Series(dtype=int)
        rows.append({
            'segmentation': arm,
            'n_folds': len(error),
            'rmse_selection_loo_m3': float(np.sqrt((error ** 2).mean())) if len(error) else np.nan,
            'mae_selection_loo_m3': float(error.abs().mean()) if len(error) else np.nan,
            'bias_selection_loo_m3': float(error.mean()) if len(error) else np.nan,
            'n_distinct_winners': int(counts.size),
            'modal_winner': counts.index[0] if counts.size else '',
            'modal_winner_folds': int(counts.iloc[0]) if counts.size else 0,
        })

    return pd.DataFrame(rows)


def apply_errors(frame: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """Predicted solid volume and the error columns, given one setting's calibration."""
    frame = frame.copy()
    frame['vs_hat_m3'] = a * frame['vg_m3'] + b
    frame['error_m3'] = frame['vs_hat_m3'] - frame['vs_ref_m3']
    frame['error_pct'] = 100 * frame['error_m3'] / frame['vs_ref_m3']
    frame['se_m3'] = frame['error_m3'] ** 2
    frame['ae_m3'] = frame['error_m3'].abs()
    frame['ape_pct'] = frame['error_pct'].abs()
    frame['ratio'] = frame['vs_ref_m3'] / frame['vg_m3']
    return frame


def build_per_load(per_bay: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """The primary table: one row per (setting, truckload, arm), fully scored."""
    per_load = per_bay_to_per_load(add_plane_angles(per_bay))
    # `vs_ref_m3` is already on the rows from the checkpoint; the reference table is the
    # authority, and also brings campaign and configuration
    per_load = per_load.drop(columns=['vs_ref_m3'], errors='ignore').merge(
        reference, on='load_id', how='left')

    # Campaign and configuration are properties of the load name, not of the reference table, so
    # they are filled in here rather than left blank for a load the reference has no row for --
    # `Truck_6_2016` is exactly that load. A NaN would also promote the column to float64, and a
    # CSV round trip then reads the year back as `2016.0`, which matches no string key anywhere.
    per_load['campaign'] = per_load['load_id'].map(campaign_of)
    per_load['config'] = per_load['load_id'].map(config_of)

    # Deltas are within-arm: a setting is compared against BASE as seen by the same segmentation
    base = per_load[per_load['setting_id'] == 'BASE'].set_index(
        ['mesh_method', 'segmentation', 'load_id'])
    base_vg = base['vg_m3']
    keyed = pd.MultiIndex.from_frame(per_load[['mesh_method', 'segmentation', 'load_id']])
    per_load['d_vg_m3'] = per_load['vg_m3'].to_numpy() - base_vg.reindex(keyed).to_numpy()
    per_load['d_vg_pct'] = 100 * per_load['d_vg_m3'] / base_vg.reindex(keyed).to_numpy()

    # And the cross-arm difference at the same setting: what the segmentation itself costs
    reference_vg = per_load[per_load['segmentation'] == REFERENCE].set_index(
        ['mesh_method', 'setting_id', 'load_id'])
    paired = reference_vg['vg_m3'].reindex(
        pd.MultiIndex.from_frame(per_load[['mesh_method', 'setting_id', 'load_id']])).to_numpy()
    per_load['d_vg_seg_pct'] = np.where(
        per_load['segmentation'] == PREDICTED, 100 * (per_load['vg_m3'] - paired) / paired, np.nan)

    # The reference arm's fit calibrates both arms
    fits = {}
    for key, frame in per_load[per_load['segmentation'] == REFERENCE].groupby(
            ['mesh_method', 'setting_id']):
        fits[key] = fit_setting(frame)

    scored = []
    for (mesh, setting_id, arm), frame in per_load.groupby(
            ['mesh_method', 'setting_id', 'segmentation'], sort=False):
        fit = fits.get((mesh, setting_id), {'a': np.nan, 'b': np.nan})
        scored.append(apply_errors(frame, fit['a'], fit['b']))

    return pd.concat(scored, ignore_index=True).sort_values(
        ['mesh_method', 'setting_id', 'segmentation', 'load_id'], ignore_index=True)


def build_summary(per_load: pd.DataFrame) -> pd.DataFrame:
    """One row per (setting, arm), derived entirely from the per-load table."""
    fits = {key: fit_setting(frame)
            for key, frame in per_load[per_load['segmentation'] == REFERENCE].groupby(
                ['mesh_method', 'setting_id'])}

    calibration_by_key = {key: frame for key, frame in
                     per_load[per_load['segmentation'] == REFERENCE].groupby(
                         ['mesh_method', 'setting_id'])}

    rows = []
    for (mesh, setting_id, arm), frame in per_load.groupby(
            ['mesh_method', 'setting_id', 'segmentation'], sort=False):
        fit = fits.get((mesh, setting_id), {})
        scored = frame.dropna(subset=['vg_m3', 'vs_ref_m3'])
        # Same calibration source as the in-sample fit -- the reference arm -- but refitted
        # without the load being predicted
        calibration = calibration_by_key.get((mesh, setting_id), frame)
        loo = loo_metrics(frame, calibration)

        rows.append({
            'mesh_method': mesh,
            'setting_id': setting_id,
            'segmentation': arm,
            'fit_source': REFERENCE,
            'stage': frame['stage'].iloc[0],
            'method': frame['method'].iloc[0],
            'parameter': frame['parameter'].iloc[0],
            'value': frame['value'].iloc[0],
            'n_fit': fit.get('n_fit', np.nan),
            'n_scored': len(scored),
            'a': fit.get('a', np.nan),
            'b': fit.get('b', np.nan),
            'adj_r2': fit.get('adj_r2', np.nan),
            'breusch_pagan_p': fit.get('breusch_pagan_p', np.nan),
            # In-sample: fitted and evaluated on the same loads, so optimistic by construction
            'rmse_m3': float(np.sqrt(scored['se_m3'].mean())) if len(scored) else np.nan,
            # Leave-one-out: the honest counterpart, reported beside it everywhere
            'rmse_loo_m3': loo['rmse_loo_m3'],
            'mae_loo_m3': loo['mae_loo_m3'],
            'mape_loo_pct': loo['mape_loo_pct'],
            'bias_loo_m3': loo['bias_loo_m3'],
            'n_loo': loo['n_loo'],
            'bias_m3': scored['error_m3'].mean(),
            'bias_pct': scored['error_pct'].mean(),
            'mae_m3': scored['ae_m3'].mean(),
            'mape_pct': scored['ape_pct'].mean(),
            'median_ae_m3': scored['ae_m3'].median(),
            'median_ape_pct': scored['ape_pct'].median(),
            'mean_vg_m3': scored['vg_m3'].mean(),
            'sd_vg_m3': scored['vg_m3'].std(),
            'mean_vs_hat_m3': scored['vs_hat_m3'].mean(),
            'mean_vs_ref_m3': scored['vs_ref_m3'].mean(),
            'mean_d_vg_m3': scored['d_vg_m3'].mean(),
            'sd_d_vg_m3': scored['d_vg_m3'].std(),
            'ratio_mean': scored['ratio'].mean(),
            'ratio_sd': scored['ratio'].std(),
            'ratio_cv': scored['ratio'].std() / scored['ratio'].mean() if len(scored) else np.nan,
            'mean_runtime_s': frame['runtime_s'].mean(),
            'total_runtime_s': frame['runtime_s'].sum(),
            'n_failed': int((frame['status'] != STATUS_OK).sum()),
            'n_loads_collapsed': int((frame['n_collapsed_bays'] > 0).sum()),
            'mean_vg_hull_ratio': scored['vg_m3'].sum() / scored['hull_vg_m3'].sum()
            if scored['hull_vg_m3'].sum() else np.nan,
            'mean_plane_angle_deg': frame['plane_angle_deg'].mean(),
            'max_plane_angle_deg': frame['plane_angle_deg'].max(),
            'mean_boundary_edge_len_m': scored['boundary_edge_len_m'].mean(),
            'mean_d_vg_seg_pct': frame['d_vg_seg_pct'].mean(),
        })

    summary = pd.DataFrame(rows)

    # Deltas against BASE, within arm
    base = summary[summary['setting_id'] == 'BASE'].set_index(['mesh_method', 'segmentation'])
    keyed = pd.MultiIndex.from_frame(summary[['mesh_method', 'segmentation']])
    for column, delta in (('rmse_m3', 'd_rmse_m3'), ('mape_pct', 'd_mape_pct'),
                          ('rmse_loo_m3', 'd_rmse_loo_m3'), ('mape_loo_pct', 'd_mape_loo_pct')):
        summary[delta] = (summary[column].to_numpy()
                          - base[column].reindex(keyed).to_numpy())

    return summary.sort_values(['mesh_method', 'setting_id', 'segmentation'],
                               ignore_index=True)


def check_invariants(summary: pd.DataFrame, expected_n: int = 29) -> list[str]:
    """Checks that must hold across a campaign. Returns the problems found, empty if none."""
    problems = []

    references = summary['mean_vs_ref_m3'].dropna().round(6).unique()
    if len(references) > 1:
        problems.append(
            'mean_vs_ref_m3 is not invariant across settings ({} distinct values: {}) -- a load '
            'was silently dropped'.format(len(references), references))

    short = summary[summary['n_fit'].notna() & (summary['n_fit'] != expected_n)]
    for row in short.itertuples():
        problems.append('{}/{} [{}]: n_fit = {}, expected {}'.format(
            row.mesh_method, row.setting_id, row.segmentation, row.n_fit, expected_n))

    failed = summary[summary['n_failed'] > 0]
    for row in failed.itertuples():
        problems.append('{}/{} [{}]: {} load(s) failed'.format(
            row.mesh_method, row.setting_id, row.segmentation, row.n_failed))

    collapsed = summary[summary.get('n_loads_collapsed', 0) > 0]
    for row in collapsed.itertuples():
        problems.append(
            '{}/{} [{}]: {} load(s) have a bay whose Eq. 1 integral cancelled '
            '(vg/hull < {:.2f}) -- the volume is recorded but does not measure the '
            'stack'.format(row.mesh_method, row.setting_id, row.segmentation,
                           row.n_loads_collapsed, 0.25))

    return problems
