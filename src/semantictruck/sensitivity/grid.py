"""The parameter grid, and the settings expanded out of it.

One campaign per reconstruction method -- Poisson, alpha shape, ball pivoting, convex hull -- each
a complete one-at-a-time sweep in its own right. A campaign's BASE holds that method at the middle
of its own grid, and the shared Stage 1 parameters are swept against it. A parameter belonging to a
*different* method is never touched, so nothing in a campaign is confounded by a reconstruction it
does not use.

The point of running the same Stage 1 sweep under four reconstructions is interaction. Sweeping
against Poisson alone cannot tell whether a parameter matters *because* of Poisson or in general.

**The values live in `configs/sensitivity_grid.csv`, not here.** That file is the source of truth
and is re-read on every run; what lives in this module is the mapping from one of its rows to a
field of `Params`, together with the unit conversions its own column headers state -- centimetres
to metres, a percentile to a fraction, total slices to the per-half count.

**`SENSITIVITY_BASE` is not `Params()`.** The package's defaults are the manuscript's published
configuration; the sweep's centre point is a different thing, and predates it. Five fields differ,
and the sweep must keep using its own or its one-at-a-time deltas would be measured against a
centre its grid was never built around.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace

import pandas as pd

from semantictruck.geometry.mesh import ALPHA, BALL_PIVOTING, HULL, POISSON
from semantictruck.pipeline.params import Params

# The centre of the grid in `configs/sensitivity_grid.csv`. See the module docstring: this is
# deliberately not the package default.
SENSITIVITY_BASE = Params(sor_std_ratio=1.5, dbscan_eps=0.06, crop_m=0.50,
                          k_normals=15, q_filter=0.075)

BASE_ID = 'BASE'


@dataclass(frozen=True)
class Setting:
    """A row of the grid resolved to a concrete `Params`, within one reconstruction campaign."""

    setting_id: str
    stage: int
    method: str        # the grid's own 'Method' column, e.g. 'SOR filter' -- not Params.method
    parameter: str
    value: str
    params: Params
    mesh_method: str = POISSON
    skip_reason: str = ''

    @property
    def is_base(self) -> bool:
        return self.setting_id == BASE_ID

    @property
    def skipped(self) -> bool:
        return bool(self.skip_reason)

    @property
    def key(self) -> tuple[str, str]:
        """What identifies a setting, given that each reconstruction has its own BASE."""
        return (self.mesh_method, self.setting_id)


# Where each campaign's BASE sits, relative to `SENSITIVITY_BASE`
MESH_BASE = {
    POISSON: {},                                                    # depth 8, 7.5th percentile
    ALPHA: {'method': ALPHA, 'alpha': 2.0},                         # middle of 1 / 2 / 4 m
    BALL_PIVOTING: {'method': BALL_PIVOTING,
                    'bpa_radii': (0.05, 0.10, 0.20)},               # middle of A / B / C
    HULL: {'method': HULL},                                         # no parameters of its own
}

# Which grid rows belong to which reconstruction. Everything not listed is shared and is swept in
# every campaign.
MESH_OWNED = {POISSON: ('P09', 'P10'), ALPHA: ('P12',),
              BALL_PIVOTING: ('P11',), HULL: ()}

OWNED_BY_ANY = {pid for pids in MESH_OWNED.values() for pid in pids}

# Reconstructions that actually consume oriented normals. Alpha shapes and convex hulls mesh point
# *positions* alone, so the normal parameters are structurally inert there -- they would contribute
# rows whose zero effect is a tautology rather than a measurement. Verified: their volumes came out
# bit-identical to BASE.
NEEDS_NORMALS = (POISSON, BALL_PIVOTING)
NORMAL_SETTINGS = ('P14', 'P15')

# (reconstruction, setting) pairs that cannot be run at all. They are still emitted, so the tables
# stay rectangular and the omission is visible, but they are recorded rather than executed.
UNATTAINABLE = {
    (BALL_PIVOTING, 'P08_none'):
        'ball pivoting on the undecimated cloud passed 300 s on a mid-size load without '
        'finishing, and the SIGALRM timeout cannot interrupt Open3D C++, so running it would '
        'hang the campaign rather than be recorded as a timeout',
}

# Setting id -> (Params field, converter from the grid's token). The converter carries the unit
# change that the grid's own column header states.
_FIELD = {
    'P01': ('sor_neighbours', lambda v: int(float(v))),
    'P02': ('sor_std_ratio', float),
    'P03': ('dbscan_eps', float),
    'P04': ('dbscan_min_points', lambda v: int(float(v))),
    'P05': ('crop_m', lambda v: float(v) / 100),          # the grid is in cm
    # the grid gives the total slice count; `nslices` is per half
    'P06': ('nslices', lambda v: int(float(v)) // 2),
    'P07': ('npoints', lambda v: int(float(v))),
    'P08': ('voxel_size', lambda v: None if v == 'none' else float(v) / 100),   # cm
    'P09': ('poisson_depth', lambda v: int(float(v))),
    'P10': ('q_filter', lambda v: float(v) / 100),        # the grid gives a percentile
    'P14': ('normal_radius', float),                      # metres, on the full-resolution bay
    'P15': ('k_normals', lambda v: int(float(v))),
}

# Rows that replace the reconstruction rather than tuning Stage 1
_STAGE2 = {'P11': BALL_PIVOTING, 'P12': ALPHA, 'P13': HULL}

# P11's radii sets, parsed out of the free-text 'Test Values' cell
_BPA_RE = re.compile(r'([A-Z]):\s*\[([0-9.,\s]+)\]')


def _tokens(cell: str) -> list[str]:
    """The comma-separated test values of a cell, with any trailing gloss dropped.

    A cell may read '0.10, 0.20, 0.30 (2g, 4g, 6g)', where the parenthesised part restates the
    values as multiples of the voxel size and is not itself a value.
    """
    cell = re.sub(r'\([^)]*\)', '', str(cell))
    return [token.strip() for token in cell.split(',') if token.strip()]


def load_settings(csv_path: str, mesh_methods=tuple(MESH_BASE),
                  base: Params = SENSITIVITY_BASE) -> list[Setting]:
    """Expand the grid into the settings to run: one full sweep per reconstruction.

    Within a campaign, a test value equal to that campaign's own BASE resolves to BASE and is not
    emitted again -- which is what keeps each sweep one-at-a-time rather than a grid.
    """
    table = pd.read_csv(csv_path)
    settings = []

    for mesh_method in mesh_methods:
        campaign_base = replace(base, **MESH_BASE[mesh_method])
        settings.append(Setting(BASE_ID, 0, 'All', 'All at baseline', '-',
                                campaign_base, mesh_method))

        for _, row in table.iterrows():
            pid = str(row['Setting ID']).strip()
            if pid == BASE_ID:
                continue
            # A different reconstruction's parameter says nothing about this one
            if pid in OWNED_BY_ANY and pid not in MESH_OWNED[mesh_method]:
                continue
            if pid in NORMAL_SETTINGS and mesh_method not in NEEDS_NORMALS:
                continue

            common = (int(row['Stage']), row['Method'], row['Parameter'])
            for setting_id, value, params in _expand(pid, str(row['Test Values']), campaign_base):
                settings.append(Setting(setting_id, *common, value, params, mesh_method,
                                        UNATTAINABLE.get((mesh_method, setting_id), '')))

    _validate(settings, mesh_methods)
    return settings


def _expand(pid: str, cell: str, base: Params):
    """The settings one grid row contributes, given the campaign's own BASE."""
    if pid == 'P11':
        for letter, radii in _BPA_RE.findall(cell):
            values = tuple(float(r) for r in radii.split(',') if r.strip())
            if values != base.bpa_radii:
                yield ('P11_{}'.format(letter), letter, replace(base, bpa_radii=values))
        return

    if pid == 'P12':
        for token in _tokens(cell):
            if float(token) != base.alpha:
                yield ('P12_{}'.format(token), token, replace(base, alpha=float(token)))
        return

    if pid == 'P13':
        return          # the convex hull has no parameters; it is a campaign, not a setting

    field, convert = _FIELD[pid]
    for token in _tokens(cell):
        value = convert(token)
        if value == getattr(base, field):
            continue    # this is the campaign's BASE run, which is not recomputed
        yield ('{}_{}'.format(pid, token), token, replace(base, **{field: value}))


def _validate(settings: list[Setting], mesh_methods) -> None:
    keys = [s.key for s in settings]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate settings: {}'.format(
            sorted({k for k in keys if keys.count(k) > 1})))

    for mesh_method in mesh_methods:
        first = next(s for s in settings if s.mesh_method == mesh_method)
        if not first.is_base:
            raise ValueError('{}: BASE must come first; deltas are computed against it'.format(
                mesh_method))

    known = {f.name for f in fields(Params)}
    for field, _ in _FIELD.values():
        if field not in known:
            raise ValueError('{} is not a field of Params'.format(field))
