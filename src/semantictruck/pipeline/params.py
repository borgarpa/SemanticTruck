"""Every parameter the pipeline takes, in one place.

`Params` is the whole configuration of a run. Nothing that changes a result is left at a call-site
constant or a method default: if it moves a number, it is a field here, and `config_used.json`
records what it was.

**The defaults are the manuscript's published configuration** (§2.4 and §2.5), so the pipeline run
with no configuration file reproduces the paper. `configs/` carries the same values as
`manuscript.json` for the record, alongside the variants that reproduce earlier states of the work.

Two values that the manuscript states in words but that earlier code buried as method defaults are
fields here for that reason: `downsample_every_k` ("retaining one in three points") and
`kmeans_max_iter` ("k-means clustering ... for 1000 iterations"). They were never varied, but a
configuration that does not mention them is not a description of the run.

Units are SI throughout -- metres, not centimetres -- which is not what the manuscript's prose uses.
Each field says which.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, fields, replace

from semantictruck.geometry.mesh import METHODS, POISSON


@dataclass(frozen=True)
class Params:
    """One point in parameter space. Defaults are the manuscript's configuration."""

    # -- Denoising (§2.4). Statistical outlier removal over the cargo, before it is split.
    sor_neighbours: int = 250
    sor_std_ratio: float = 1.0

    # -- Bay individualisation (§2.4). DBSCAN finds the small components that k-means put on the
    #    wrong side of its cutting plane; `dbscan_radius` is how close such a component must come
    #    to the other bay to be reassigned to it.
    dbscan_eps: float = 0.10                    # metres (§2.4: 10 cm)
    dbscan_min_points: int = 10
    dbscan_radius: float = 0.06                 # metres

    # -- Base-plane alignment (§2.4).
    crop_m: float = 0.25                        # metres of margin removed at each end (§2.4: 25 cm)
    # Slices **per half**: the tiling is (2, nslices, 1), so the manuscript's 16 slices -- "8
    # slices each" either side of the longitudinal axis -- is nslices = 8.
    nslices: int = 8
    npoints: int = 100                          # lowest points sampled per slice

    # -- Surface reconstruction (§2.5).
    # Normals are estimated on the full-resolution bay, *before* voxelization, so this radius is
    # independent of the cell size below -- see `geometry.mesh.prepare_for_meshing`.
    normal_radius: float = 0.05                 # metres (§2.5: 5 cm)
    k_normals: int = 30                         # neighbours for consistent orientation (§2.5)
    voxel_size: float | None = 0.025            # metres (§2.5: 2.5 cm). None disables decimation.
    poisson_depth: int = 8
    q_filter: float = 0.10                      # density quantile trimmed (§2.5: 10th percentile)

    # -- Reconstruction method. Poisson is the manuscript's; the others exist for the comparison
    #    in the sensitivity analysis, and the convex hull is also the collapse reference.
    method: str = POISSON
    alpha: float = 2.0                          # metres; only the alpha shape reads it
    bpa_radii: tuple[float, ...] = (0.05, 0.10, 0.20)   # metres; ball pivoting only

    # -- Bay split.
    n_bays: int = 2                             # for multi-bay loads; single-bay loads never split
    kmeans_max_iter: int = 1000                 # §2.4
    kmeans_seed: int = 0                        # fixed, or the split is not reproducible

    # -- Whole-truck preprocessing (§2.4).
    downsample_every_k: int = 3                 # keep one point in three
    # Drop exact coordinate duplicates before decimating. The merged clouds are 9-17 % duplicates,
    # and index-based decimation over a set with repeats does not thin it evenly. False reproduces
    # the behaviour of the published run.
    dedup: bool = True

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError('unknown reconstruction method {!r}; expected one of {}'.format(
                self.method, ', '.join(METHODS)))

    @property
    def effective_normal_radius(self) -> float:
        """What `prepare_for_meshing` will actually use, resolved so it can be recorded."""
        return self.normal_radius

    def stage1_key(self) -> tuple:
        """Identity of everything up to and including the aligned, cropped bay cloud.

        Two configurations sharing this key share their Stage 1 output entirely, which is what lets
        the sensitivity sweep compute it once for a group of settings instead of once per setting.

        Normals are in the key even though they feel like a meshing concern: they are estimated
        inside Stage 1, at full resolution. Leaving them out would let two settings that differ
        only in the normal parameters share a cached cloud and report a spurious insensitivity.
        """
        return (self.dedup, self.downsample_every_k,
                self.sor_neighbours, self.sor_std_ratio,
                self.dbscan_eps, self.dbscan_min_points, self.dbscan_radius,
                self.crop_m, self.nslices, self.npoints,
                self.normal_radius, self.k_normals,
                self.n_bays, self.kmeans_max_iter, self.kmeans_seed)


PUBLISHED = Params()


def describe(params: Params, against: Params = PUBLISHED) -> str:
    """One line naming what differs from a reference configuration, for logs."""
    diff = ['{}={}'.format(f.name, getattr(params, f.name))
            for f in fields(Params) if getattr(params, f.name) != getattr(against, f.name)]
    return ', '.join(diff) if diff else 'the published configuration'


# ------------------------------------------------------------------------ configuration files

# What `kmeans_seed` may say instead of a number: draw one. `null` means the same.
RANDOM_SEED = 'random'


def params_from_mapping(mapping: dict) -> Params:
    """A `Params` from a JSON object, with the coercions JSON cannot express."""
    known = {f.name for f in fields(Params)}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise SystemExit('unknown parameter(s) {} -- valid names are: {}'.format(
            ', '.join(unknown), ', '.join(sorted(known))))

    return replace(PUBLISHED, **{name: _coerce(name, value) for name, value in mapping.items()})


def _coerce(name: str, value):
    if name == 'kmeans_seed' and value in (None, RANDOM_SEED):
        # Drawn **once per run**, not per truckload and not per arm, and written into the run's
        # `config_used.json` -- so the run stays as reproducible afterwards as a fixed seed would
        # make it, and both arms split their bays identically. A seed that varied between arms
        # would put clustering jitter inside the arm-to-arm difference, where only segmentation
        # error belongs.
        return secrets.randbelow(2 ** 31)
    if name == 'bpa_radii':
        return tuple(float(v) for v in value)
    if name == 'voxel_size':
        # `null`, and the token the sweep grid uses, both mean "do not decimate"
        return None if value in (None, 'none', '') else float(value)

    default = getattr(PUBLISHED, name)
    if isinstance(default, bool):               # before int: bool is an int subclass
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def config_to_mapping(params: Params) -> dict:
    """A JSON-serialisable object, round-tripping through `params_from_mapping`."""
    mapping = asdict(params)
    mapping['bpa_radii'] = list(mapping['bpa_radii'])
    return mapping


def load_config(path: str | None) -> tuple[Params, bool]:
    """The configuration, and whether its k-means seed was drawn rather than given.

    No path means the published configuration, which is what makes running this package with no
    arguments reproduce the paper.
    """
    if not path:
        return PUBLISHED, False
    if not os.path.exists(path):
        raise SystemExit('no config file at {}'.format(path))

    with open(path) as handle:
        mapping = json.load(handle)
    return params_from_mapping(mapping), mapping.get('kmeans_seed', 0) in (None, RANDOM_SEED)


def write_config(params: Params, path: str) -> None:
    with open(path, 'w') as handle:
        json.dump(config_to_mapping(params), handle, indent=2)
        handle.write('\n')
