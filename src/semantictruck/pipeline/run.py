"""One truckload at one configuration: a LAZ file in, a gross volume and its diagnostics out.

The run is in two stages, and the boundary between them is deliberate.

    Stage 1   read, mask the cargo, decimate, align the truck, split it into bays, level each bay
              against its own base plane, crop below the bed, and estimate normals at full
              resolution. Ends at the aligned bay clouds.
    Stage 2   voxelize, reconstruct a surface, integrate the volume beneath it.

Stage 1 is the expensive half and does not depend on the reconstruction, so the sensitivity sweep
caches it and re-runs only Stage 2 for every setting that differs after that point.

**Nothing here raises.** A load that fails is a row with a `status` and a note, because a campaign
that stops on its eleventh truckload of twenty-nine has told you nothing about the other eighteen.

Two asymmetries worth knowing about before reading the code:

* **The single-bay path pins its DBSCAN**, see `SINGLE_BAY_DBSCAN`.
* **Volume follows `geometry.volume.get_volume`**: the closed-mesh volume where the surface is
  watertight and orientable, Equation 1 otherwise. That puts a discontinuity in the response
  wherever a parameter changes watertightness, so `volume_closed` records which branch each bay
  took and `eq1_m3` carries the Equation 1 value unconditionally, keeping the two comparable on
  every row.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import open3d as o3d

from semantictruck.arms import REFERENCE
from semantictruck.cloud.io import (
    cargo_agreement, cargo_mask, load_cloud, semantic_column)
from semantictruck.cloud.truck import Bay, Truck
from semantictruck.geometry import mesh as meshing
from semantictruck.geometry.diagnostics import mesh_stats
from semantictruck.geometry.volume import get_trimesh_volume, get_volume
from semantictruck.pipeline.loads import MULTI_BAY_TAG, is_multi_bay, load_id_of
from semantictruck.pipeline.params import Params
from semantictruck.pipeline.status import (
    STATUS_CLUSTERING, STATUS_DEGENERATE, STATUS_OK)

# The reference wood mask, carried through the pipeline as a per-point field so that what outlier
# removal discarded can be scored against it afterwards. It has to ride along rather than be
# matched back: by the time bays exist the cloud has been deduplicated, decimated, filtered and
# rotated, and there is no correspondence left to the rows of the file. int32 because Open3D point
# attributes take a numeric dtype, not a bool.
WOOD_FIELD = 'reference_wood'

# The reference bay id, carried the same way so the k-means + DBSCAN split can be scored.
BAY_FIELD = 'reference_bay'

# DBSCAN on the single-bay path is held at these values whatever the configuration asks for.
#
# It is not the same operation as the one the configuration names. On a multi-bay load, DBSCAN
# repairs the k-means split -- it is bay individualisation, and `Params.dbscan_eps` is the
# manuscript's 10 cm for exactly that. A single-bay load never reaches k-means; the DBSCAN inside
# `Bay.preprocess` is there only to keep the largest connected body and drop detached debris. Tying
# the two together would make a parameter documented as affecting bay individualisation silently
# change loads that have no bays to individualise. These are the values the published run used.
SINGLE_BAY_DBSCAN_EPS = 0.06
SINGLE_BAY_DBSCAN_MIN_POINTS = 10


# Below this fraction of the bay's own convex hull, Equation 1 has cancelled rather than measured.
# A sound Poisson bay sits at 0.78-0.91; a fully inverted one came out at 0.022.
#
# **This catches a total inversion only.** A partial one -- around 60 % of a bay's normals flipped,
# which is the failure that fixed the normal search radius -- sits at 0.434 and passes straight
# through. That is deliberate: a low ratio is not always a fault, since ball pivoting at a coarse
# starting radius genuinely leaves holes, and raising the bar to catch 0.434 would condemn those
# runs too. Read `vg_hull_ratio` itself rather than trusting this flag to fire.
COLLAPSE_RATIO = 0.25


# ------------------------------------------------------------------------------------- results

@dataclass
class BayResult:
    """One bay's volume and everything recorded about how it was arrived at."""

    bay_id: int
    bay_runtime_s: float = float('nan')
    vg_m3: float = float('nan')
    eq1_m3: float = float('nan')
    vol_o3d_m3: float = float('nan')
    volume_closed: bool = False
    points_in_bay: int = 0
    points_after_voxel: int = 0
    hull_vg_m3: float = float('nan')
    vg_hull_ratio: float = float('nan')
    n_triangles: int = 0
    n_vertices: int = 0
    is_watertight: bool = False
    boundary_edge_len_m: float = float('nan')
    n_boundary_loops: int = 0
    plane: tuple[float, float, float] | None = None
    min_points_per_slice: int = 0
    slices_filled: int = 0
    slices_requested: int = 0
    slices_underfilled: int = 0
    crop_span_m: float = float('nan')
    base_span_m: float = float('nan')
    # One row of the bay-split confusion matrix: `bay_truth_counts[g]` is how many of the points
    # the split put in this bay carry reference bay `g`, index 0 being "no reference". Counts
    # rather than a metric, so any matching and any score can be recomputed from the written table
    # without re-running the split. Empty when the load carries no reference at all.
    bay_truth_counts: tuple[int, ...] = ()


@dataclass
class LoadResult:
    """One truckload: the load total, the per-bay detail, and the cost of each step."""

    load_id: str
    segmentation: str
    n_bays: int = 0
    vg_m3: float = float('nan')
    status: str = STATUS_OK
    notes: str = ''
    runtime_s: float = float('nan')
    stage1_runtime_s: float = float('nan')
    # Stage 1 broken into its three costs, plus Stage 2 summed over the bays. Measured where the
    # work happens, so `t_read_s + t_preprocess_s + t_bays_s` is Stage 1 and adding `t_mesh_s`
    # accounts for `runtime_s` bar bookkeeping. A run reusing a cached Stage 1 inherits that
    # Stage 1's timings, exactly as it inherits `stage1_runtime_s`.
    t_read_s: float = float('nan')
    t_preprocess_s: float = float('nan')
    t_bays_s: float = float('nan')
    t_mesh_s: float = float('nan')
    points_in: int = 0
    points_after_sor: int = 0
    frac_removed: float = float('nan')
    frac_removed_labelled_wood: float = float('nan')
    points_reassigned: int = 0
    points_discarded: int = 0
    cargo_agreement: float = float('nan')
    normal_radius_m: float = float('nan')
    n_collapsed_bays: int = 0
    # The same histogram over the whole outlier-filtered cargo, i.e. the column totals of that
    # confusion matrix. Points the split discarded are in here and in no bay's row, which is what
    # lets a score charge for them instead of quietly dropping them from both sides.
    bay_truth_total: tuple[int, ...] = ()
    bays: list[BayResult] = field(default_factory=list)

    def as_rows(self) -> list[dict]:
        """Flatten to one row per bay, with the load-level fields repeated.

        The bay's own volume is carried as `vg_m3_bay` while `vg_m3` stays the load total, so
        summing the former back up is a check on the latter rather than a restatement of it.
        """
        head = {k: v for k, v in asdict(self).items() if k != 'bays'}
        head.update(_counts_columns(head.pop('bay_truth_total'), 'ref_total'))

        if not self.bays:
            return [dict(head, bay_id=0, vg_m3_bay=float('nan'))]

        rows = []
        for bay in self.bays:
            values = asdict(bay)
            values['vg_m3_bay'] = values.pop('vg_m3')
            values.update(_counts_columns(values.pop('bay_truth_counts'), 'ref_bay'))
            rows.append(dict(head, **values))
        return rows


def _counts_columns(counts: tuple[int, ...], prefix: str) -> dict:
    """A reference-bay histogram as flat columns, or nothing at all when there is no reference.

    Absent rather than zero on a load with no reference: a zero would read as "no point carried
    bay 1", which is a measurement, and this is the lack of one.
    """
    return {'{}_{}'.format(prefix, bay_id): int(n) for bay_id, n in enumerate(counts)}


# ------------------------------------------------------------------------------------- stage 1

def stage1(laz_path: str, params: Params, arm: str = REFERENCE, pred_dir: str | None = None,
           wood_label: int = 0, multi_bay_tag: str = MULTI_BAY_TAG
           ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """LAZ file -> the aligned, cropped bay clouds with oriented normals, plus diagnostics.

    Each bay comes back as `(points, normals)`, both `(N, 3)`. Normals belong to Stage 1 because
    they are estimated at full resolution, before any voxelization, and depend on nothing the
    reconstruction varies -- so they are computed once per bay and reused by every setting that
    shares this cache.
    """
    load_id = load_id_of(laz_path)

    started = time.perf_counter()
    array, reference_wood, reference_bay = load_cloud(laz_path)

    mask = cargo_mask(load_id, reference_wood, arm, pred_dir, wood_label)
    array[:, 3] = semantic_column(mask)

    diagnostics = {'cargo_agreement': cargo_agreement(reference_wood, mask),
                   't_read_s': time.perf_counter() - started}

    started = time.perf_counter()
    truck = Truck.from_array(array)
    truck.add_field(reference_wood.astype(np.int32), WOOD_FIELD, dtype=o3d.core.int32)
    if reference_bay is not None and reference_bay.any():
        truck.add_field(reference_bay.astype(np.int32), BAY_FIELD, dtype=o3d.core.int32)

    truck = truck.preprocess(every_k_point=params.downsample_every_k, dedup=params.dedup)
    diagnostics['t_preprocess_s'] = time.perf_counter() - started

    # The source array is a copy the Open3D cloud no longer needs -- 364 MB on the largest load
    del array, mask, reference_wood, reference_bay

    started = time.perf_counter()
    if is_multi_bay(load_id, multi_bay_tag):
        bays = _stage1_multi(truck, params, diagnostics)
    else:
        bays = _stage1_single(truck, params, diagnostics)
    diagnostics['t_bays_s'] = time.perf_counter() - started

    return bays, diagnostics


def _stage1_multi(truck: Truck, params: Params, diagnostics: dict) -> list:
    """Two or more bays: denoise the whole cargo, split it, repair the split, level each bay."""
    cargo = truck.get_cargo()

    sor_stats: dict = {}
    cargo_filtered = cargo.preprocess(params.sor_neighbours, params.sor_std_ratio, stats=sor_stats)
    _record_sor(diagnostics, sor_stats, cargo)

    # Counted before the split, so points it discards are in the totals and in no bay -- a score
    # built from these then charges for them rather than dropping them from both sides
    size = _bay_truth_size(cargo_filtered, params.n_bays)
    if size:
        diagnostics['bay_truth_total'] = _bay_truth_counts(cargo_filtered, size)

    clean_stats: dict = {}
    bays = []
    for bay in cargo_filtered.iter_bays(params.n_bays, radius=params.dbscan_radius,
                                        eps=params.dbscan_eps,
                                        min_points=params.dbscan_min_points,
                                        max_iter=params.kmeans_max_iter,
                                        random_state=params.kmeans_seed, stats=clean_stats):
        # Scored as the split leaves it, before `align` and the z >= 0 crop: what is measured is
        # the clustering, and a point the crop drops is not one the clustering got wrong
        if size:
            diagnostics.setdefault('bay_truth_counts', []).append(_bay_truth_counts(bay, size))

        plane_stats: dict = {}
        aligned = bay.align(params.nslices, params.npoints, params.crop_m, stats=plane_stats)
        bays.append(_with_normals(aligned.get_inliers(), params))
        diagnostics.setdefault('bay_stats', []).append(plane_stats)

    diagnostics['points_reassigned'] = clean_stats.get('points_reassigned', 0)
    diagnostics['points_discarded'] = clean_stats.get('points_discarded', 0)

    return bays


def _stage1_single(truck: Truck, params: Params, diagnostics: dict) -> list:
    """One bay. Its DBSCAN is pinned -- see `SINGLE_BAY_DBSCAN_EPS`."""
    bay = truck.get_cargo().to_bay()

    stats: dict = {}
    aligned = bay.preprocess(params.sor_neighbours, params.sor_std_ratio,
                             SINGLE_BAY_DBSCAN_EPS, SINGLE_BAY_DBSCAN_MIN_POINTS,
                             params.nslices, params.npoints, params.crop_m, stats=stats)

    _record_sor(diagnostics, stats, bay)
    diagnostics['points_reassigned'] = 0
    diagnostics['points_discarded'] = 0
    diagnostics['bay_stats'] = [stats]

    return [_with_normals(aligned, params)]


def _has_field(cloud, name: str) -> bool:
    return any(key == name for key, _ in cloud.cld.point.items())


def _bay_truth_size(cloud, n_bays: int) -> int:
    """Length of the reference histogram: one slot per bay id present, plus slot 0 for none.

    Zero when the cloud carries no reference, which is how every caller tests for one. Taken from
    the data rather than from `n_bays`, so a reference naming more bays than the split looks for is
    still counted in full instead of being silently truncated.
    """
    if not _has_field(cloud, BAY_FIELD):
        return 0
    return max(int(cloud[BAY_FIELD].max()), n_bays) + 1


def _bay_truth_counts(cloud, size: int) -> tuple[int, ...]:
    """How many of `cloud`'s points carry each reference bay id, index 0 meaning none."""
    values = cloud[BAY_FIELD].reshape(-1).astype(np.int64)
    return tuple(int(n) for n in np.bincount(values, minlength=size))


def _with_normals(bay: Bay, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """An aligned bay as `(points, normals)`, the normals oriented at full resolution."""
    pcd = bay.prepare_for_meshing(voxel_size=None, k_normals=params.k_normals,
                                  normal_radius=params.normal_radius)
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def _record_sor(diagnostics: dict, stats: dict, before) -> None:
    """Outlier-removal counts, and what fraction of the removed points the reference calls wood."""
    points_in = stats.get('points_in', 0)
    points_after = stats.get('points_after_sor', 0)
    diagnostics['points_in'] = points_in
    diagnostics['points_after_sor'] = points_after
    diagnostics['frac_removed'] = 1 - points_after / points_in if points_in else float('nan')

    keep = stats.get('sor_keep_mask')
    if keep is None or keep.all() or not _has_field(before, WOOD_FIELD):
        diagnostics['frac_removed_labelled_wood'] = float('nan')
        return

    wood = before[WOOD_FIELD].reshape(-1).astype(bool)
    diagnostics['frac_removed_labelled_wood'] = float(wood[~keep].mean())


# ------------------------------------------------------------------------------------- stage 2

def stage2(bay_cloud: tuple[np.ndarray, np.ndarray], params: Params) -> tuple[dict, object]:
    """One aligned bay cloud -> its gross volume and mesh diagnostics.

    Voxelization happens here, on normals Stage 1 has already oriented: `voxel_down_sample`
    averages them across each cell, which is why they must be attached before it runs.
    """
    points, normals = bay_cloud
    pcd = _to_legacy(points, normals)

    if params.voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size=params.voxel_size)
        pcd.normalize_normals()

    mesh = meshing.reconstruct(params.method, pcd, alpha=params.alpha,
                               bpa_radii=params.bpa_radii, poisson_depth=params.poisson_depth,
                               q_filter=params.q_filter)

    result = {'points_in_bay': len(points), 'points_after_voxel': len(pcd.points)}
    result.update(mesh_stats(mesh))

    closed = result['is_watertight'] and mesh.is_orientable()
    result['volume_closed'] = closed
    result['eq1_m3'] = get_trimesh_volume(mesh)
    result['vol_o3d_m3'] = mesh.get_volume() if closed else float('nan')
    result['vg_m3'] = get_volume(mesh)

    # Equation 1 sums *signed* volumes, so a mesh with inconsistent winding cancels towards zero
    # and returns a plausible small number instead of an error. The convex hull of the same cloud
    # is winding-independent and costs milliseconds, so it is the reference that makes such a
    # collapse visible.
    hull_volume = get_volume(meshing.convex_hull(pcd))
    result['hull_vg_m3'] = hull_volume
    result['vg_hull_ratio'] = result['vg_m3'] / hull_volume if hull_volume else float('nan')

    return result, mesh


def _to_legacy(points: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    """The legacy cloud the reconstructions take, carrying Stage 1's oriented normals."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# --------------------------------------------------------------------------------------- the run

def run_load(laz_path: str, params: Params, arm: str = REFERENCE, pred_dir: str | None = None,
             wood_label: int = 0, multi_bay_tag: str = MULTI_BAY_TAG,
             cached: tuple[list, dict] | None = None) -> LoadResult:
    """Run one truckload at one configuration on one arm. Never raises: failures land in `status`.

    `cached` is a Stage 1 result from a configuration whose `stage1_key` matches, in which case
    only Stage 2 re-runs and the Stage 1 diagnostics come from the cache.
    """
    load_id = load_id_of(laz_path)
    result = LoadResult(load_id=load_id, segmentation=arm)
    started = time.perf_counter()

    try:
        if cached is None:
            bays, diagnostics = stage1(laz_path, params, arm, pred_dir, wood_label, multi_bay_tag)
            result.stage1_runtime_s = time.perf_counter() - started
        else:
            bays, diagnostics = cached
            result.stage1_runtime_s = 0.0
    except Exception as exc:                          # noqa: BLE001 -- recorded, not raised
        result.status = STATUS_CLUSTERING
        result.notes = _short(exc)
        result.runtime_s = time.perf_counter() - started
        return result

    _apply_stage1_diagnostics(result, diagnostics, params)
    result.n_bays = len(bays)

    bay_stats = diagnostics.get('bay_stats', [])
    bay_truth = diagnostics.get('bay_truth_counts', [])
    volumes = []
    result.t_mesh_s = 0.0

    for bay_id, bay_cloud in enumerate(bays):
        bay_result = BayResult(bay_id=bay_id)
        _apply_plane_diagnostics(bay_result, bay_stats[bay_id] if bay_id < len(bay_stats) else {})
        if bay_id < len(bay_truth):
            bay_result.bay_truth_counts = bay_truth[bay_id]
        bay_started = time.perf_counter()

        try:
            stats, _ = stage2(bay_cloud, params)
        except Exception as exc:                      # noqa: BLE001
            result.status = STATUS_DEGENERATE
            result.notes = _append(result.notes, 'bay {}: {}'.format(bay_id, _short(exc)))
            # A bay that failed still spent the time, and on the slow reconstructions most of it
            bay_result.bay_runtime_s = time.perf_counter() - bay_started
            result.t_mesh_s += bay_result.bay_runtime_s
            result.bays.append(bay_result)
            continue

        for key, value in stats.items():
            setattr(bay_result, key, value)
        bay_result.bay_runtime_s = time.perf_counter() - bay_started
        result.t_mesh_s += bay_result.bay_runtime_s
        volumes.append(bay_result.vg_m3)
        result.bays.append(bay_result)

        # Recorded, not discarded: the mesh really is what the method produced, so the row keeps
        # its volume and the reader is told the integral cancelled rather than measured
        if bay_result.vg_hull_ratio < COLLAPSE_RATIO:
            result.notes = _append(result.notes,
                                   'bay {} collapsed: vg is {:.3f} of its convex hull '
                                   '(inconsistent mesh normals)'.format(
                                       bay_id, bay_result.vg_hull_ratio))
            result.n_collapsed_bays += 1

    if result.status == STATUS_OK and len(volumes) == len(bays) and bays:
        result.vg_m3 = float(np.sum(volumes))

    result.runtime_s = time.perf_counter() - started
    return result


def _apply_stage1_diagnostics(result: LoadResult, diagnostics: dict, params: Params) -> None:
    for key in ('points_in', 'points_after_sor', 'frac_removed', 'frac_removed_labelled_wood',
                'points_reassigned', 'points_discarded', 'cargo_agreement',
                'bay_truth_total', 't_read_s', 't_preprocess_s', 't_bays_s'):
        if key in diagnostics:
            setattr(result, key, diagnostics[key])
    result.normal_radius_m = params.effective_normal_radius


def _apply_plane_diagnostics(bay_result: BayResult, stats: dict) -> None:
    plane = stats.get('plane')
    if plane is not None:
        bay_result.plane = tuple(float(v) for v in plane)
    for key in ('min_points_per_slice', 'slices_filled', 'slices_requested',
                'slices_underfilled', 'crop_span_m', 'base_span_m'):
        if key in stats:
            setattr(bay_result, key, stats[key])


def _short(exc: Exception) -> str:
    """One line of an exception, enough to identify it in the notes column."""
    return ' '.join('{}: {}'.format(type(exc).__name__, exc).split())[:200]


def _append(notes: str, extra: str) -> str:
    return '{}; {}'.format(notes, extra) if notes else extra


def format_traceback() -> str:
    return traceback.format_exc()
