"""Surface reconstruction: a prepared point cloud in, a triangle mesh out.

Four methods are available. Poisson is the manuscript's (§2.5); the other three exist because the
sensitivity analysis compares them, and because the convex hull is the winding-independent
reference that makes a collapsed Equation 1 integral visible (`geometry.volume`).

These are free functions over an `open3d.geometry.PointCloud`, not methods on the cloud classes.
Every call site already prepares the cloud first and passes it in, so the object the method would
have been bound to was never read -- and keeping the reconstruction here means the four methods
mesh demonstrably the same input, which is the whole basis of comparing them.

**`prepare_for_meshing` estimates normals before voxelizing, and the order is load-bearing.**
Orienting normals on a decimated cloud is bistable: at a 5 cm cell a 7 cm search disc holds about
six lattice points, fewer than the orientation graph needs, and whole regions of the surface come
out flipped. Equation 1 sums *signed* volumes, so a flipped region cancels rather than raising and
the bay returns a plausible small number -- 0.5 m³ against a true 20 m³ on one measured load. The
failure is not a smooth response either: at a 0.10 m radius it collapses for k = 30 but not k = 15.
Estimating at full resolution is well conditioned, and it decouples the search radius from the cell
size, so the two can be varied independently. Never move the estimation after the voxelization.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

POISSON = 'poisson'
ALPHA = 'alpha'
HULL = 'hull'
BALL_PIVOTING = 'ball_pivoting'
METHODS = (POISSON, ALPHA, HULL, BALL_PIVOTING)

# Points inspected per normal estimation, alongside the search radius. Manuscript §2.5 calls this
# the maximum neighbourhood size.
MAX_NN = 30


def prepare_for_meshing(cld: o3d.geometry.PointCloud, voxel_size: float | None = None,
                        k_normals: int = 30, normal_radius: float = 0.05,
                        with_normals: bool = True) -> o3d.geometry.PointCloud:
    """Attach oriented normals, then optionally decimate. See the module docstring on the order.

    `voxel_size=None` (or 0) skips the decimation. Normals are re-normalised after it because
    averaging unit vectors inside a cell shortens them, and Poisson wants them unit length.
    """
    if with_normals:
        cld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=MAX_NN))
        cld.orient_normals_consistent_tangent_plane(k_normals)

    if voxel_size:
        cld = cld.voxel_down_sample(voxel_size=voxel_size)
        if with_normals:
            cld.normalize_normals()

    return cld


def convex_hull(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """The convex hull. Reads positions only -- no normals, so orientation cannot affect it."""
    mesh = pcd.compute_convex_hull()[0]
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def alpha_shape(pcd: o3d.geometry.PointCloud, alpha: float = 2.0) -> o3d.geometry.TriangleMesh:
    """Alpha shape, `alpha` in **metres**.

    Also position-only. The complex closes and is watertight at alpha = 2 m; an order of magnitude
    below that it is open, and its interior facets make Equation 1 cancel non-monotonically -- on
    one bay against a 29.3 m³ hull, alpha 0.1 / 0.3 / 0.5 / 1.0 m give 0.8 / 12.3 / 7.1 / 15.3 m³.
    """
    tetra_mesh, point_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, point_map)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def ball_pivoting(pcd: o3d.geometry.PointCloud, radii) -> o3d.geometry.TriangleMesh:
    """Ball-pivoting reconstruction over an ascending sequence of radii. Needs oriented normals.

    Open3D pivots ascending and each pass reuses only the edges the previous ones left free, so the
    *smallest* radius governs both cost and completeness. A set starting near the object scale
    degrades badly -- radii from 0.20 m ran for over half an hour on a single load and still
    returned a degenerate mesh.
    """
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(sorted(radii)))
    mesh = clean_mesh(mesh)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def poisson(pcd: o3d.geometry.PointCloud, depth: int = 8,
            q_filter: float = 0.10) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction -- the manuscript's method (§2.5).

    Each vertex comes back with the density of the samples that supported it, and the sparsest
    `q_filter` of them are removed: those are the vertices the solver invented where it had no
    data, and they are what produce the aberrant geometry that would otherwise be integrated.
    Manuscript §2.5 uses octree depth 8 and the 10th percentile.

    `n_threads=1` because the campaign runs loads sequentially and gives each the whole machine;
    Open3D's threaded Poisson is also not bit-reproducible across thread counts.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=1)

    mesh.remove_vertices_by_mask(np.asarray(densities) < np.quantile(densities, q_filter))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    mesh = clean_mesh(mesh)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove the degeneracies that break watertightness tests and Equation 1 alike."""
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    return mesh


def reconstruct(method: str, pcd: o3d.geometry.PointCloud, *, alpha: float = 2.0,
                bpa_radii=(0.05, 0.10, 0.20), poisson_depth: int = 8,
                q_filter: float = 0.10) -> o3d.geometry.TriangleMesh:
    """Dispatch to one reconstruction. The single place a method name is resolved."""
    if method == POISSON:
        return poisson(pcd, poisson_depth, q_filter)
    if method == ALPHA:
        return alpha_shape(pcd, alpha)
    if method == HULL:
        return convex_hull(pcd)
    if method == BALL_PIVOTING:
        return ball_pivoting(pcd, list(bpa_radii))
    raise ValueError('unknown reconstruction method {!r}; expected one of {}'.format(
        method, ', '.join(METHODS)))
