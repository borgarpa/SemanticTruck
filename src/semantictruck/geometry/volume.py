"""Gross volume from a reconstructed surface -- Equation 1 of the manuscript.

The volume beneath a mesh is the sum, over every triangle, of the signed volume of the prism
between that triangle and the plane `z = 0`:

    V_i = (z1 + z2 + z3) * ((x1*y2 - x2*y1) + (x2*y3 - x3*y2) + (x3*y1 - x1*y3)) / 6

The sign follows the z component of the triangle's normal, so downward-facing triangles subtract
what upward-facing ones added and a closed surface leaves exactly its enclosed volume. The total is
taken absolute, because the winding of a reconstructed surface is arbitrary.

**That cancellation is also this pipeline's most dangerous failure mode.** If normal orientation
inverts a *region* of the surface rather than all of it, the region's contribution is subtracted
instead of added and the bay returns a plausible smaller number -- no exception, no warning, a
volume that looks like a slightly light load. The pipeline defends against it in two places: normals
are estimated at full resolution *before* voxelization (`Bay.prepare_for_meshing`), and every bay's
volume is recorded as a fraction of its own convex hull, which is winding-independent and therefore
makes a cancelled integral visible. Sound bays sit around 0.78-0.91 of their hull.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from open3d.geometry import TriangleMesh


def volume_under_triangle(vertices: np.ndarray, triangle_idx: np.ndarray) -> float:
    """Equation 1 for a single triangle.

    Kept as the readable statement of the formula and as the reference `get_trimesh_volume` is
    checked against; the vectorised form agrees with it to ~1e-14 relative.
    """
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = vertices[triangle_idx, :]
    return (z1 + z2 + z3) * (x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3) / 6


def get_trimesh_volume(mesh: TriangleMesh) -> float:
    """Equation 1 over the whole mesh: signed volumes summed, then taken absolute."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not len(triangles):
        return 0.0

    p1, p2, p3 = (vertices[triangles[:, i], :] for i in range(3))
    x1, y1, z1 = p1.T
    x2, y2, z2 = p2.T
    x3, y3, z3 = p3.T

    volumes = (z1 + z2 + z3) * (x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3) / 6
    return abs(volumes.sum())


def get_volume(mesh: TriangleMesh) -> float:
    """The mesh's volume: closed-form where the surface admits it, Equation 1 otherwise.

    Open3D's `get_volume` is exact but requires a mesh that is both watertight and orientable.
    Alpha-shape and Poisson surfaces frequently are neither -- which is precisely why Equation 1
    exists, and why a gross volume is still reported for them rather than a failure.
    """
    if mesh.is_watertight() and mesh.is_orientable():
        return mesh.get_volume()
    return get_trimesh_volume(mesh)


def get_normal_signed_triangles(mesh: TriangleMesh) -> tuple[TriangleMesh, TriangleMesh]:
    """Split a mesh by the sign of each triangle normal's z component: (upward, downward).

    Painted blue and red respectively -- this is the manuscript's Figure 2, which shows that a
    sound reconstruction carries a consistent proportion of each. It is the visual form of the
    cancellation described in the module docstring, and the quickest way to see an inverted region.
    """
    normals = np.asarray(mesh.triangle_normals)
    downward = np.where(normals[:, -1] < 0)[0]
    upward = np.where(normals[:, -1] >= 0)[0]

    up_mesh, down_mesh = deepcopy(mesh), deepcopy(mesh)
    up_mesh.remove_triangles_by_index(downward)
    down_mesh.remove_triangles_by_index(upward)

    up_mesh.paint_uniform_color((0, 0, 1))
    down_mesh.paint_uniform_color((1, 0, 0))

    return up_mesh, down_mesh
