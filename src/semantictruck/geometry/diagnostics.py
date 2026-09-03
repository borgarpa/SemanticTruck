"""Mesh and plane diagnostics: what is recorded alongside every volume.

Volume alone cannot separate two failure modes that a calibrated regression hides equally well.
A mesh with large holes biases the magnitude of Eq. 1's integral; a mesh with inverted normals
biases its sign. `boundary_stats` measures the first, `is_watertight` flags the second, and the
plane angle gives the alignment parameters a low-noise response even where their effect on the
volume sits under the residual scatter at N = 29.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def plane_normal(plane) -> np.ndarray:
    """Unit normal of `fit_plane`'s `[a, b, c]`, i.e. of `z = a*x + b*y + c`."""
    a, b = float(plane[0]), float(plane[1])
    normal = np.array([-a, -b, 1.0])
    return normal / np.linalg.norm(normal)


def plane_angle_deg(plane, reference) -> float:
    """Angle between two fitted base planes, in degrees."""
    cosine = float(np.dot(plane_normal(plane), plane_normal(reference)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def boundary_stats(mesh) -> tuple[float, int]:
    """Total length of the mesh's boundary edges and the number of boundary loops.

    A boundary edge belongs to exactly one triangle. Loops are the connected components of the
    graph those edges form, so a closed surface gives (0.0, 0) and every hole adds one loop.
    """
    triangles = np.asarray(mesh.triangles)
    if not len(triangles):
        return 0.0, 0

    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if not len(boundary):
        return 0.0, 0

    vertices = np.asarray(mesh.vertices)
    length = float(np.linalg.norm(vertices[boundary[:, 0]] - vertices[boundary[:, 1]],
                                  axis=1).sum())

    n_vertices = len(vertices)
    graph = coo_matrix(
        (np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])),
        shape=(n_vertices, n_vertices))
    n_components, labels = connected_components(graph, directed=False)
    # Every vertex not on the boundary is its own component; only the ones carrying an edge count
    on_boundary = np.unique(boundary)
    n_loops = int(len(np.unique(labels[on_boundary])))

    return length, n_loops


def mesh_stats(mesh) -> dict:
    """Everything recorded about one reconstructed mesh."""
    boundary_len, n_loops = boundary_stats(mesh)
    return {
        'n_triangles': int(len(mesh.triangles)),
        'n_vertices': int(len(mesh.vertices)),
        'is_watertight': bool(mesh.is_watertight()),
        'boundary_edge_len_m': boundary_len,
        'n_boundary_loops': n_loops,
    }
