"""Aligning a truckload: principal axes in the XY plane, and the base plane beneath the stack.

Two alignments happen, for different reasons.

The **eigenvector alignment** turns the load so its long axis runs along Y. It is applied twice --
once in XY and once in YZ -- because the truck arrives at an arbitrary heading and the bay split,
the lengthwise slicing and the bounding boxes all assume a canonical orientation.

The **base-plane alignment** levels each bay against the surface it rests on. Gross volume is the
signed sum of the volume beneath every mesh triangle (`geometry.volume`, Equation 1 of the
manuscript), which is measured from `z = 0`; a bay sitting at a tilt therefore integrates a wedge
of empty space, or loses one. Manuscript §2.4: the base is the *k* lowest points sampled at regular
intervals along the longitudinal axis, the best-fit plane through them is rotated parallel to XY,
and the load is translated down by that plane's intercept.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq


def eigenvector_2d_align(arr: np.ndarray, major_axis: int = 0) -> np.ndarray:
    """Rotate a 2-D point set onto its principal axes, major axis first.

    **Centres `arr` in place** before rotating: the caller is expected to have passed a copy, which
    is what `CloudProcessor.points` returns. The column reordering is what pins the *major* axis to
    the requested output axis -- the eigenvectors come back sorted by eigenvalue, but the sign and
    the assignment of each to X or Y do not follow from that.
    """
    arr -= arr.mean(0)
    eigvec = compute_eigenvector_rotation_matrix(arr)

    # Send the object's major axis to `major_axis` and the minor one to the other column
    idx = np.transpose(eigvec.argmax(0)[[major_axis, int(not major_axis)]])
    eigvec = eigvec[:, idx]

    return np.dot(arr, eigvec)


def compute_eigenvector_rotation_matrix(arr: np.ndarray) -> np.ndarray:
    """Eigenvectors of the covariance, as columns, ordered by decreasing eigenvalue."""
    cov = np.cov(arr, rowvar=False)
    eigval, eigvec = np.linalg.eig(cov)
    return eigvec[:, eigval.argsort()[::-1]]


def fit_plane(array: np.ndarray) -> np.ndarray:
    """Least-squares plane through `(x, y, z)` points, returned as `[a, b, c]` of `z = ax + by + c`.

    The L2 minimisation of manuscript §2.4. It is a *height* fit, not an orthogonal one: it cannot
    represent a vertical plane, which is the right restriction for a load bed.
    """
    A = np.hstack([array[:, :-1], np.ones((array.shape[0], 1), dtype=np.float32)])
    coefficients, _, _, _ = lstsq(A, array[:, -1])
    return coefficients


def compute_plane_rotation_matrix(a: float, b: float) -> np.ndarray:
    """Rotation making the plane `z = ax + by + c` parallel to XY.

    About X and Y only -- a rotation about Z would turn the load out of the lengthwise alignment
    that the bay split and the slicing depend on, and would not change the plane's tilt anyway.
    """
    theta = np.arctan2(b, np.sqrt(a ** 2 + 1))          # about X
    phi = -np.arctan(a)                                 # about Y, after the X rotation

    rx_inv = np.array([[1, 0, 0],
                       [0, np.cos(-theta), -np.sin(-theta)],
                       [0, np.sin(-theta), np.cos(-theta)]])
    ry_inv = np.array([[np.cos(-phi), 0, np.sin(-phi)],
                       [0, 1, 0],
                       [-np.sin(-phi), 0, np.cos(-phi)]])

    # Inverse of the forward rotation, so the factors multiply in the reverse order
    return rx_inv @ ry_inv
