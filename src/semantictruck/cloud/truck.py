"""The truckload classes: `Truck` -> `Cargo` -> `Bay`, and everything that happens to them.

The pipeline reads as a chain of narrowings, each class holding only what makes sense at its stage:

    Truck    the whole scan, vehicle and timber together, carrying a semantic label per point
    Cargo    the points labelled timber, before it has been divided
    Bay      one stack of logs, levelled against its own base plane and ready to be meshed

All three are Open3D tensor clouds (`semantictruck.cloud.processor.CloudProcessor`), so `crop`,
`select_by_index` and `clone` return the class they were called on and the narrowing survives them.

**The semantic label is 0 = cargo, 1 = vehicle here**, which is not the convention of the file on
disk (1 = wood, 2 = vehicle) nor of every segmenter's output. `semantictruck.cloud.io` is where
that is resolved, once, on the way in -- see its docstring, because getting it wrong inverts the
cargo mask and produces a confident measurement of the truck.

Once bays exist, the same `label` field is reused for **cluster ids**: the k-means bay index, then
DBSCAN's component ids. A "label" therefore means something different before and after the split,
and the classes are what tell you which stage you are at.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import open3d as o3d
from open3d.t.geometry import PointCloud
from sklearn.cluster import KMeans

from semantictruck.cloud.processor import CloudProcessor, T
from semantictruck.geometry import mesh as meshing
from semantictruck.geometry.align import (
    compute_plane_rotation_matrix, eigenvector_2d_align, fit_plane)
from semantictruck.geometry.volume import get_normal_signed_triangles

# Colour is stored 0-1 internally and arrives as 8-bit values in the N x 7 array
COLOR_MAX = np.iinfo(np.uint8).max


def as_bool_mask(mask) -> np.ndarray:
    """An Open3D filter's keep-mask as a plain boolean array, whatever tensor type it came as."""
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    return np.asarray(mask).reshape(-1).astype(bool)


class BaseCloud(CloudProcessor[T]):
    """Colour, semantic label, principal-axis alignment and cluster handling.

    Everything here is shared by the three stages. The colour channels are stored as three separate
    scalar fields rather than one `(N, 3)` attribute because that is how they survive a LAZ round
    trip, and `rgb` assembles them on demand.
    """

    @classmethod
    def from_array(cls, array: np.ndarray):
        """Build from the canonical `N x 7` layout: `[x, y, z, label, r, g, b]`, RGB 8-bit."""
        pcd = PointCloud(o3d.core.Device('CPU:0'))
        pcd.point.positions = cls.format_field(array[:, :3])

        for offset, channel in enumerate(('red', 'green', 'blue')):
            cls.add_field_to_cld(pcd, array[:, 4 + offset] / COLOR_MAX, channel, o3d.core.float64)
        cls.add_field_to_cld(pcd, array[:, 3], 'label', o3d.core.int32)

        return cls(pcd)

    # ------------------------------------------------------------------------- fields

    @property
    def red(self) -> np.ndarray:
        return self._cld.point.red.numpy()

    @red.setter
    def red(self, arr: np.ndarray) -> None:
        self.add_field(arr, 'red', o3d.core.float64)

    @property
    def green(self) -> np.ndarray:
        return self._cld.point.green.numpy()

    @green.setter
    def green(self, arr: np.ndarray) -> None:
        self.add_field(arr, 'green', o3d.core.float64)

    @property
    def blue(self) -> np.ndarray:
        return self._cld.point.blue.numpy()

    @blue.setter
    def blue(self, arr: np.ndarray) -> None:
        self.add_field(arr, 'blue', o3d.core.float64)

    @property
    def rgb(self) -> np.ndarray:
        return np.hstack([self.red.reshape(-1, 1), self.green.reshape(-1, 1),
                          self.blue.reshape(-1, 1)])

    @rgb.setter
    def rgb(self, arr: np.ndarray) -> None:
        self.red, self.green, self.blue = arr[:, 0], arr[:, 1], arr[:, 2]

    @property
    def label(self) -> np.ndarray:
        """The semantic label, or -- after the split -- the cluster id. `(N, 1)`."""
        return self._cld.point.label.numpy()

    @label.setter
    def label(self, arr: np.ndarray) -> None:
        # uint16: labels and cluster ids are small and non-negative by the time they are stored.
        # DBSCAN's -1 for noise is shifted up by `Bay.cluster_dbscan` before it reaches here, and
        # the cast now refuses a negative value rather than wrapping it.
        self.add_field(arr, 'label', o3d.core.uint16)

    def to_numpy(self) -> np.ndarray:
        """Back to the canonical `N x 7` layout, RGB still 0-1."""
        return np.hstack([self.points, self.label, self.rgb]).astype(np.float32)

    def to_legacy(self) -> tuple[o3d.geometry.PointCloud, o3d.geometry.AxisAlignedBoundingBox]:
        cld, bbox = super().to_legacy()
        cld.colors = o3d.utility.Vector3dVector(self.rgb.astype(np.float64))
        return cld, bbox

    # -------------------------------------------------------------------- alignment

    def align_eigenvector(self) -> T:
        """Rotate onto the principal axes, in XY and then in YZ.

        Twice, in two planes, because one 2-D alignment leaves the load free to sit at a pitch: the
        first puts the long axis along Y, the second levels it in the YZ plane. `align` follows
        with the base-plane fit, which is what actually levels the timber against its bed.
        """
        cld = self.clone()
        xyz = cld.points

        xyz[:, :-1] = eigenvector_2d_align(xyz[:, :-1].copy(), major_axis=1)
        xyz[:, 1:] = eigenvector_2d_align(xyz[:, 1:].copy(), major_axis=0)

        cld.points = xyz
        return cld

    def align_and_shift_eigenvector(self) -> T:
        """Align to the principal axes and move the minimum corner to the origin.

        The shift is not cosmetic. `tile_by_patch_number` anchors its grid at the world origin, so
        the lengthwise slicing only lines up with the data on a cloud that starts there -- see that
        method's docstring.
        """
        cld = self.align_eigenvector()
        return cld.shift(-cld.min_bounds)

    # --------------------------------------------------------------------- clusters

    def iter_unique(self) -> Iterator[T]:
        """One sub-cloud per distinct label, in ascending label order."""
        labels = self.label
        for value in np.unique(labels):
            yield self.select_by_index(np.where(labels == value)[0])

    def get_greatest_cluster(self) -> T:
        clusters = list(self.iter_unique())
        return clusters[int(np.argmax([len(c) for c in clusters]))]

    def get_greatest_cluster_index(self) -> np.ndarray:
        return np.where(self.label == self.get_greatest_cluster().label[0].item())[0]

    def split_clusters(self) -> tuple[T, T]:
        """The largest cluster, and everything else."""
        index = self.get_greatest_cluster_index()
        return self.select_by_index(index), self.remove_by_index(index)

    # ---------------------------------------------------------------------- meshing

    def prepare_for_meshing(self, voxel_size: float | None = None, k_normals: int = 30,
                            normal_radius: float = 0.05,
                            with_normals: bool = True) -> o3d.geometry.PointCloud:
        """The legacy cloud every reconstruction consumes: normals first, then decimation.

        A thin bridge to `geometry.mesh.prepare_for_meshing`, which is where the ordering and its
        consequences are explained. Every method meshes this same output, which is what makes them
        comparable.
        """
        return meshing.prepare_for_meshing(self.to_legacy()[0], voxel_size, k_normals,
                                           normal_radius, with_normals)

    def visualize_mesh(self, mesh: o3d.geometry.TriangleMesh, split_by_sign: bool = False,
                       window_name: str = '', draw_box: bool = True,
                       bbox_color: tuple = (0, 0, 1)) -> None:
        """Draw a reconstructed mesh, optionally split by the sign of each triangle normal.

        `split_by_sign=True` is the manuscript's Figure 2, and the quickest way to see whether a
        bay's normals are consistent -- which is what decides whether Equation 1 measures its
        volume or cancels part of it away.
        """
        _, bbox = self.to_legacy()

        if split_by_sign:
            elements = list(get_normal_signed_triangles(mesh))
        else:
            mesh.paint_uniform_color((0.5, 0.5, 0.5))
            elements = [mesh]

        if draw_box:
            bbox.color = bbox_color
            elements.append(bbox)
        o3d.visualization.draw_geometries(elements, mesh_show_back_face=True,
                                          window_name=window_name, mesh_show_wireframe=False)


class Bay(BaseCloud[T]):
    """One stack of logs, and the levelling that makes its volume measurable."""

    def cluster_dbscan(self, eps: float = 0.02, min_points: int = 10) -> T:
        """DBSCAN into the `label` field, shifted so noise is 0 and the first cluster is 1.

        The shift is what lets `clean_bays` tell a real cluster from noise with `label > 0`, and it
        is also what keeps the ids non-negative for the uint16 `label` field.
        """
        cld = super().cluster_dbscan(eps, min_points, 'label')
        cld.label = cld.label + 1
        return cld

    def slice_klowest(self, nslices: int = 8, npoints: int = 100,
                      stats: dict | None = None) -> np.ndarray:
        """The `npoints` lowest points of each lengthwise slice -- the sampled base of the stack.

        `nslices` is **per half**: the tiling is `(2, nslices, 1)`, so the manuscript's 16 slices
        (§2.4, "8 slices each" either side of the longitudinal axis) is `nslices=8`.

        Empty tiles are dropped by the tiler, so fewer slices contribute than were asked for --
        see `tile_by_patch_number` on why the grid's anchoring guarantees some are empty here.
        `stats` records the shortfall rather than leaving it to be inferred.
        """
        points = np.empty((0, 3), dtype=np.float32)
        realised = []

        for slc in self.tile_by_patch_number((2, nslices, 1), with_strides=False):
            slice_points = slc.points.astype(np.float32)
            index = np.argsort(slice_points[:, -1])[:npoints]
            points = np.vstack([points, slice_points[index, :]])
            realised.append(index.size)

        if stats is not None:
            stats['slices_requested'] = 2 * nslices
            stats['slices_filled'] = len(realised)
            stats['min_points_per_slice'] = int(min(realised)) if realised else 0
            stats['slices_underfilled'] = int(sum(n < npoints for n in realised))

        return points

    def fit_plane_to_base(self, nslices: int = 8, npoints: int = 100, extremes_crop: float = 0.25,
                          stats: dict | None = None) -> np.ndarray:
        """The best-fit plane through the base of the stack, as `[a, b, c]` of `z = ax + by + c`.

        `extremes_crop` is the margin removed from each end of the load before sampling
        (manuscript §2.4: 25 cm), so that protruding branches and log ends past the bed do not pull
        the plane down.
        """
        cropped = self.crop(self.min_bounds + np.array([0, extremes_crop, 0]),
                            self.max_bounds - np.array([0, extremes_crop, 0]))
        lowest = cropped.slice_klowest(nslices, npoints, stats=stats)

        if stats is not None:
            # The tile grid is anchored at the origin while the cropped cloud now starts at
            # `extremes_crop`, so the sampled base spans less of the stack than the crop left
            # available. Recording both makes the shortfall visible instead of silent.
            stats['crop_span_m'] = float(cropped.size[1])
            stats['base_span_m'] = (float(lowest[:, 1].max() - lowest[:, 1].min())
                                    if len(lowest) else 0.0)

        return fit_plane(lowest)

    def align_to_plane(self, plane: np.ndarray) -> T:
        """Rotate so the given plane is parallel to XY."""
        rotation = compute_plane_rotation_matrix(plane[0], plane[1])
        cld = self.clone()
        cld.points = np.dot(cld.points, rotation.T)
        return cld

    def align_and_shift_plane(self, nslices: int = 8, npoints: int = 100,
                              extremes_crop: float = 0.25, stats: dict | None = None) -> T:
        """Level against the base plane and drop it to `z = 0`, which is where Equation 1 measures
        from."""
        plane = self.fit_plane_to_base(nslices, npoints, extremes_crop, stats=stats)
        if stats is not None:
            stats['plane'] = np.asarray(plane, dtype=float)

        cld = self.align_to_plane(plane)
        return cld.shift(-np.array([0, 0, plane[-1]]))

    def align(self, nslices: int = 8, npoints: int = 100, extremes_crop: float = 0.25,
              stats: dict | None = None) -> T:
        """Principal axes, then the base plane. The full §2.4 alignment for one bay."""
        cld = self.align_and_shift_eigenvector()
        return cld.align_and_shift_plane(nslices, npoints, extremes_crop, stats=stats)

    def get_inliers(self) -> T:
        """Drop everything below the estimated bed, i.e. `z < 0` after the base-plane shift."""
        return self.select_by_index(np.where(self.points[:, -1] >= 0)[0])

    def preprocess(self, nb_neighbours: int = 250, std_ratio: float = 1.0,
                   eps: float = 0.10, min_points: int = 10,
                   nslices: int = 8, npoints: int = 100, extremes_crop: float = 0.25,
                   stats: dict | None = None) -> T:
        """The single-bay path: denoise, keep the main body, level it, crop below the bed."""
        cld, keep = self.remove_noise_sor(nb_neighbours, std_ratio)
        if stats is not None:
            stats['points_in'] = len(self)
            stats['points_after_sor'] = len(cld)
            stats['sor_keep_mask'] = as_bool_mask(keep)

        cld = cld.cluster_dbscan(eps=eps, min_points=min_points).split_clusters()[0]
        if stats is not None:
            stats['points_after_dbscan'] = len(cld)

        return cld.align(nslices, npoints, extremes_crop, stats=stats).get_inliers()


class Cargo(Bay[T]):
    """The timber of a whole load, before it has been divided into bays."""

    def cluster_kmeans(self, n_clusters: int = 2, max_iter: int = 1000,
                       random_state: int = 0) -> T:
        """k-means over the point coordinates, writing the cluster id to `label`.

        `random_state` is fixed by default. Without it the B-double bay split is not reproducible
        between runs, and neither is anything downstream of it.
        """
        cld = self.clone()
        cld.label = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                           random_state=random_state).fit_predict(cld.points)
        return cld

    def cluster_and_shift_labels(self, n_bays: int, max_iter: int = 1000,
                                 random_state: int = 0) -> T:
        return self.cluster_kmeans(n_bays, max_iter, random_state).shift_cluster_labels()

    def shift_cluster_labels(self) -> T:
        """Renumber so bay 0 is the one nearer the origin along Y.

        k-means numbers its clusters arbitrarily, so without this the same load can come back with
        its bays the other way round from one run to the next, and a per-bay reference volume would
        be matched to the wrong stack.
        """
        labels = self.label
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]

        min_0, min_1 = [cld.min_bounds[1] for cld in self.iter_unique()]
        if min_0 > min_1:
            labels[idx_0] = 1
            labels[idx_1] = 0

        cld = self.clone()
        cld.label = labels
        return cld

    def to_bay(self) -> Bay:
        """The whole cargo as one bay: the single-stack path, where there is nothing to split."""
        return Bay(self.cld.clone())

    def preprocess(self, nb_neighbours: int = 250, std_ratio: float = 1.0,
                   stats: dict | None = None) -> T:
        """Statistical outlier removal over the whole cargo, before it is split.

        Deliberately only the denoising: the two-bay path aligns each bay separately afterwards,
        because bays are often loaded to different heights and levelling them together would tilt
        both.
        """
        cld, keep = self.remove_noise_sor(nb_neighbours, std_ratio)
        if stats is not None:
            stats['points_in'] = len(self)
            stats['points_after_sor'] = len(cld)
            stats['sor_keep_mask'] = as_bool_mask(keep)
        return cld

    def clean_bays(self, radius: float = 0.06, eps: float = 0.10, min_points: int = 10,
                   keep_blobs: bool = False, stats: dict | None = None) -> tuple[Bay, Bay]:
        """Repair the k-means split, which cuts on a plane and so misassigns what straddles it.

        Manuscript §2.4: protruding branches and log ends near the boundary are assigned to the
        neighbouring load, and they are recognisable because they come away as small DBSCAN
        components disconnected from the body they were put in but touching the one they belong to.
        Each bay is clustered, its largest component kept, and every leftover component offered to
        the other bay if it is adjacent to it.

        A leftover that neither bay will take is **discarded** unless `keep_blobs` is set, and the
        count is recorded: those points are genuinely lost from the volume, so a bay-split score
        must charge for them rather than drop them from both sides.
        """
        bay_0, bay_1 = [bay.cluster_dbscan(eps=eps, min_points=min_points)
                        for bay in self.iter_unique()]
        bay_0, leftover_0 = bay_0.split_clusters()
        bay_1, leftover_1 = bay_1.split_clusters()

        groups = ([bay_0], [bay_1])
        reassigned = discarded = 0

        for source, (leftover, other) in enumerate(((leftover_0, bay_1), (leftover_1, bay_0))):
            for blob in leftover.iter_unique():
                # `label > 0` excludes DBSCAN noise, which `Bay.cluster_dbscan` shifted to 0
                if blob.is_adjacent(other, radius) and blob.label[0].item() > 0:
                    groups[1 - source].append(blob)
                    reassigned += len(blob)
                elif keep_blobs:
                    groups[source].append(blob)
                else:
                    discarded += len(blob)

        if stats is not None:
            stats['points_reassigned'] = reassigned
            stats['points_discarded'] = discarded

        bays = []
        for bay_id, group in enumerate(groups):
            bay = Bay.concatenate(group)
            bay.label = np.full_like(bay.label, bay_id)
            bays.append(bay)

        return bays[0], bays[1]

    def iter_bays(self, n_bays: int, radius: float = 0.06, eps: float = 0.10,
                  min_points: int = 10, max_iter: int = 1000, random_state: int = 0,
                  stats: dict | None = None) -> Iterator[Bay]:
        """Split into bays and repair the split: k-means, renumber, then DBSCAN cleanup."""
        clustered = self.cluster_and_shift_labels(n_bays, max_iter, random_state)
        yield from clustered.clean_bays(radius, eps, min_points, stats=stats)


class Truck(BaseCloud[T]):
    """A whole scan: vehicle and timber, before the cargo has been taken out of it."""

    @property
    def cargo_idx(self) -> np.ndarray:
        return np.where(self.label == 0)[0]

    @property
    def truck_idx(self) -> np.ndarray:
        return np.where(self.label == 1)[0]

    def preprocess(self, every_k_point: int = 3, dedup: bool = True) -> T:
        """Decimate and orient the whole truck (manuscript §2.4: one point in three).

        **Duplicates are dropped before the decimation, not after.** The merged clouds carry 9-17 %
        exact coordinate duplicates, already present in the source data, and `uniform_downsample`
        keeps every k-th point *by index* -- so with duplicates left in it thins a set where some
        positions appear several times, and the survivors are unevenly weighted. `dedup=False`
        reproduces the behaviour the published run had.
        """
        cld = self.remove_duplicates() if dedup else self
        cld = cld.uniform_downsample(every_k_point=every_k_point)
        cld = cld.align_and_shift_eigenvector()
        return cld.align_cabin()

    @staticmethod
    def get_180_rot_matrix() -> np.ndarray:
        """Half a turn about Z."""
        return np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])

    def align_cabin(self, every_k_point: int = 2) -> T:
        """Turn the truck so the cabin always ends up at the same end.

        The eigenvector alignment fixes the axis but not the direction, so a load can come out
        either way round. The cabin is found by asking which end of the vehicle the timber stops
        further from -- there is a cab's length of vehicle beyond the cargo at one end and almost
        nothing at the other.

        Note this returns a **rotated self** on one branch and a **clone** on the other, following
        the mutation contract in `processor`. Every caller discards the receiver, so the asymmetry
        does not currently bite; it would if one stopped.
        """
        truck = self.uniform_downsample(every_k_point=every_k_point)
        truck = truck.remove_noise_sor(100, 1)[0]
        truck = truck.shift(-truck.min_bounds)

        cargo = truck.get_cargo()
        distance_to_near_end = cargo.min_bounds[1]
        distance_to_far_end = abs(truck.max_bounds[1] - cargo.max_bounds[1])

        if distance_to_near_end > distance_to_far_end:
            return self.rotate(self.get_180_rot_matrix())
        return self.clone()

    def to_cargo(self) -> Cargo:
        return Cargo(self.cld.clone())

    def get_cargo(self) -> Cargo:
        """The points the segmentation calls timber."""
        return self.select_by_index(self.cargo_idx).to_cargo()
