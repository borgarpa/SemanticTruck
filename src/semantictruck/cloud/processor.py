"""A thin, honest wrapper around an Open3D tensor point cloud.

`CloudProcessor` holds an `open3d.t.geometry.PointCloud` together with a cached axis-aligned
bounding box, and gives it the small vocabulary the truckload pipeline is written in: crop, select,
downsample, cluster, denoise, shift, rotate, tile. `semantictruck.cloud.truck` builds
`Truck` -> `Cargo` -> `Bay` on top of it and adds everything specific to a load of timber.

It replaces the base class the original implementation inherited from: an unpublished sibling
package that could not be installed by a reader and that pulled in CuPy, COMPAS, seaborn and tqdm
for code paths this pipeline never executes. Only the members the pipeline reaches are here; the
file I/O, the pandas and PCD/PLY converters, and the alternative outlier estimators are gone.
Reading and writing LAZ belongs to `semantictruck.cloud.io`, which uses `laspy` directly.

**Two conventions live side by side here, and both have callers.**

    shift, rotate                       mutate self and return self
    clone, crop, select_by_index,       return a new object; self is untouched
    remove_by_index, remove_duplicates,
    uniform_downsample, cluster_dbscan,
    remove_noise_sor, concatenate,
    tile_by_patch_number

Neither is "the" convention and neither may be quietly normalised to the other. Call sites reassign
in both cases -- `cld = cld.shift(...)` reads the same whichever it is -- so moving a method from
one column to the other does not raise, does not fail a type check, and returns a plausible cloud.
It only changes which object the caller is now holding. `Truck.align_cabin` returns `self.rotate(R)`
on one branch and `self.clone()` on the other, and the parameter sweep shares a single Stage 1
across every setting in a group, so a mutation that escapes is discovered as a wrong volume rather
than as an error.

One asymmetry to know about before writing to a field: **`points` returns a copy, `__getitem__`
returns a view.** `BaseCloud.align_eigenvector` depends on the copy -- it does
`xyz = cld.points; xyz[:, :-1] = xy` and expects the tensor not to move -- while
`Cargo.shift_cluster_labels`' `labels = self.label; labels[idx] = 1` writes straight into the
tensor, which is harmless only because the receiver there is the throwaway clone that
`cluster_kmeans` just produced. Both call sites are correct as they stand. Do not equalise the two
without re-reading them.
"""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

import numpy as np
import open3d as o3d

T = TypeVar('T', bound='CloudProcessor')

# np.iinfo for integers, np.finfo for floats -- the limits `check_safe_cast` compares against
_INFO_DTYPE = {'i': np.iinfo, 'u': np.iinfo, 'f': np.finfo}


class CloudProcessor(Generic[T]):
    """An Open3D tensor cloud plus a cached bounding box."""

    _dtype_map: dict[np.dtype, o3d.core.Dtype] = {
        np.dtype('float32'): o3d.core.float32,
        np.dtype('float64'): o3d.core.float64,
        np.dtype('int8'): o3d.core.int8,
        np.dtype('int16'): o3d.core.int16,
        np.dtype('int32'): o3d.core.int32,
        np.dtype('int64'): o3d.core.int64,
        np.dtype('uint8'): o3d.core.uint8,
        np.dtype('uint16'): o3d.core.uint16,
        np.dtype('uint32'): o3d.core.uint32,
        np.dtype('uint64'): o3d.core.uint64,
    }
    _inv_dtype_map: dict[o3d.core.Dtype, np.dtype] = {v: k for k, v in _dtype_map.items()}

    # ------------------------------------------------------------------ construction

    def __init__(self, cld: o3d.t.geometry.PointCloud):
        # By reference, not cloned: every operation that produces a new object hands in a cloud it
        # has just built, so cloning here would double the peak memory of each one.
        self._cld = cld
        self._bbox = self.compute_bbox()

    def _instantiate_class(self, cld: o3d.t.geometry.PointCloud) -> T:
        """The factory every new-object operation routes through.

        It is `self.__class__`, not `CloudProcessor`, so `crop`, `select_by_index` and `clone`
        return a `Bay` when called on a `Bay`. The pipeline depends on that throughout.
        """
        return self.__class__(cld)

    def __repr__(self) -> str:
        return self._cld.__repr__().replace('PointCloud', self.__class__.__name__)

    def __len__(self) -> int:
        return self._cld.point.positions.shape[0]

    def __getitem__(self, field: str) -> np.ndarray:
        # A view into the tensor, deliberately -- see the module docstring.
        return self._cld.point[field].numpy()

    def __setitem__(self, field: str, value) -> None:
        self._cld.point[field] = value

    # -------------------------------------------------------------------- properties

    @property
    def cld(self) -> o3d.t.geometry.PointCloud:
        return self._cld

    @cld.setter
    def cld(self, cld: o3d.t.geometry.PointCloud) -> None:
        self._cld = cld
        self._bbox = self.compute_bbox()

    @property
    def points(self) -> np.ndarray:
        """The `(N, 3)` positions, as a copy. The copy is part of the contract."""
        return self._cld.point.positions.numpy().copy()

    @points.setter
    def points(self, arr: np.ndarray) -> None:
        self._cld.point.positions = self.format_field(arr)
        self._bbox = self.compute_bbox()

    @property
    def bbox(self) -> o3d.t.geometry.AxisAlignedBoundingBox:
        """The cached box, as a **tensor** bounding box -- callers read `.min_bound.numpy()`."""
        return self._bbox

    @property
    def min_bounds(self) -> np.ndarray:
        return self._bbox.min_bound.numpy().copy()

    @property
    def max_bounds(self) -> np.ndarray:
        return self._bbox.max_bound.numpy().copy()

    @property
    def size(self) -> np.ndarray:
        """The **extent**, `(3,)` -- `max_bound - min_bound`, not a position."""
        return self.max_bounds - self.min_bounds

    @property
    def fields(self) -> list[str]:
        return [name for name, _ in self._cld.point.items()]

    # --------------------------------------------------------------- casting helpers

    @staticmethod
    def check_safe_cast(arr: np.ndarray, dtype: np.dtype, thresh: float | None = None) -> bool:
        """Whether `arr` survives a cast to `dtype` without leaving its range or losing precision.

        For float targets the precision check is on the two extremes only, which is what the
        original did; it is a range sanity check, not a per-element guarantee.
        """
        if arr.size == 0:
            return True

        info = _INFO_DTYPE[dtype.kind](dtype)
        low, high = arr.min(), arr.max()
        if low < info.min or high > info.max:
            return False

        if dtype.kind != 'f':
            return True

        if thresh is None:
            thresh = info.eps
        return (abs(low.astype(dtype) - low) < thresh
                and abs(high.astype(dtype) - high) < thresh)

    @staticmethod
    def safe_cast(arr: np.ndarray, dtype: np.dtype, thresh: float | None = None) -> np.ndarray:
        """Cast `arr` to `dtype`, refusing rather than corrupting.

        The original did two things this does not. It dropped rows containing NaN --
        `arr[(~np.isnan(arr)).any(axis=1)]`, marked FIXME in its own source -- which removed points
        from *one field* and left it a different length from `positions`, a desynchronisation
        nothing downstream could detect; and it raised `AxisError` on any 1-D input, which is why
        the caller used to have to add a channel axis by hand before every write. It also carried a
        dtype-widening loop that reconstructed the identical dtype each pass, so an unsafe cast
        spun forever instead of widening. That path is reachable: a DBSCAN noise label of -1 headed
        for the uint16 `label` field lands in it.

        Both now raise. A refused cast is a bug in the caller, and it is better seen than survived.
        """
        arr = np.asarray(arr)
        if arr.size and np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
            raise ValueError('refusing to cast a field containing NaN to {}'.format(dtype))
        if not CloudProcessor.check_safe_cast(arr, dtype, thresh):
            raise ValueError('cannot cast {} (range {} to {}) to {} without loss'.format(
                arr.dtype, arr.min(), arr.max(), dtype))
        return arr.astype(dtype, copy=False)

    @staticmethod
    def format_field(array: np.ndarray,
                     dtype: o3d.core.Dtype = o3d.core.float64,
                     device: o3d.core.Device = o3d.core.Device('CPU:0')) -> o3d.core.Tensor:
        """A tensor of exactly `dtype`, shape unchanged.

        **Shape-preserving on purpose.** Adding a channel axis here would be convenient for the
        per-point scalar fields and wrong for `rotate`, which passes a `(3,)` centre. Use
        `add_field` when a field needs the `(N, 1)` layout Open3D's attributes want.
        """
        array = CloudProcessor.safe_cast(array, CloudProcessor._inv_dtype_map[dtype])
        return o3d.core.Tensor(array, dtype, device)

    @staticmethod
    def _as_column(arr: np.ndarray) -> np.ndarray:
        """A per-point scalar field with the channel axis Open3D's `(N, channels)` layout wants."""
        arr = np.asarray(arr)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    @staticmethod
    def add_field_to_cld(cld: o3d.t.geometry.PointCloud, arr: np.ndarray, name: str,
                         dtype: o3d.core.Dtype = o3d.core.float64,
                         device: str = 'CPU:0') -> None:
        """Write a per-point field, giving a 1-D array the channel axis it needs.

        The default is float64 and must stay in step with `add_field`: the two disagreeing is the
        one way a caller relying on the default could silently downcast the colour channels.
        """
        cld.point[name] = CloudProcessor.format_field(
            CloudProcessor._as_column(arr), dtype, o3d.core.Device(device))

    def add_field(self, arr: np.ndarray, name: str,
                  dtype: o3d.core.Dtype = o3d.core.float64, device: str = 'CPU:0') -> None:
        CloudProcessor.add_field_to_cld(self._cld, arr, name, dtype, device)

    # ------------------------------------------------------------------- constructors

    @classmethod
    def concatenate(cls, clouds: list[T]) -> T:
        """One cloud from several, carrying the fields of the first.

        Fields are stacked as columns, so only per-point scalar fields survive intact -- which is
        all this pipeline concatenates.
        """
        merged = o3d.t.geometry.PointCloud()
        positions = np.vstack([cloud.points for cloud in clouds])
        cls.add_field_to_cld(merged, positions, 'positions', cls._dtype_map[positions.dtype])

        for field, tensor in clouds[0]._cld.point.items():
            if field == 'positions':
                continue
            values = np.vstack([cloud._cld.point[field].numpy().reshape(-1, 1)
                                for cloud in clouds])
            cls.add_field_to_cld(merged, values, field, tensor.dtype)

        return cls(merged)

    def clone(self) -> T:
        return self._instantiate_class(self._cld.clone())

    # ----------------------------------------------------------------------- geometry

    def compute_bbox(self) -> o3d.t.geometry.AxisAlignedBoundingBox:
        """The cloud's axis-aligned bounds.

        Open3D warns to stderr and returns a zero box on an empty cloud rather than raising, and
        `tile_by_patch_number` constructs empty crops by the dozen (see its docstring), so the
        warning is suppressed here. An empty cloud with a zero box is the expected case, not a
        problem to be reported thousands of times per campaign.
        """
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            return self._cld.get_axis_aligned_bounding_box()

    @staticmethod
    def min_max_to_bbox(min_coords, max_coords) -> o3d.t.geometry.AxisAlignedBoundingBox:
        # float32 is what every crop in this pipeline has always used
        return o3d.t.geometry.AxisAlignedBoundingBox(
            np.asarray(min_coords, dtype=np.float32),
            np.asarray(max_coords, dtype=np.float32))

    @staticmethod
    def bbox_overlap(a: o3d.t.geometry.AxisAlignedBoundingBox,
                     b: o3d.t.geometry.AxisAlignedBoundingBox) -> tuple[np.ndarray, np.ndarray]:
        """The intersection of two boxes. May come back inverted; callers must check."""
        return (np.maximum(a.min_bound.numpy(), b.min_bound.numpy()),
                np.minimum(a.max_bound.numpy(), b.max_bound.numpy()))

    @staticmethod
    def distance_to(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
        return np.linalg.norm(arr_1 - arr_2, axis=1)

    # --------------------------------------------------------------------- selection

    def crop(self, min_coords, max_coords) -> T:
        """Crop to an absolute-coordinate box. Boundaries are inclusive."""
        return self._instantiate_class(self._cld.crop(self.min_max_to_bbox(min_coords, max_coords)))

    def select_by_index(self, index: np.ndarray, invert: bool = False) -> T:
        index = o3d.core.Tensor(np.ascontiguousarray(index, dtype=np.int64),
                                o3d.core.int64, o3d.core.Device('CPU:0'))
        return self._instantiate_class(self._cld.select_by_index(index, invert=invert))

    def remove_by_index(self, index: np.ndarray) -> T:
        return self.select_by_index(index, invert=True)

    def remove_duplicates(self) -> T:
        """Drop exact coordinate duplicates.

        The merged clouds are 9-17 % exact duplicates, already present in the source data, and
        index-based decimation over a set with repeats does not thin it evenly -- which is why
        `Truck.preprocess` runs this before `uniform_downsample` rather than after.
        """
        return self._instantiate_class(self._cld.remove_duplicated_points()[0])

    def uniform_downsample(self, every_k_point: int) -> T:
        """Keep one point in `every_k_point`, by index."""
        return self._instantiate_class(self._cld.uniform_down_sample(every_k_point))

    # ------------------------------------------------------------ clustering, denoise

    def cluster_dbscan(self, eps: float = 0.02, min_points: int = 10,
                       field_name: str = 'cluster') -> T:
        """DBSCAN over the positions, labels written to `field_name` as int32.

        The dtype matters: Open3D labels noise `-1`, and `Bay.cluster_dbscan` is what shifts the
        labels non-negative afterwards. Storing them unsigned here would refuse the cast.
        """
        cld = self.clone()
        cld.add_field(cld._cld.cluster_dbscan(eps, min_points).numpy(), field_name,
                      dtype=o3d.core.int32)
        return cld

    def remove_noise_sor(self, nb_neighbours: int, std_ratio: float
                         ) -> tuple[T, o3d.core.Tensor]:
        """Statistical outlier removal: the filtered cloud, and the `(N,)` Bool keep-mask.

        The mask is returned raw so the caller can score what was removed against a reference
        labelling; `semantictruck.cloud.truck.as_bool_mask` normalises it.
        """
        cld, mask = self._cld.remove_statistical_outliers(nb_neighbours, std_ratio)
        return self._instantiate_class(cld), mask

    # ------------------------------------------------------------------ transformation

    def shift(self, vector: np.ndarray) -> T:
        """Translate. **In place, and returns self** -- see the module docstring."""
        # Going through the `points` setter is what refreshes the cached bounding box
        self.points = self.points + vector
        return self

    def rotate(self, r: np.ndarray) -> T:
        """Rotate about the origin. **In place, and returns self.**

        The centre is the origin, not the centroid, because every caller has already aligned the
        cloud to it. Open3D's in-place rotation is bitwise identical to `points @ r.T` on a float64
        cloud and does not allocate a second positions tensor, which on the largest load is 364 MB.
        """
        self._cld.rotate(self.format_field(r), self.format_field(np.zeros((3,))))
        self._bbox = self.compute_bbox()
        return self

    # ------------------------------------------------------------------- adjacency

    def is_adjacent(self, cld: T, radius: float = 0.05, *, strict: bool = False) -> bool:
        """Whether `cld` reaches into this cloud's neighbourhood. **Ported as published.**

        Read carefully, because this does not test adjacency between the two clouds. It intersects
        the two bounding boxes, expands the intersection by `radius`, and asks whether that box
        contains any point *of `cld`*. `self` contributes its bounding box and nothing else -- not
        one of its points is looked at.

        That is not a description of the intent; it is a description of the code that produced the
        published results. The original built a KD-tree over `cld` cropped to the expanded overlap
        and then queried it with `cld` cropped to the expanded overlap -- the second was meant to be
        `self`. Querying a tree with the points it was built from finds each point at distance zero,
        so the loop returned True on its first iteration whenever that crop was non-empty, for any
        radius including zero. The emptiness test below is therefore exactly equivalent on every
        input this pipeline can produce, and it does the work without a per-point Python loop.

        `Cargo.clean_bays` uses this to decide which leftover DBSCAN blobs are reassigned to the
        other bay, so the behaviour is inside the published gross volumes. Correcting it changes
        them. `strict=True` runs the intended test -- points of `self` against points of `cld` --
        so the difference can be measured; it is never the default, and making it one requires
        re-running the campaign.
        """
        min_bounds, max_bounds = self.bbox_overlap(self.bbox, cld.bbox)
        min_bounds = min_bounds - radius
        max_bounds = max_bounds + radius
        if np.any(max_bounds < min_bounds):
            return False

        neighbourhood = cld.crop(min_bounds, max_bounds)
        if not strict:
            return len(neighbourhood) > 0

        from scipy.spatial import KDTree                       # only the corrected path needs it

        here = self.crop(min_bounds, max_bounds)
        if not len(here) or not len(neighbourhood):
            return False
        counts = KDTree(neighbourhood.points).query_ball_point(
            here.points, radius, return_length=True)
        return bool(np.any(counts))

    # --------------------------------------------------------------------- tiling

    def tile_by_patch_number(self, n_tiles, with_strides: bool = True) -> Iterator[T]:
        """Split into `n_tiles` pieces per axis, yielding only the non-empty ones.

        **The tile grid is anchored at the world origin, not at the cloud's bounding box.** The
        intervals are built from `self.size`, the *extent*, so they span `[0, extent]` on each
        axis -- and they are then handed to `crop`, which takes absolute coordinates. The grid
        therefore lines up with the data only when the cloud's minimum bound is already near zero,
        and is otherwise offset by exactly that minimum.

        This is the manuscript's base-plane slicing and is reproduced deliberately.
        `Bay.slice_klowest` calls it as `(2, nslices, 1)` on a cloud that `Bay.fit_plane_to_base`
        has just cropped by the margin `crop_m` at each end of the long axis. That crop moves the
        minimum bound to `crop_m` while leaving the grid at zero, so the first `crop_m` of tiles
        come out empty and are dropped here, and the last `crop_m` of the cloud falls off the end
        of the grid and is never sampled. The base plane is fitted to a stack shorter than the crop
        made available. `fit_plane_to_base` records `crop_span_m` against `base_span_m`, and
        `slice_klowest` records `slices_filled` against `slices_requested`, so the shortfall is on
        the record rather than silent.

        `with_strides` is accepted for call-site compatibility and has no effect: with
        `crop_size = extent / n` the stride works out to zero either way.
        """
        for min_coords, max_coords in self._iter_tile_coords(self.size, n_tiles):
            tile = self.crop(min_coords, max_coords)
            if len(tile):
                yield tile

    @staticmethod
    def _iter_tile_coords(size: np.ndarray, n_tiles) -> Iterator[tuple[tuple, tuple]]:
        per_axis = [CloudProcessor._tile_axis(float(extent), int(count))
                    for extent, count in zip(size, n_tiles)]
        for x0, x1 in per_axis[0]:
            for y0, y1 in per_axis[1]:
                for z0, z1 in per_axis[2]:
                    yield (x0, y0, z0), (x1, y1, z1)

    @staticmethod
    def _tile_axis(extent: float, n_crops: int) -> list[tuple[float, float]]:
        """`n_crops` intervals spanning `[0, extent]`, the last one clamped to end at `extent`."""
        crop_size = extent / n_crops
        starts = [min(i * crop_size, extent - crop_size) for i in range(n_crops)]
        return [(start, start + crop_size) for start in starts]

    # ----------------------------------------------------------------- visualisation

    def to_legacy(self) -> tuple[o3d.geometry.PointCloud, o3d.geometry.AxisAlignedBoundingBox]:
        """The legacy cloud and box that the meshing and drawing APIs take."""
        return self._cld.to_legacy(), self._bbox.to_legacy()

    @staticmethod
    def _rgb_by_index(index: int) -> np.ndarray:
        colour = np.zeros((3,), dtype=np.float32)
        colour[index] = 1.0
        return colour

    @staticmethod
    def paint_label(cld: o3d.geometry.PointCloud, labels: np.ndarray) -> None:
        """Colour a legacy cloud by label: the first three get red, green, blue, the rest random."""
        unique = np.unique(labels)
        palette = [CloudProcessor._rgb_by_index(i) for i in range(min(unique.size, 3))]
        palette += [np.random.uniform(0, 1, size=3) for _ in range(max(0, unique.size - 3))]

        colours = np.zeros((labels.shape[0], 3), dtype=np.float32)
        for slot, value in enumerate(unique):
            colours[np.where(labels == value)[0]] = palette[slot]
        cld.colors = o3d.utility.Vector3dVector(colours.astype(np.float32))

    def visualize(self, window_name: str = '', label_field: str | None = None,
                  draw_box: bool = True, bbox_color: tuple = (0, 0, 1)) -> None:
        cld, bbox = self.to_legacy()
        if label_field is not None:
            self.paint_label(cld, self[label_field])

        elements = [cld]
        if draw_box:
            bbox.color = bbox_color
            elements.append(bbox)
        o3d.visualization.draw_geometries(elements, window_name=window_name)
