"""The truckload dataset, as Pointcept sees it.

Importing this module **registers `SemanticTruckDataset` with Pointcept's registry**, which is
why `semantictruck.training.backend` imports it before running any Pointcept tool. It therefore
imports Pointcept at module level and will not import at all without it -- deliberately, since a
dataset class for a framework that is not installed cannot do anything useful.

**The training stage reads a different file layout from the rest of this package.** The geometry
stage reads one `.laz` per truckload (`docs/input-data.md`); training reads the same clouds as
`N x 7` `.npy` arrays under `train/` and `test/` folders. They hold the same points in the same
order, which is what lets a prediction be matched back to a `.laz` row for volume estimation.

On-disk layout of one cloud is `[x, y, z, label, r, g, b]`. Two things about it are easy to get
wrong: **the label sits at column 3, not last**, and colours are 0-255 rather than 0-1, because
Pointcept's `NormalizeColor` divides by 255 later in the pipeline.

Labels on disk are `{0: unclassified, 1: wood, 2: truck}`, where 0 is declared but never occurs.
They are remapped here to `{0: wood, 1: truck}` with anything else ignored, so the network emits
two logits. **The consequence is the prediction convention**: inference writes 0 = wood = cargo,
which is what `--pred-wood-label 0` expects on the geometry side. A model trained some other way
will write some other convention, and that flag is where it is declared.

Splits are **cross-validation folds, not directories**. `split` names a fold and `fold_file` points
at the manifest mapping fold index to load names, so the fold assignment is read rather than
regenerated and every campaign holds out the same loads.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

try:
    from pointcept.datasets.builder import DATASETS
    from pointcept.datasets.defaults import DefaultDataset
except ImportError as missing:                                  # pragma: no cover
    raise ImportError(
        '{}\n\nsemantictruck.training.dataset needs Pointcept importable. It is a dependency of '
        'the training stage only; see docs/reproducing.md.'.format(missing))

# Folders that together hold every load exactly once. A `val/` folder, if the dataset has one,
# is deliberately not read: in the published dataset it holds byte-identical copies of `test/`,
# and globbing all three would count ten loads twice and leak them across folds.
SOURCE_SPLITS = ('train', 'test')

# On disk -> contiguous training ids. Everything not listed becomes `ignore_index`.
LABEL_MAP = {1: 0, 2: 1}                                        # 1 wood -> 0, 2 truck -> 1


# Registered under this package's own name rather than a generic one. A Pointcept checkout that
# still carries a vendored copy of an earlier version of this class registers it as
# `TruckSegmDataset`, and Pointcept's registry raises on a duplicate name -- so sharing that name
# would make the two mutually exclusive. `SemanticTruckDataset` is unambiguous: the configuration
# asking for it can only get this file.
@DATASETS.register_module()
class SemanticTruckDataset(DefaultDataset):
    """Truckload point clouds, split by cross-validation fold.

    Args:
        split: a fold name (`"fold_3"`) or a tuple of them. The bare index (`"3"`) is accepted too.
        data_root: the directory holding `train/` and `test/`.
        fold_file: the folds manifest, `{"folds": {"0": [load names], ...}}`. Required.
    """

    def __init__(self, fold_file=None, **kwargs):
        self.fold_file = fold_file
        super().__init__(**kwargs)

    # ------------------------------------------------------------------- the data list

    def _load_folds(self) -> dict:
        if self.fold_file is None:
            raise ValueError('TruckSegmDataset needs fold_file=<path to the folds manifest>; '
                             'splits here are cross-validation folds, not directories')
        with open(self.fold_file) as handle:
            manifest = json.load(handle)
        return manifest.get('folds', manifest)

    def _index_loads(self) -> dict:
        """Load name -> path, over the folders that hold each load exactly once."""
        paths = {}
        for source in SOURCE_SPLITS:
            source_dir = os.path.join(self.data_root, source)
            if not os.path.isdir(source_dir):
                raise FileNotFoundError('no {} directory in {}'.format(source, self.data_root))
            for path in sorted(glob.glob(os.path.join(source_dir, '*.npy'))):
                name = os.path.basename(path)[:-4]
                if name in paths:
                    raise ValueError('load {} appears in more than one of {}'.format(
                        name, SOURCE_SPLITS))
                paths[name] = path
        return paths

    def get_data_list(self) -> list:
        folds = self._load_folds()
        paths = self._index_loads()
        wanted = [self.split] if isinstance(self.split, str) else list(self.split)

        data_list = []
        for split in wanted:
            key = str(split)[len('fold_'):] if str(split).startswith('fold_') else str(split)
            if key not in folds:
                raise ValueError('no fold {} in {} (it has {})'.format(
                    key, self.fold_file, sorted(folds)))
            for name in folds[key]:
                if name not in paths:
                    raise FileNotFoundError('load {} of fold {} is not under {}'.format(
                        name, key, SOURCE_SPLITS))
                data_list.append(paths[name])

        # The folds must partition the loads; overlapping ones would leak a held-out truck
        if len(data_list) != len(set(data_list)):
            raise ValueError('the requested folds overlap: {}'.format(wanted))

        return data_list

    # ------------------------------------------------------------------------ the data

    def get_data(self, idx: int) -> dict:
        cloud = np.load(self.data_list[idx % len(self.data_list)])

        coord = cloud[:, 0:3].astype(np.float32)
        color = cloud[:, 4:7].astype(np.float32)

        label = cloud[:, 3]
        segment = np.full(cloud.shape[0], self.ignore_index, dtype=np.int32)
        for on_disk, training_id in LABEL_MAP.items():
            segment[label == on_disk] = training_id

        return dict(coord=coord, color=color, segment=segment,
                    name=self.get_data_name(idx), split=self.get_split_name(idx))

    def get_data_name(self, idx: int) -> str:
        # Used verbatim for the prediction file that inference writes, `<name>_pred.npy`
        return os.path.basename(self.data_list[idx % len(self.data_list)])[:-4]

    def get_split_name(self, idx: int) -> str:
        # The source folder, not the fold. Only multi-dataset conditioning reads it, which is off.
        return os.path.basename(os.path.dirname(self.data_list[idx % len(self.data_list)]))
