"""Reading a truckload, and deciding which of its points are timber.

Two arms run through the whole pipeline, identical downstream of this module, which is what makes
them comparable -- a load's `predicted` result differs from its `reference` result only because the
cargo mask differs:

    reference    the `label` dimension of the LAZ file: a human segmentation
    predicted    `<pred_dir>/<load_id>_pred.npy`: whatever a segmentation model produced

The calibration is fitted on the `reference` arm alone and applied to both, so the `predicted`
arm's extra error is attributable to the segmentation rather than absorbed by a refit.

**Label conventions differ between stages and getting one wrong is silent.** The file on disk uses
1 = wood, 2 = vehicle. The analysis stack uses 0 = cargo, 1 = vehicle, because `Truck.cargo_idx`
selects on it. A model's output uses whatever its framework wrote: the model behind the published
results emitted 0 = wood, while other segmentation stacks in the same project emitted 1 = wood and
needed shifting first. Applying the wrong one does not raise -- it inverts the cargo mask and
returns a confident measurement of the vehicle. So the prediction convention is an explicit
argument, not a constant, and `cargo_mask` verifies the file's values against what was declared.

Predictions must be **row-aligned** with the cloud: one label per point, in file order. There is no
way to check that beyond the length, which is why `check_predictions` also reports each load's
agreement with the reference mask -- a row-shifted file shows up there as agreement near chance.
"""

from __future__ import annotations

import os

import laspy
import numpy as np

from semantictruck.arms import ARMS, PREDICTED, REFERENCE

PRED_SUFFIX = '_pred.npy'

# In the LAZ file
WOOD_LABEL = 1

# The extra dimension carrying a human bay segmentation: 1-based, 0 meaning "no bay" -- every
# vehicle point, and every point of a load nobody segmented by hand. Optional.
BAY_DIM = 'bay'

# In the analysis stack, as `Truck.cargo_idx` expects
CARGO, TRUCK = 0, 1

# Canonical column order of the N x 7 array `Truck.from_array` takes
FIELDS = ('x', 'y', 'z', 'label', 'red', 'green', 'blue')


def load_cloud(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read a LAZ truckload into the `N x 7` layout, plus its reference masks.

    Returns `(array, wood_mask, bay_ids)`. The array's semantic column is left at zero: it is
    filled in from whichever arm's mask the caller chose. `bay_ids` is `None` on a cloud with no
    `bay` dimension -- read the caveat in `pipeline.bays` before treating it as ground truth, since
    a single-bay load's `bay` field is not a segmentation of anything.
    """
    las = laspy.read(path)

    rgb = np.stack([las.red, las.green, las.blue], axis=1) / np.iinfo(np.uint16).max
    array = np.column_stack([las.x, las.y, las.z,
                             np.zeros(len(las.x)),
                             rgb * np.iinfo(np.uint8).max])

    bays = (np.asarray(las[BAY_DIM]).astype(np.int32)
            if BAY_DIM in las.point_format.dimension_names else None)

    return array, np.asarray(las.label) == WOOD_LABEL, bays


def prediction_path(load_id: str, pred_dir: str) -> str:
    return os.path.join(pred_dir, load_id + PRED_SUFFIX)


def cargo_mask(load_id: str, reference_wood: np.ndarray, arm: str = REFERENCE,
               pred_dir: str | None = None, wood_label: int = 0) -> np.ndarray:
    """The per-point cargo mask of one arm.

    `wood_label` is the value the prediction file uses for wood -- 0 for a model whose classes are
    ordered (wood, vehicle), 1 for one that writes 1-based labels. The file's distinct values are
    checked against `{wood_label, other}` and a mismatch raises, because a wrong convention would
    otherwise segment the vehicle and report it as timber.
    """
    if arm == REFERENCE:
        return reference_wood
    if arm != PREDICTED:
        raise ValueError('unknown arm {!r}; expected one of {}'.format(arm, ARMS))
    if not pred_dir:
        raise ValueError('the {!r} arm needs a directory of prediction files'.format(PREDICTED))

    path = prediction_path(load_id, pred_dir)
    if not os.path.exists(path):
        raise FileNotFoundError('no prediction for {} at {}'.format(load_id, path))

    predicted = np.load(path).reshape(-1)
    if len(predicted) != len(reference_wood):
        raise ValueError('{}: {:,} predictions for {:,} points -- predictions must be row-aligned '
                         'with the cloud'.format(load_id, len(predicted), len(reference_wood)))

    values = np.unique(predicted)
    expected = {int(wood_label), int(wood_label) + 1 if wood_label == 0 else int(wood_label) - 1}
    if not set(int(v) for v in values) <= expected:
        raise ValueError(
            '{}: predictions carry labels {} but --pred-wood-label {} declares {}. A wrong '
            'convention inverts the cargo mask silently.'.format(
                load_id, list(values), wood_label, sorted(expected)))

    return predicted == wood_label


def semantic_column(mask: np.ndarray) -> np.ndarray:
    """The analysis stack's label column: 0 = cargo, 1 = vehicle."""
    return np.where(mask, CARGO, TRUCK).astype(float)


def cargo_agreement(reference_wood: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of points the two arms agree on. Constant per load; recorded for context."""
    return float((reference_wood == mask).mean())


def check_predictions(load_ids, cloud_dir: str, pred_dir: str,
                      wood_label: int = 0) -> list[dict]:
    """Verify every prediction against its cloud before a campaign spends hours on it.

    Raises on the first length or convention mismatch. The returned agreement per load is the only
    available check on row alignment: a shifted file reads as agreement near chance rather than the
    high nineties, and is worth stopping for even though nothing here can prove it.
    """
    report = []
    for load_id in load_ids:
        _, reference_wood, _ = load_cloud(os.path.join(cloud_dir, load_id + '.laz'))
        predicted = cargo_mask(load_id, reference_wood, PREDICTED, pred_dir, wood_label)
        report.append({
            'load_id': load_id,
            'points': len(reference_wood),
            'reference_cargo': int(reference_wood.sum()),
            'predicted_cargo': int(predicted.sum()),
            'cargo_agreement': cargo_agreement(reference_wood, predicted),
        })
    return report
