"""Where the data lives -- the only module in this package that names a location.

Nothing else may hardcode a path. The predecessor of this repository had roughly twenty absolute
paths scattered across its tools, several of them Windows OneDrive paths that could not run on the
machine the analysis was performed on, and the effect was that no tool could be run by anyone but
its author without editing it first.

Resolution order, most specific first:

    an explicit argument           what the CLI's own flag was given
    $SEMANTICTRUCK_DATA            the environment
    ./data, if it exists           a repository-local dataset
    the current working directory  the last resort, so a wrong guess is visible immediately

`data_root()` is deliberately not required to exist: the plotting commands read a results directory
and never touch the clouds, so demanding a dataset to draw a figure would be an obstacle and not a
check. The commands that do need the clouds validate what they were given and say what is missing.

The layout under the root mirrors what the pipeline produces and consumes:

    <root>/clouds/          merged LAZ, one per truckload      (docs/input-data.md)
    <root>/predictions/     per-point label .npy, one per truckload, for the `predicted` arm
    <root>/reference.csv    measured solid volume per bay      (docs/input-data.md)
    <root>/results/         everything this package writes
    <root>/results/figures/

The training stage reads a second representation of the same clouds, because Pointcept consumes
arrays rather than LAZ:

    <root>/training/        train/ and test/ folders of N x 7 .npy clouds
    <root>/folds.json       which truckloads are held out in which cross-validation fold
"""

from __future__ import annotations

import os

ENV_VAR = 'SEMANTICTRUCK_DATA'

CLOUDS_DIR = 'clouds'
PREDICTIONS_DIR = 'predictions'
REFERENCE_CSV = 'reference.csv'
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
TRAINING_DIR = 'training'
FOLDS_FILE = 'folds.json'


def data_root(explicit: str | None = None) -> str:
    """The dataset root: an explicit path, else the environment, else a local `data/`, else `.`."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))

    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return os.path.abspath(os.path.expanduser(from_env))

    if os.path.isdir('data'):
        return os.path.abspath('data')

    return os.path.abspath('.')


def clouds_dir(root: str | None = None) -> str:
    """The merged LAZ truckloads the pipeline reads."""
    return os.path.join(data_root(root), CLOUDS_DIR)


def predictions_dir(root: str | None = None) -> str:
    """Per-point segmentation labels for the `predicted` arm, `<load_id>_pred.npy`."""
    return os.path.join(data_root(root), PREDICTIONS_DIR)


def reference_csv(root: str | None = None) -> str:
    """The measured solid-volume table that calibrates and scores the model."""
    return os.path.join(data_root(root), REFERENCE_CSV)


def results_dir(root: str | None = None) -> str:
    """Where tables are written."""
    return os.path.join(data_root(root), RESULTS_DIR)


def figures_dir(root: str | None = None) -> str:
    return os.path.join(results_dir(root), FIGURES_DIR)


def training_dir(root: str | None = None) -> str:
    """The .npy clouds the segmentation training stage reads. See `docs/reproducing.md`."""
    return os.path.join(data_root(root), TRAINING_DIR)


def folds_file(root: str | None = None) -> str:
    """The cross-validation manifest: which truckloads are held out in which fold."""
    return os.path.join(data_root(root), FOLDS_FILE)


def package_dir() -> str:
    """This package's own directory -- the anchor for `configs/`, which ships with the code."""
    return os.path.dirname(os.path.abspath(__file__))


def repo_file(*parts: str) -> str:
    """A file shipped alongside the source tree, e.g. `repo_file('configs', 'manuscript.json')`.

    The sensitivity grid and the reference configurations are code-adjacent, not data: they define
    what the sweep *is*, so they live in the repository and are found relative to it rather than
    relative to a dataset that may not be present.
    """
    return os.path.abspath(os.path.join(package_dir(), '..', '..', *parts))
