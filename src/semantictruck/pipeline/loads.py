"""What a truckload is called, and what its name implies.

The dataset encodes one processing decision in its file names: how many bays a load carries. The
2021 campaign was B-doubles, which carry two stacks and must be split; the 2016 campaign was
semitrailers, which carry one and never reach the clustering at all.

The tag is a parameter, so a dataset naming its loads some other way can be processed without
renaming anything.
"""

from __future__ import annotations

import os
from glob import glob

MULTI_BAY_TAG = '_2021'

CLOUD_EXTENSION = '.laz'


def is_multi_bay(load_id: str, tag: str = MULTI_BAY_TAG) -> bool:
    """Whether this load carries more than one bay and so needs splitting."""
    return tag in load_id


def load_id_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def cloud_path(load_id: str, cloud_dir: str) -> str:
    return os.path.join(cloud_dir, load_id + CLOUD_EXTENSION)


def discover_loads(cloud_dir: str) -> list[str]:
    """Every truckload in a directory, in name order."""
    return sorted(load_id_of(p) for p in glob(os.path.join(cloud_dir, '*' + CLOUD_EXTENSION)))
