"""The segmentation training stage, and the plots that describe it.

Two halves with very different requirements:

* `curves` reads a metrics CSV and draws it. Pure matplotlib -- no framework, no GPU, no data.
* `dataset`, `backend` and `crossval` drive model training through **Pointcept**, which is a
  dependency installed from source rather than from PyPI and which needs its own environment.

Only `dataset` imports Pointcept, and it is imported only inside the process that runs a Pointcept
tool. So `crossval` -- and therefore `semantictruck-train --help` -- works on a machine that has no
Pointcept at all, and fails with an explanation rather than a traceback when it needs one.
"""

# The two semantic classes, in training-id order: 0 = wood (cargo), 1 = vehicle. Declared here
# rather than beside the dataset class so that reading a campaign's metrics does not require the
# framework that produced them.
CLASS_NAMES = ('wood', 'truck')
