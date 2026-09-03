"""Solid wood content estimation for roundwood truckloads.

The pipeline turns a segmented photogrammetric point cloud of a loaded truck into an estimate of
the solid (net) wood volume it carries: split the cargo into bays, level each bay against its own
base plane, reconstruct a surface, integrate the volume beneath it, and regress solid volume on
that gross volume.

**This package deliberately exports nothing at import time.** Importing a submodule must not pull
in the ones it does not need -- `semantictruck.analysis.figures` draws from CSV files and has no
business loading Open3D, and the plotting commands must stay usable on a machine that has no
dataset and no GPU. Import what you need:

    from semantictruck.pipeline.run import run_load        # geometry; needs open3d
    from semantictruck.analysis import stats, figures      # tables and plots; pandas only
"""

__version__ = '1.0.0'
