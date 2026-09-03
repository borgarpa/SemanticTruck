"""How a run ended.

Its own module, with no imports at all, because both halves of the package need it: the pipeline
sets it, and the analysis filters on it. Were it to live beside the geometry, reading a results CSV
would pull in Open3D -- and the plotting and table commands are meant to run on a machine that has
neither the clouds nor a GPU.
"""

STATUS_OK = 'ok'
STATUS_DEGENERATE = 'degenerate_mesh'
STATUS_CLUSTERING = 'clustering_failed'
STATUS_TIMEOUT = 'timeout'

STATUSES = (STATUS_OK, STATUS_DEGENERATE, STATUS_CLUSTERING, STATUS_TIMEOUT)
