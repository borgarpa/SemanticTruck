"""Everything downstream of the gross volume: the regression, the error tables and the figures.

Nothing here imports Open3D. These modules read the CSV files the pipeline writes, so they run on a
machine that has neither the clouds nor a GPU.
"""
