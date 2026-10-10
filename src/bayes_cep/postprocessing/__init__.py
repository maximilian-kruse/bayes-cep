"""Generic plotting utilities for the results of runs.

Plotting needs `pyvista` off-screen rendering and `matplotlib`, which a cluster node may lack, so
nothing here is imported by the runs themselves: it is imported only by the separate report step
(`Run.report`). The utilities take arrays and paths, never run configurations; what to plot for
which kind of run is decided in `single_runs/plots.py`.

Modules:
    mesh_plots: Fields on the mesh, rendered off-screen.
    curve_plots: Line plots, e.g. autocorrelation functions and optimization histories.
"""
