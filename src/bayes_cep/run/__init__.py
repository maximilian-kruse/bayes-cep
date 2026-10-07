"""Generic machinery for runs and studies; the concrete runs live in the top-level `single_runs`.

A run (`template.Run`) is a pure function of one frozen configuration (`config.RunConfig`). A study
(`study.Study`) is a fixed list of runs from a base configuration and sweeps, executed by an
`executor.Executor` and analyzed by a `collector.Collector`; `directories` defines the layout on
disk.
"""
