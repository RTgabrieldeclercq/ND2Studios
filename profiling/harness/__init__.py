"""Profiling scenarios driven by :mod:`profiling.harness.run_all`.

Each ``scenario_*`` module exposes a ``run()`` function that returns a
list of ``Measurement.to_dict()`` records. Scenarios degrade gracefully
when their inputs (test data, optional dependencies) are missing.
"""
