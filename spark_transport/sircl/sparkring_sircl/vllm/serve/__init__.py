"""Serve launcher: SparkRing serving profiles on groups of Sparks of a ring, with SIRCL in front of every collective.

``python -m sparkring_sircl.vllm.serve <command>`` (:mod:`.cli`); the
procedure and safety classes are in ``sparkring_sircl/vllm/RUNBOOK.md``.

- :mod:`.profile`: reads a profile from a SparkRing checkout and refuses drift
  between its sources;
- :mod:`.sitefile`: the ring harness's site file plus the launcher's per-Spark
  ``model_path`` and ``sudo`` keys;
- :mod:`.plan`: the launch plan (per-rank environment, command, mounts,
  labels and the ``docker run`` command) and the prefill estimate;
- :mod:`.staging`: the package tree with its generated dist-info;
- :mod:`.probe`: what serving will import, checked inside the serving image;
- :mod:`.commands`: the shell command of every remote action;
- :mod:`.checks`: known-answer prompts and receipt evaluation;
- :mod:`.cli`: the operator commands;
- :mod:`.bundle`: SIRCL's mounts, environment and checks for containers that
  another launcher starts.
"""
