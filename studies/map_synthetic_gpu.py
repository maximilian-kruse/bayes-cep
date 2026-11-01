"""MAP estimate from a synthetic ground truth, on a GPU.

The runs of `map_synthetic` (same configurations, so the same run ids), executed on the GPU
partition of the cluster with one GPU per run. The forward solves and the derivative partials of
the eikonal map run on the GPU, which needs a CUDA build of jax: the executor installs the `gpu`
pixi environment before submitting, and the tasks run in it. The rest (SPDE prior, sparse adjoint
solves, optimizer) runs on the CPU cores of the task. Create and run the `preprocessing_synthetic`
study first. The results can be compared with those of `map_synthetic`.
"""

from dataclasses import replace

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import StudySetup
from studies.map_synthetic import STUDY as CPU_STUDY

# The GPU partition; its time limit (`scontrol show partition gpu_a100_short`) must be at least
# `time_min` below, or the jobs are rejected.
GPU_PARTITION = "gpu_a100_short"

STUDY: StudySetup = replace(
    CPU_STUDY,
    name="map_synthetic_gpu",
    description=(f"{CPU_STUDY.description} Executed on a GPU, one per run (see `map_synthetic`)."),
    root=module_relative_path(__file__, "../working_data/map_synthetic_gpu"),
    executor=ExecutorSettings(
        cluster="slurm",
        time_min=60,
        cpus_per_task=4,
        mem_gb=8,
        partition=GPU_PARTITION,
        gpus_per_task=1,
        pixi_environment="gpu",
        max_parallel=4,
    ),
)
