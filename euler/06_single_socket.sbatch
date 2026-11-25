#!/usr/bin/env bash
# E4 repeat policy/settings, with nested 1/2/4/8 physical cores on one NUMA node.
# E4: 268 s / 98 configurations; E6: 182 configurations, ~498 s before margin.
# 48 logical CPUs give >=24 physical cores on the recorded SMT2/two-socket host.
# The topology gate, not this estimate, decides whether the allocation is usable.
#SBATCH --partition=instruction
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=baselines_E6
#SBATCH --output=results/euler/E6_%j.slurm.out
#SBATCH --error=results/euler/E6_%j.slurm.err
set -euo pipefail

cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash euler/run_baseline_job.sh E6 \
    --sizes 64 128 256 512 1024 2048 4096 --cores 1 2 4 8 \
    --solvers cpp_serial cpp_openmp pyscf slepc --sweeps 2 \
    --min-time 1 --min-runs 5 --max-runs 200 --timeout 300
