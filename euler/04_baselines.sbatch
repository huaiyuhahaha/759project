#!/usr/bin/env bash
# Two complete sweeps in one allocation; submit from the repository root.
# E3 129445 passed. See docs/baselines/design.md for resource estimates.
#SBATCH --partition=instruction
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=baselines_E4
#SBATCH --output=results/euler/E4_%j.slurm.out
#SBATCH --error=results/euler/E4_%j.slurm.err
set -euo pipefail

cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash euler/run_baseline_job.sh E4 \
    --sizes 64 128 256 512 1024 2048 4096 --cores 1 8 \
    --solvers cpp_serial cpp_openmp pyscf slepc --sweeps 2 \
    --min-time 1 --min-runs 5 --max-runs 200 --timeout 300
