#!/usr/bin/env bash
# Optional diagnostic job: keep profiled timings out of the E4 report tables.
# One warm-up and one measured/profiled solve per configuration.
#SBATCH --partition=instruction
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=baselines_E5
#SBATCH --output=results/euler/E5_%j.slurm.out
#SBATCH --error=results/euler/E5_%j.slurm.err
set -euo pipefail

cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash euler/run_baseline_job.sh E5 \
    --sizes 1024 4096 --cores 1 8 --solvers pyscf slepc --sweeps 1 \
    --min-time 0 --min-runs 1 --max-runs 1 --timeout 300 \
    --log-view --profile-pyscf
