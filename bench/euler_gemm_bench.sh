#!/usr/bin/env zsh
# Euler gemm benchmark: correctness checks, then the full naive / loop /
# huaiyu / blocked / OpenBLAS comparison, run twice to show the run-to-run
# spread, on up to 24 of the 48 cores of an EPYC 9254 node (Slurm may spread
# them over both sockets; the job prints its CPU list). Slurm hands out CPUs
# as hyper-thread pairs, so -c 48 gives 24 physical cores; the benchmark runs
# one thread per physical core. --constraint=genoa keeps every run on the
# same node type.
#
# OpenBLAS: the OpenMP build (~/opt/openblas-omp, from
# `sbatch bench/euler_build_openblas.sh omp`) with --openblas-openmp, so its
# threads are pinned exactly like the blocked gemm's. The pthreads build does
# not pin its threads, and on these nodes it fell to ~200 GFLOP/s on 8 cores
# (vs ~400 pinned) although no other process used our cores
# (bench/euler_gemm_diag.sh).
#
# The AVX-512 default blocking came from
#   python3 bench/run_gemm_bench.py --sweep --threads 1 24 --blas-flags "$BLAS" --openblas-openmp
# Submit from the repo root:   sbatch bench/euler_gemm_bench.sh
#SBATCH -p instruction
#SBATCH -J gemm_bench
#SBATCH -t 0-00:45:00
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH --constraint=genoa
#SBATCH -o gemm_bench-%j.out -e gemm_bench-%j.err

cd $SLURM_SUBMIT_DIR
module load gcc/14.3.0
OB=$HOME/opt/openblas-omp
BLAS="-I$OB/include -L$OB/lib -Wl,-rpath,$OB/lib -lopenblas"

echo "== node"; hostname; date
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA'
scontrol show node $SLURMD_NODENAME | grep -i -E 'cpualloc|features'
python3 -c "import os; print('CPUs in this job:', sorted(os.sched_getaffinity(0)))"
g++ --version | head -1

echo "== gemm correctness checks"
python3 checks/run_gemm_check.py

for rep in 1 2; do
  echo "== full benchmark, repetition $rep"
  python3 bench/run_gemm_bench.py --sizes 512 1024 2048 4096 --threads 1 2 4 8 16 24 \
    --thin 1 4 16 --blas-flags "$BLAS" --openblas-openmp
done
echo "== done"; date
