#!/usr/bin/env zsh
# Euler diagnosis job for the multi-threaded gemm numbers. On 8 cores it
#   1. checks whether other processes use our cores while we are idle,
#   2. times blocked gemm vs the pthreads OpenBLAS (threads not pinned) and
#      vs the OpenMP OpenBLAS (threads pinned like ours), n = 1024 and 2048,
#   3. samples busy share and clock of our CPUs during all of that.
# Needs ~/opt/openblas and ~/opt/openblas-omp (bench/euler_build_openblas.sh).
# Submit from the repo root:   sbatch bench/euler_gemm_diag.sh
#SBATCH -p instruction
#SBATCH -J gemm_diag
#SBATCH -t 0-00:20:00
#SBATCH -c 16
#SBATCH --mem=8G
#SBATCH --constraint=genoa
#SBATCH -o gemm_diag-%j.out -e gemm_diag-%j.err

cd $SLURM_SUBMIT_DIR
module load gcc/14.3.0
blas_flags() { echo "-I$1/include -L$1/lib -Wl,-rpath,$1/lib -lopenblas"; }

echo "== node"; hostname; date
scontrol show node $SLURMD_NODENAME | grep -i -E 'cpualloc|features'
python3 -c "import os; print('CPUs in this job:', sorted(os.sched_getaffinity(0)))"

echo "== our CPUs while this job is idle (busy should be ~0)"
python3 bench/cpu_monitor.py --duration 10 --interval 1

MON=gemm_diag-$SLURM_JOB_ID-monitor.txt
python3 bench/cpu_monitor.py --duration 1200 --interval 2 > $MON &
MON_PID=$!

echo "== pthreads OpenBLAS (threads not pinned)"
python3 bench/run_gemm_bench.py --sizes 1024 2048 --threads 1 4 8 --naive-max 0 \
  --blas-flags "$(blas_flags $HOME/opt/openblas)"

echo "== OpenMP OpenBLAS (threads pinned like the blocked gemm)"
python3 bench/run_gemm_bench.py --sizes 1024 2048 --threads 1 4 8 --naive-max 0 \
  --blas-flags "$(blas_flags $HOME/opt/openblas-omp)" --openblas-openmp

kill $MON_PID
echo "== busy share and clock of our CPUs during the runs (every 2 s)"
cat $MON
echo "== done"; date
