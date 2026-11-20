#!/usr/bin/env zsh
# Euler probe job for the gemm work: reports the node's CPU (AVX-512?),
# available compiler/BLAS modules, runs the gemm correctness checks, and
# takes a quick 1024^3 timing (including which core type OpenBLAS picks).
# Submit from the repo root:   sbatch bench/euler_gemm_probe.sh
#SBATCH -p instruction
#SBATCH -J gemm_probe
#SBATCH -t 0-00:15:00
#SBATCH -c 8
#SBATCH -o gemm_probe-%j.out -e gemm_probe-%j.err

cd $SLURM_SUBMIT_DIR

echo "== node";  hostname; date
echo "== lscpu"; lscpu
echo "== AVX-512 flags"; grep -o 'avx512[a-z0-9_]*' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "== CPUs in this job: $(nproc)"

echo "== modules (gcc, BLAS)"
module avail gcc 2>&1
module avail 2>&1 | grep -i -E 'blas|mkl|lapack|python'
module load gcc/13.2.0
module list 2>&1
g++ --version | head -1
python3 --version
echo "== system OpenBLAS"
ls /usr/include/cblas*.h /usr/include/openblas* /usr/lib64/libopenblas* /usr/lib/x86_64-linux-gnu/libopenblas* 2>/dev/null

echo "== gemm correctness checks"
python3 checks/run_gemm_check.py

echo "== quick timing, n = 1024"
python3 bench/run_gemm_bench.py --sizes 1024 --threads 1 8 --naive-max 0 --runs 3 \
  || python3 bench/run_gemm_bench.py --sizes 1024 --threads 1 8 --naive-max 0 --runs 3 --no-openblas
echo "== done"; date
