#!/usr/bin/env zsh
# Build OpenBLAS (no LAPACK/Fortran) on an Euler compute node; Euler has no
# system OpenBLAS. Clone the source on the login node first (compute nodes
# may have no internet):
#   git clone --depth 1 --branch v0.3.28 https://github.com/OpenMathLib/OpenBLAS.git ~/opt/src/OpenBLAS
# then, from the repo root:
#   sbatch bench/euler_build_openblas.sh        # pthreads build -> ~/opt/openblas
#   sbatch bench/euler_build_openblas.sh omp    # OpenMP build   -> ~/opt/openblas-omp
# The OpenMP build's threads are OpenMP threads, so bench/run_gemm_bench.py
# --openblas-openmp can pin them exactly like the blocked gemm's threads.
# DYNAMIC_ARCH=1 picks the kernels at run time, so the library works on any
# node type; bench/run_gemm_bench.py prints the core type and build string.
#SBATCH -p instruction
#SBATCH -J openblas_build
#SBATCH -t 0-00:45:00
#SBATCH -c 16
#SBATCH --mem=8G
#SBATCH --constraint=genoa
#SBATCH -o openblas_build-%j.out -e openblas_build-%j.err

set -e
if [[ "$1" == "omp" ]]; then
  USE_OPENMP=1; PREFIX=$HOME/opt/openblas-omp
else
  USE_OPENMP=0; PREFIX=$HOME/opt/openblas
fi
module load gcc/14.3.0
hostname; lscpu | grep 'Model name'; g++ --version | head -1
echo "USE_OPENMP=$USE_OPENMP PREFIX=$PREFIX"

cd $HOME/opt/src/OpenBLAS
make clean
make -j16 DYNAMIC_ARCH=1 NO_LAPACK=1 NOFORTRAN=1 USE_OPENMP=$USE_OPENMP NUM_THREADS=128
make PREFIX=$PREFIX install
make clean   # the object files of all DYNAMIC_ARCH targets take a lot of space

ls $PREFIX/include $PREFIX/lib
echo "OpenBLAS build OK"
