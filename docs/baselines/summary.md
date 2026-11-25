# Euler Davidson baselines — slide summary

**E6 job 130120:** four lowest eigenpairs, n=64–4096; nested 1/2/4/8 physical cores on socket 0 / NUMA 0. Numactl unavailable: pinned first-touch. **Median solve times below are ms, sweep 1 / sweep 2.**

| n | C++ serial-1 | C++ OpenMP-1 | C++ OpenMP-8 | PySCF-1 | PySCF-8 | SLEPc-1 | SLEPc-8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.659 / 0.665 | 0.704 / 0.709 | 0.621 / 0.612 | 3.486 / 3.488 | 3.471 / 3.527 | 3.047 / 3.160 | 7.785 / 7.510 |
| 128 | 0.985 / 0.986 | 1.066 / 1.061 | 0.499 / 0.474 | 3.832 / 3.837 | 3.815 / 3.791 | 3.181 / 3.193 | 7.974 / 7.718 |
| 256 | 3.296 / 3.292 | 3.467 / 3.436 | 0.874 / 0.841 | 3.865 / 3.864 | 3.928 / 3.891 | 3.199 / 3.221 | 7.340 / 7.221 |
| 512 | 11.869 / 11.813 | 12.206 / 12.131 | 2.096 / 2.057 | 4.879 / 4.901 | 21.843 / 21.944 | 4.427 / 4.271 | 7.504 / 7.344 |
| 1024 | 40.881 / 40.589 | 41.489 / 41.248 | 5.969 / 5.928 | 8.563 / 8.251 | 33.775 / 44.841 | 8.617 / 8.335 | 8.700 / 8.394 |
| 2048 | 150.651 / 149.700 | 152.086 / 151.112 | 20.092 / 19.895 | 29.784 / 28.026 | 98.731 / 14.776 | 32.910 / 33.148 | 12.142 / 11.856 |
| 4096 | 602.368 / 600.619 | 607.819 / 606.175 | 78.574 / 78.404 | 110.488 / 109.567 | 168.270 / 132.147 | 123.021 / 125.758 | 34.939 / 34.693 |

- **All 182 configurations passed:** 22,569 measured solves; 75 configurations reached the 200-call cap below one timed second. [Accuracy/repeats](report.md#accuracy).
- **OpenMP, n=4096:** 7.736 / 7.731× speedup and 96.69 / 96.64% efficiency at eight cores. [Scaling](report.md#scaling).
- **SLEPc-8 is fastest at n=4096** in these settings: 34.939 / 34.693 ms; 3.521 / 3.625× speedup. Matvec-normalized SLEPc timing remains unverified. [Counts](report.md#counts).
- **PySCF instability persists:** n=2048 PySCF-8 differs 6.68× between sweeps; at n=1024 and 4096 eight threads are slower than one in both sweeps. [Main table](report.md#main-times).
- **E4→E6 at n=4096:** OpenMP-8 and SLEPc-8 improved; PySCF-8 worsened. Separate shared-node jobs do not isolate a causal NUMA effect. [Comparison](report.md#cross-socket).

[Full report, setup and caveats](report.md) · [E6 per-row evidence](details.md) · [E4 comparison evidence](e4-details.md) · [PNG/PDF figures](report.md#figures).
