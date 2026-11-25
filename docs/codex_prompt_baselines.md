# Codex prompt: PySCF and SLEPc baselines on Euler

> **How to use this file.** Start Codex in the repo root and paste everything below the line.
> Codex works in stages and stops after each stage. It tells you what to run on Euler
> and what to bring back. Put the Euler outputs under `results/euler/` and reply
> "Euler step E<k> done, outputs in <path>".

---

## Role and goal

You are helping a 5-person team with the ECE 759 final project in this repo, a C++17 Davidson eigensolver. Read `README.md`, `docs/progress-report.md` and `docs/task-plan.md` first.

**Goal of this task.** Measure two external Davidson baselines on **Euler** (UW-Madison's Slurm cluster), on the same hardware and the same inputs:

1. **PySCF** `pyscf.lib.linalg_helper.davidson1`, which already has a runner in `bench/pyscf_davidson_bench.py`;
2. **SLEPc** `EPSGD` (Generalized Davidson), through petsc4py/slepc4py. It is new; nobody on the team has used it before.

Sweep the matrix size over **n = 64, 128, 256, 512, 1024, 2048, 4096**.

The end product is **`docs/baselines/report.md`**. Its main table gives the solve time for each solver × n × core count, with iteration counts and accuracy next to it.

Also run our own C++ serial and C++ OpenMP solvers in the same jobs, as reference columns. They cost almost nothing, and we will need them in the same table later.

**Out of scope:** CUDA, new solver features, and any change to the Davidson algorithm in `davidson.cpp`. Do not change the behavior of the existing `bench/run_davidson_bench.py` or `checks/`; add new files next to them.

## How we work together (read carefully)

- **You cannot reach Euler. I run everything there.** Split the work into **stages**. At the end of each stage, **stop** and print a section with exactly this shape:
  ```
  ## Run on Euler (step E<k>)
  1. <copy-paste command>   # where (login node or sbatch), expected wall time
  ...
  ## Bring back
  - <files/paths>, plus one tar command that packs them
  ## What happens next
  - <one or two lines>
  ```
- **Never invent Euler results or Euler facts.** If you need to know something about Euler (modules, conda, quotas, internet access on compute nodes, the MPI launcher), write a probe script, don't guess.
- **Keep `docs/baselines/STATUS.md` up to date.** It records the current stage, the decisions and why, the open issues, and the Euler steps done so far, with their job IDs. Assume a later Codex session will read only this file to continue.
- **Branch and commits.** Work on a new local branch `baselines-euler`. Don't commit or push unless I ask.
- **Local testing.** If a local Linux or WSL environment is available (`wsl -d Ubuntu --cd "<repo>" -- ...`), smoke-test locally at n = 64 and 128 before handing a stage to me. Label local numbers clearly as local; they never go in the report.
- **Research.** Read the official docs before you write SLEPc code: the SLEPc users manual, the manual pages for the EPS/GD options you use, and the petsc4py/slepc4py API. Cite the pages you relied on in `docs/baselines/design.md`.

## What already exists (reuse it)

- **`bench/davidson_bench.cpp`:**
  - `davidson_bench <n> dump <file>` writes the inputs as raw float64:
    - A (n×n, row-major); A is the seeded, strictly diagonally dominant matrix (seed 759, A_ii = i+1, A_ij = u/n);
    - then the guess (n×4, row-major), from `initialize_vector(4, A)`;
  - `davidson_bench <n> serial|openmp` times our solver and prints one JSON line;
  - `input_checksum()` is implemented identically in C++ and in `bench/pyscf_davidson_bench.py`. Every runner must print it, and the driver must check it.
- **`bench/pyscf_davidson_bench.py`:** PySCF `davidson1` on the dumped inputs, already aligned with our solver:
  - `nroots = 4`, `tol_residual = 1e-8`, `max_cycle = 100`;
  - the same guarded diagonal preconditioner as our solver;
  - `lindep = 1e-16`. **Keep this.** With the default 1e-14, PySCF cannot reach a residual of 1e-8; see the README.

  It prints one JSON line and recomputes the residuals and the orthogonality error.
- **`bench/run_davidson_bench.py`:** the current laptop driver (thread env, checksums, `eigvalsh` reference). Use it as a model.
- **The Euler conventions used by a teammate** are on branch `origin/gemm-blocked-omp`. Read them with `git show origin/gemm-blocked-omp:<path>`:
  - `bench/euler_gemm_bench.sh`, `bench/euler_build_openblas.sh`, `bench/euler_gemm_probe.sh`, `bench/euler_gemm_diag.sh`;
  - the "Results on Euler" and "OpenBLAS caveats" sections of that branch's `README.md`.

  What we know so far:
  - Slurm partition `-p instruction`, with `--constraint=genoa`: AMD EPYC 9254 (Zen 4), 2 sockets × 24 cores.
  - Slurm hands out CPUs as hyper-thread pairs, so `-c 16` gives 8 physical cores. Run one thread or rank per physical core.
  - `module load gcc/14.3.0` exists.
  - Compute nodes may have no internet; download on the login node.
  - **Pinning pitfalls with OpenBLAS:**
    - With `OMP_PROC_BIND` set, libgomp pins the main thread, and pthreads OpenBLAS workers inherit that mask, which collapses performance.
    - Unpinned pthreads OpenBLAS was about 2× slower on Euler.
    - Restrict BLAS-threaded runs to their cores with `sched_setaffinity`/`taskset`, not `OMP_PROC_BIND`.
    - Record which OpenBLAS threading variant each environment actually uses.

## Experimental design (write it down in `docs/baselines/design.md` in Stage 1)

- **Same node and same inputs.** All solvers run inside one Slurm job on one `genoa` node. They read the same dump file, and the checksums must match. Record the hostname, `lscpu`, the job's CPU list, and every software version.
- **Same BLAS for both baselines.** Prefer one conda-forge environment in which NumPy (used by PySCF), PETSc and SLEPc all link the same OpenBLAS (pin `libblas=*=*openblas`). Install PySCF from conda-forge if available, otherwise with pip into the same env; record which. Verify the BLAS each one actually loads; don't assume.
- **Cores.** Default: `--cores 1 8`, and keep it configurable so we can add 24 later.
  - PySCF: p BLAS threads, pinned to p physical cores.
  - SLEPc: `mpiexec -n p`, one rank per physical core, BLAS threads = 1 per rank; use the binding flags of whichever MPI the env provides.
  - C++ reference: serial, plus OpenMP with p threads.
- **Solver settings to align, then document in a table.** For each setting, give our value, the PySCF value and the SLEPc value, and how each was set:
  - number of roots = 4, smallest eigenvalues, symmetric/Hermitian problem;
  - **absolute** residual tolerance ‖Ax − λx‖₂ < 1e-8 (SLEPc: `EPS_CONV_ABS`, because SLEPc's default is relative);
  - maximum iterations 100;
  - a diagonal (Jacobi) preconditioner. Explain how SLEPc's GD preconditioning with `PCJACOBI` differs from our guarded `r_i / (θ − A_ii)`;
  - the same 4 initial vectors (SLEPc: `EPSSetInitialSpace`);
  - block size, subspace size and restart.

  These settings differ between solvers. Choose reasonable values, record SLEPc's full configuration with `eps.view()` for every run, and state the remaining differences honestly. Do not tune SLEPc beyond matching our setup.
- **What is timed.** Time only the solver call, consistently across solvers: PySCF's `davidson1(...)`; for SLEPc, setup plus `EPSSolve`; our `Davidson(...)`. Exclude matrix creation and file I/O. For MPI runs, put a barrier before and after the timed region and take the maximum over ranks. Document exactly what is included.
- **Repeats.**
  - Small n takes microseconds, so 5 solves are not enough.
  - Do 1 warm-up, then repeat until the total timed time is ≥ 1 s **and** at least 5 runs are done, with a cap of 200 runs.
  - Report the median, min, max and the run count.
  - Make sure each repeat solves from the same initial state; with SLEPc, reset the initial space every time.
- **Record per run:**
  - converged, iterations, number of matvecs (SLEPc: e.g. the MatMult event count from PETSc logging), median/min/max time in ms;
  - max |λ − `eigvalsh`|, recomputed max residual, ‖XᵀX − I‖_F;
  - checksum, cores, solver settings, versions, hostname, job ID.
- **Correctness gate.** A run counts only if it converged, every eigenvalue is within 1e-8 of `eigvalsh(A)[:4]`, and every recomputed residual is < 1e-8, up to a documented small slack if a solver's internal residual differs. Report failures; do not drop them silently.
- **Output.** One raw JSON line per run, plus one combined CSV per job, under `results/euler/<step>_<jobid>/`, together with an `env.txt` that has versions, `numpy.show_config()`, `PETSc.Sys.getVersion()`, the threadpoolctl info and `lscpu`.

## Stages

### Stage 1: study and design (local only)
1. Read the repo and the gemm branch's Euler scripts. Research the SLEPc GD options listed above.
2. Write `docs/baselines/design.md`, covering:
   - the alignment table;
   - what is timed;
   - the repeat policy;
   - the CSV schema;
   - the planned Euler steps E0–E5;
   - the risks, including a fallback if SLEPc cannot be installed: run PySCF only and document why.
3. Write `docs/baselines/STATUS.md`.
4. Write **`euler/00_probe.sh`** (login node) and **`euler/00_probe_node.sbatch`** (a short compute-node job). Together they report:
   - `module avail` filtered for conda/miniforge/anaconda/mamba/python/petsc/slepc/mpi/openblas/gcc;
   - whether conda/mamba is already available, and the home/scratch quotas (`df -h ~`, the cluster's quota command if any);
   - `sinfo` for the instruction partition;
   - `lscpu` and `numactl -H` on a genoa node, and whether that node can reach conda-forge;
   - `gcc` versions.
5. **STOP:** ask me to run **E0** (the probe).

### Stage 2: environment
1. Based on the probe output, write:
   - `euler/env/environment.yml` (conda-forge: python, numpy, scipy, openblas-backed libblas, petsc, petsc4py, slepc, slepc4py, an MPI, threadpoolctl, pyscf);
   - `euler/01_setup_env.sh`, run on the login node. If no conda exists, it installs Miniforge into a directory with enough quota, then creates the env. It must be idempotent and print the versions at the end.
2. Write `euler/02_check_env.sbatch`, a ~2-minute job that:
   - imports everything and prints versions and BLAS info;
   - runs a tiny `EPSGD` solve on a 100×100 dense symmetric matrix, on 1 and 2 ranks, and checks it against `eigvalsh`;
   - runs a tiny PySCF `davidson1`;
   - builds `bench/davidson_bench.cpp` with `gcc/14.3.0` and runs `dump` for n = 64.
3. **STOP:** ask me to run **E1** (env setup) and **E2** (env check).

### Stage 3: benchmark code
1. **`bench/slepc_davidson_bench.py`**:
   - mirrors `pyscf_davidson_bench.py` (same CLI style, same checksum, same JSON fields, plus the SLEPc-specific ones);
   - builds a PETSc dense matrix distributed by rows when it runs on several ranks;
   - applies the aligned settings and the repeat policy;
   - saves `eps.view()` output, and has an optional `--log-view` flag for later profiling.
2. **Update `bench/pyscf_davidson_bench.py`** to the same repeat policy. Add a flag that defaults to the old behavior, so `run_davidson_bench.py` is unchanged.
3. **`bench/run_baselines.py`**, the new driver. For each n it:
   - dumps the inputs and computes the `eigvalsh` reference;
   - runs C++ serial and C++ OpenMP-p, PySCF-p and SLEPc-p for every p in `--cores`, with the thread/rank/pinning env described above;
   - checks the checksums and the correctness gate;
   - writes the JSON lines and the CSV.

   Options: `--sizes`, `--cores`, `--solvers`, `--min-time`, `--max-runs`.
4. Smoke-test locally if you can.
5. Write `euler/03_smoke.sbatch`: n = 64, 128 and 256, cores 1 and 2, at most 3 runs, all solvers. It checks the whole pipeline in under 10 minutes.
6. **STOP:** ask me to run **E3** (smoke).

### Stage 4: full run
1. Fix whatever the smoke run showed.
2. Write `euler/04_baselines.sbatch`:
   - n = 64 … 4096, cores 1 and 8;
   - run the whole sweep **twice** in the same job, to show the run-to-run spread;
   - request enough cores and wall time (estimate it from the smoke results), with `--constraint=genoa`.
3. Write `euler/05_profile.sbatch` (optional, for our later diagnosis step):
   - SLEPc with `-log_view` at n = 1024 and 4096, on 1 and 8 ranks;
   - PySCF under `cProfile` at the same sizes.
4. **STOP:** ask me to run **E4**, and optionally **E5**.

### Stage 5: analysis and report
1. Write `analysis/baseline_report.py`. It reads the result CSVs and produces markdown tables, and optionally plots. Create the tables with the script; don't type them in by hand.
2. Write **`docs/baselines/report.md`**, with these sections:
   1. **Setup:** node, CPU, core layout, software versions, BLAS, MPI, and the job IDs.
   2. **Solver settings:** the alignment table, and the remaining differences between solvers.
   3. **Table 1, median solve time in ms:**
      - one row per n;
      - columns PySCF-1, PySCF-8, SLEPc-1, SLEPc-8, plus the C++ serial and C++ OpenMP-8 references;
      - min–max or run-to-run spread shown, either in the table or in a companion table.
   4. **Table 2:** iterations and matvecs per solver and n, and time per matvec.
   5. **Table 3:** accuracy (max eigenvalue error, max residual, orthogonality); flag every failed gate.
   6. **Observations, backed by the numbers:**
      - where framework or interpreter overhead dominates (small n) and where compute dominates;
      - how 8 cores compare with 1 core at each n;
      - iteration-count differences and what explains them;
      - any anomalies.
   7. **Caveats:** a shared node, the MPI-vs-threads difference between SLEPc and the others, and the setting differences.
3. Update `STATUS.md`. Then stop, summarize the key findings in five bullets or fewer, and list the open questions.

## Quality bar
- All Euler-facing scripts:
  - use `set -euo pipefail` (or the zsh equivalent);
  - print what they are doing;
  - fail loudly;
  - never need internet on compute nodes.
- Every number in the report is traceable to a CSV row and a job ID.
- If SLEPc's GD cannot be aligned on some setting, say so in the report rather than hiding it.
- Prefer small, readable scripts over one big one, and keep the commands I copy-paste short.

Begin with Stage 1.
