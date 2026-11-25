# Euler baseline team handoff

Updated November 25, 2026. E4 **129538**, E5 **129548**, and E6 **130120** are complete. The final report uses E6 as its primary single-socket result, E4 as the cross-socket comparison and E5 for diagnostic profiles. The user authorized committing and pushing the `baselines-euler` branch; no merge into master is authorized.

## Results

| Job | Configurations | Measured solves / warm-ups | Outcome |
|---|---|---|---|
| E4 129538 | 98 | 11,798 / 98 | All gates passed; 42 configurations capped below one timed second |
| E5 129548 | 8 | 8 / 8 | All gates passed; diagnostic profiling only |
| E6 130120 | 182 | 22,569 / 182 | All gates passed; 75 configurations capped below one timed second |

E6 ran for 579 seconds on `euler01.engr.wisc.edu`. Its 48 allocated logical CPUs represented 24 physical cores, 12 per socket. Selection chose socket 0 / NUMA node 0 and CPUs **3,4,6,13,16,17,18,19**, with nested 1/2/4/8-core subsets. All 210 rank-placement records passed. Numactl was unavailable: the policy is pinned first-touch, not verified resident-page locality.

At n=4096, OpenMP achieved approximately **7.73x** speedup and **96.6–96.7%** efficiency; SLEPc-8 medians were **34.939 / 34.693 ms**. **PySCF-8 instability persists:** n=2048 medians were **98.731 / 14.776 ms**, a **6.68x** sweep difference. Both sweeps remain visible. SLEPc vector-equivalent matvec counts/time-per-matvec remain unknown. See the generated report tables for all values and caveats.

The numerical sources, compiler records, library versions, aligned settings and original input/reference fingerprints agree between E4 and E6. Placement-only harness changes are recorded. Separate shared-node jobs do not isolate a causal NUMA effect.

## Deliverables and reproduction

- [report.md](report.md): final environment, settings, timing/scaling, E4 comparison, counts, accuracy, figures and limitations.
- [summary.md](summary.md): one-page main table and five slide takeaways.
- [details.md](details.md) and [e4-details.md](e4-details.md): complete rows with CSV lines, sample counts, min/max, accuracy and profile links.
- [report-audit.json](report-audit.json): audited counts, input fingerprints, archive hashes and generator hashes.
- [figures/](figures/): five PNG/PDF pairs and `figure-data.json` with plotted values/configuration IDs.
- [design.md](design.md): experimental choices and official API references.
- [Retained evidence](../../results/euler/README.md): compact data layout and provenance limits.

From the repository root, with Python 3.10+ and matplotlib:

```bash
python3 -B analysis/baseline_report.py --e6 results/euler/E6_130120 \
    --e4 results/euler/E4_129538 --e5 results/euler/E5_129548
```

This audits saved observations and regenerates documentation/figures without Euler access or solver execution. Small linked evidence is retained directly; original raw samples and repeated EPS views are stored losslessly in each job's `evidence.zip`. Generated matrix dumps, compiled executables, duplicate transfer archives, local environments/editor settings and intermediate E0–E3 outputs are excluded from Git. Input lengths and SHA-256 were checked during packaging; archive-only analysis compares those recorded fingerprints rather than claiming to rehash missing matrices. The two small E1 package/version inventories remain as installation provenance.

After staging, run `python3 -B analysis/check_report_links.py`. It checks links in the report/summary against the exact Git index, rejects excluded artifacts and files of 50 MiB or more, and reports the total staged content size. Full report regeneration from an isolated export of the staged tree passed for E4/E5/E6, without input dumps, uncompressed raw samples or external EPS-view files. The report, audit and figures were refreshed from that export.

## Validation and preserved decisions

The full E4/E5/E6 audits passed, including raw/CSV consistency, strict numerical gates, repeat policy, worker/rank masks and topology. All 312 plotted values independently matched CSV; all figures passed visual review. Packaging verifies exact member hashes before publication; compact-reader tests reject corrupt archives and inconsistent omitted-input fingerprints. Solver algorithms and existing laptop/check behavior are unchanged.

All Euler builds use absolute `/usr/bin/g++` (recorded GCC 14.3.1), without compiler module load/unload/purge. The old gcc/14.3.0 module failed C++ headers on the new platform; gnu15/15.2.0 is a documented fallback, not automatic. Runtime inspection verifies shared OpenBLAS and MPI rather than assuming package selectors guarantee alignment. Missing hardware/software values say **not recorded**.

Historical setup/smoke waypoints were E0 128609, E1 adoption, E2 129073 (earlier 128770 failed), and E3 129445 (earlier 129413 failed). Their intermediate result logs are excluded from the team snapshot. The setup/runtime code remains available under `euler/`; no installation or new cluster run is required to regenerate the report.
