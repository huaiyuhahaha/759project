# Retained Euler baseline evidence

| Job | Role | Readable results | Original sample evidence |
|---|---|---|---|
| E4 129538 | Cross-socket comparison, two sweeps, cores 1/8 | [CSV](E4_129538/summary.csv) | [evidence.zip](E4_129538/evidence.zip), [manifest](E4_129538/package.json) |
| E5 129548 | Diagnostic profiles; excluded from solve-time comparisons | [CSV](E5_129548/summary.csv) | [evidence.zip](E5_129548/evidence.zip), [manifest](E5_129548/package.json) |
| E6 130120 | Primary single-socket scaling, two sweeps, cores 1/2/4/8 | [CSV](E6_130120/summary.csv) | [evidence.zip](E6_130120/evidence.zip), [manifest](E6_130120/package.json) |

Small per-configuration summaries, launch/placement records, profiles, reference values, hardware/software JSON, compiler identity, lscpu and Slurm logs remain ordinary files so report links work on GitHub. The two small E1 package/version inventories are retained as installation provenance; setup logs and intermediate discovery/smoke outputs are excluded.

Each `evidence.zip` losslessly preserves the original raw timing records and repeated EPS views. The packaging manifest records each member's original length/SHA-256 and the archive hash. `analysis/baseline_storage.py` verifies and reads archive members directly; extraction is unnecessary. Timing observations are retained because rerunning a solver cannot reproduce past measurements.

The Git snapshot excludes generated `input_*.bin`, uncompressed raw logs/sample duplicates, compiled executables, transfer `*.tar.gz`, `.euler/`, and `.vscode/`. Input byte lengths and hashes were checked against the original matrices during packaging. Archive-only regeneration uses those recorded input fingerprints, not a fresh hash of missing bytes; small reference binaries remain included. Ignored local originals may still exist on the originating workstation.

From the repository root, with Python 3.10+ and matplotlib:

```bash
python3 -B analysis/baseline_report.py --e6 results/euler/E6_130120 \
    --e4 results/euler/E4_129538 --e5 results/euler/E5_129548
```

After staging a change, check report links and file sizes against the exact Git index:

```bash
python3 -B analysis/check_report_links.py
```

To package a future returned job, use `analysis/package_baselines.py --help`. Packaging writes new evidence/manifest files and never deletes the original inputs. The benchmark scripts in `euler/` remain available for reproducing the experiment; report regeneration needs no Euler access or environment installation.
