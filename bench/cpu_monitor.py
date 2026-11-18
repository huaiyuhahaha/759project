"""Sample the CPUs this job may use: how busy they are and how fast they clock.

Every --interval seconds for --duration seconds, prints one line with the
busy share (from /proc/stat) and the current clock ("cpu MHz" in
/proc/cpuinfo) over the CPUs in this process's affinity mask: average, min
and max across CPUs. Used in Euler jobs to check
  - whether other processes run on our cores while we are idle (busy > 0),
  - how far the clock drops when many cores run AVX-512 code.

Usage: python3 bench/cpu_monitor.py [--duration 10] [--interval 1]
"""

import argparse
import os
import time

parser = argparse.ArgumentParser(description="Sample busy share and clock of this job's CPUs.")
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--interval", type=float, default=1.0)
args = parser.parse_args()

CPUS = sorted(os.sched_getaffinity(0))


def cpu_times():
    """cpu -> (busy jiffies, total jiffies)"""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            name, *vals = line.split()
            cpu = int(name[3:])
            v = [int(x) for x in vals]
            idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
            total = sum(v[:8])                         # exclude guest time (already in user)
            out[cpu] = (total - idle, total)
    return out


def cpu_mhz():
    """cpu -> MHz"""
    out, cpu = {}, None
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("processor"):
                cpu = int(line.split(":")[1])
            elif line.startswith("cpu MHz") and cpu is not None:
                out[cpu] = float(line.split(":")[1])
    return out


def stats(values):
    return (sum(values) / len(values), min(values), max(values)) if values else (0.0, 0.0, 0.0)


print(f"cpu_monitor: {len(CPUS)} CPUs {CPUS}; busy % and MHz as avg/min/max over them", flush=True)
start = time.time()
prev = cpu_times()
while time.time() - start < args.duration:
    time.sleep(args.interval)
    cur = cpu_times()
    busy = []
    for c in CPUS:
        if c in cur and c in prev and cur[c][1] > prev[c][1]:
            busy.append(100.0 * (cur[c][0] - prev[c][0]) / (cur[c][1] - prev[c][1]))
    prev = cur
    mhz_all = cpu_mhz()
    mhz = [mhz_all[c] for c in CPUS if c in mhz_all]
    b, m = stats(busy), stats(mhz)
    print(f"t={time.time() - start:6.1f}s  busy% {b[0]:5.1f} / {b[1]:5.1f} / {b[2]:5.1f}   "
          f"MHz {m[0]:6.0f} / {m[1]:6.0f} / {m[2]:6.0f}", flush=True)
