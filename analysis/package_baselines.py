#!/usr/bin/env python3
"""Package final evidence losslessly; never delete source artifacts.

Creates evidence.zip/package.json in each named job, plus an optional removal
plan for a separate reviewed cleanup. Input matrix dumps and executables are
omitted; their verified fingerprints remain in package.json.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXECUTABLES = {"build/davidson_bench", "build/cpp_baselines_bench", "build/compiler_probe",
               "build/cpp", "build/dumper"}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def keep_plain(relative):
    # Keep readable summaries, commands, placement, hardware and diagnostic
    # profiles directly linkable. Only repetitive per-call data is packed.
    return not (relative == "raw.jsonl" or relative.endswith("/samples.jsonl")
                or (relative.startswith("configs/") and "/views/" in relative
                    and not relative.endswith("/petsc-log.txt")))


def validate_package(directory, metadata, verify_sources=True):
    archive = directory / metadata["archive"]["path"]
    if fingerprint(archive) != {key: metadata["archive"][key] for key in ("sha256", "size")}:
        raise ValueError(f"Archive fingerprint mismatch: {archive}")
    with zipfile.ZipFile(archive) as packed:
        if set(packed.namelist()) != set(metadata["members"]):
            raise ValueError("Archive member inventory mismatch")
        for name, expected in metadata["members"].items():
            digest = hashlib.sha256()
            total = 0
            with packed.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    total += len(chunk)
            if {"sha256": digest.hexdigest(), "size": total} != expected:
                raise ValueError(f"Archive member content mismatch: {name}")
    if verify_sources:
        for category in ("members", "retained_files", "verified_omitted_inputs", "omitted_executables"):
            for name, expected in metadata[category].items():
                path = directory / name
                if path.exists() and fingerprint(path) != expected:
                    raise ValueError(f"Source changed during packaging: {name}")


def package_job(directory):
    directory = directory.resolve()
    directory.relative_to(ROOT / "results/euler")
    if not re.fullmatch(r"E[456]_\d+", directory.name):
        raise ValueError(f"Only final E4/E5/E6 job directories may be packaged: {directory}")
    archive, destination = directory / "evidence.zip", directory / "package.json"
    if archive.exists() or destination.exists():
        if not archive.is_file() or not destination.is_file():
            raise ValueError(f"Incomplete existing package; review manually: {directory}")
        metadata = json.loads(destination.read_text(encoding="utf-8"))
        validate_package(directory, metadata)
        return metadata
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    metadata = {"schema_version": 1, "job_id": directory.name.split("_", 1)[1],
                "step": directory.name.split("_", 1)[0], "members": {}, "retained_files": {},
                "verified_omitted_inputs": {}, "omitted_executables": {},
                "notes": ["Archived files preserve their original bytes and paths.",
                          "Input dump SHA256 and sizes were verified during packaging; omitted input bytes cannot be rehashed later.",
                          "Compiled executables are reproducible build products and are omitted.",
                          "This packaging command never deletes source files."]}
    for path in files:
        relative = path.relative_to(directory).as_posix()
        if re.fullmatch(r"input_n\d+\.bin", relative):
            category = "verified_omitted_inputs"
        elif relative in EXECUTABLES:
            category = "omitted_executables"
        elif keep_plain(relative):
            category = "retained_files"
        else:
            category = "members"
        metadata[category][relative] = fingerprint(path)
    # Input fingerprints must agree with the pre-existing independent reference
    # metadata; do not merely record the bytes of a corrupted matrix.
    for name, info in metadata["verified_omitted_inputs"].items():
        n = int(re.fullmatch(r"input_n(\d+)\.bin", name)[1])
        reference = json.loads((directory / f"reference_n{n}.json").read_text())
        if reference["input_sha256"] != info["sha256"] or info["size"] != 8 * (n * n + 4 * n):
            raise ValueError(f"Input fingerprint contradicts reference evidence: {name}")
    # Temporary files stay outside the job inventory and are moved only after
    # every source/member has been verified. No archive extraction is needed.
    with tempfile.TemporaryDirectory(prefix="baseline-package-") as temporary:
        temp = Path(temporary)
        pending_zip = temp / "evidence.zip"
        with zipfile.ZipFile(pending_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as packed:
            for relative in metadata["members"]:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info._compresslevel = 9
                with (directory / relative).open("rb") as source, packed.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        metadata["archive"] = {"path": "evidence.zip", **fingerprint(pending_zip)}
        # Validate the temporary ZIP against all recorded member hashes before
        # publishing it. Then validate source files again in the real directory.
        validate_package(temp, metadata, verify_sources=False)
        for category in ("members", "retained_files", "verified_omitted_inputs", "omitted_executables"):
            for relative, expected in metadata[category].items():
                if fingerprint(directory / relative) != expected:
                    raise ValueError(f"Source changed during packaging: {relative}")
        pending_archive = directory / ".evidence.zip.pending"
        pending_manifest = directory / ".package.json.pending"
        with pending_archive.open("xb") as output, pending_zip.open("rb") as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if fingerprint(pending_archive) != {key: metadata["archive"][key] for key in ("sha256", "size")}:
            raise ValueError("Copied archive changed before publication")
        with pending_manifest.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")
        pending_archive.rename(archive)
        pending_manifest.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", type=Path, required=True)
    parser.add_argument("--removal-plan", type=Path,
                        help="write exact repository-relative paths safe to remove after archive verification")
    args = parser.parse_args()
    jobs, removed = [], []
    for directory in args.jobs:
        directory = directory.resolve()
        metadata = package_job(directory)
        relative_root = directory.relative_to(ROOT).as_posix()
        for category in ("members", "verified_omitted_inputs", "omitted_executables"):
            removed.extend(f"{relative_root}/{name}" for name in metadata[category])
        jobs.append({"directory": relative_root, "archive_bytes": metadata["archive"]["size"],
                     "archive_members": len(metadata["members"]), "plain_files": len(metadata["retained_files"]),
                     "omitted_input_bytes": sum(v["size"] for v in metadata["verified_omitted_inputs"].values())})
        print(f"Verified {relative_root}: {len(metadata['members'])} archived artifacts, "
              f"{metadata['archive']['size'] / 1024**2:.2f} MiB ZIP", flush=True)
    plan = {"schema_version": 1, "jobs": jobs, "removed_paths": sorted(removed),
            "instruction": "Packaging verified these paths. This script has not deleted any source artifacts."}
    if args.removal_plan:
        with args.removal_plan.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(plan, output, indent=2)
            output.write("\n")
    print(json.dumps({"jobs": jobs, "removable_paths": len(removed),
                      "removal_plan": str(args.removal_plan) if args.removal_plan else None}))


if __name__ == "__main__":
    main()
