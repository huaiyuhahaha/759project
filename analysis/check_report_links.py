#!/usr/bin/env python3
"""Check report links and size limits against Git's staged tree, not local files.

Run after git add: python3 -B analysis/check_report_links.py
The output reports uncompressed staged-file content, not a network pack size.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), *args])


def anchors(text):
    result = set(re.findall(r'<a\s+(?:id|name)=[\"\']([^\"\']+)', text))
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-file-mib", type=float, default=50)
    parser.add_argument("--docs", nargs="+", default=["docs/baselines/report.md", "docs/baselines/summary.md"])
    args = parser.parse_args()
    entries = {}
    for item in git("ls-files", "--stage", "-z").decode().split("\0"):
        if not item:
            continue
        metadata, path = item.split("\t", 1)
        mode, oid, stage = metadata.split()
        if stage != "0":
            raise ValueError(f"unmerged index entry: {path}")
        entries[path] = (oid, mode)
    changed = set(git("diff", "--cached", "--name-only", "-z").decode().split("\0")) - {""}
    identifiers = sorted({oid for oid, _ in entries.values()})
    proc = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
                           "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                          input="\n".join(identifiers) + "\n", text=True, capture_output=True, check=True)
    sizes = {line.split()[0]: int(line.split()[2]) for line in proc.stdout.splitlines()}
    errors, checked, external = [], 0, 0
    for path, (oid, _) in entries.items():
        if sizes[oid] >= args.max_file_mib * 1024**2:
            errors.append(f"file reaches/exceeds {args.max_file_mib:g} MiB: {path}")
        if (path.startswith((".euler/", ".vscode/")) or
                re.match(r"results/(?:.*/)?(?:input_[^/]*\.bin|raw\.jsonl|[^/]*\.tar\.gz)$", path)):
            errors.append(f"excluded artifact is staged: {path}")
    if "matrix_optimize_4.cpp" in changed:
        errors.append("matrix_optimize_4.cpp must not be part of this commit")
    contents = {}
    def read(path):
        if path not in contents:
            contents[path] = git("cat-file", "blob", entries[path][0]).decode("utf-8")
        return contents[path]
    for document in args.docs:
        if document not in entries:
            errors.append(f"document is not staged: {document}")
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", read(document)):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc:
                external += 1
                continue
            path = unquote(parsed.path)
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(document), path)) if path else document
            if resolved.startswith("../") or resolved.startswith("/"):
                errors.append(f"{document}: link escapes repository: {target}")
                continue
            if resolved not in entries:
                errors.append(f"{document}: target is absent from staged tree: {target}")
                continue
            if parsed.fragment and PurePosixPath(resolved).suffix == ".md":
                if unquote(parsed.fragment) not in anchors(read(resolved)):
                    errors.append(f"{document}: missing staged Markdown anchor: {target}")
                    continue
            checked += 1
    payload = sum(sizes[entries[path][0]] for path in changed if path in entries)
    result = {"ok": not errors, "checked_local_links": checked, "external_links_not_fetched": external,
              "changed_files": len(changed), "staged_content_bytes": payload,
              "staged_content_mib": round(payload / 1024**2, 3),
              "snapshot_files": len(entries), "snapshot_content_mib": round(sum(sizes[oid] for oid, _ in entries.values()) / 1024**2, 3),
              "largest_file": max(entries, key=lambda path: sizes[entries[path][0]]),
              "largest_file_mib": round(max(sizes.values()) / 1024**2, 3), "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Staged-tree check failed: {error}", file=sys.stderr)
        sys.exit(2)
