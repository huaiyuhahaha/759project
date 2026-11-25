"""Read ordinary or packaged baseline artifacts without extracting archives.

Large input matrices are reproducible intermediates. For packaged results their
size/SHA256 remains packaging-time provenance, not a new check of omitted bytes.
"""

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile


_PACKAGES = {}
_ARCHIVES = {}
_PACKAGE_ROOTS = set()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package(directory):
    path = Path(directory) / "package.json"
    if not path.is_file():
        return None
    stat = path.stat()
    key = (str(path.absolute()), stat.st_size, stat.st_mtime_ns)
    if key not in _PACKAGES:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value.get("archive", {}).get("path") != "evidence.zip":
            raise ValueError(f"Unsupported result package: {path}")
        _PACKAGES[key] = value
        _PACKAGE_ROOTS.add(Path(directory).absolute())
    return _PACKAGES[key]


def archive_for(path):
    """Return an archive member even when an ignored original also exists."""
    path = Path(os.path.abspath(path))
    known = [root for root in _PACKAGE_ROOTS if path.is_relative_to(root)]
    candidates = sorted(known, key=lambda root: len(root.parts), reverse=True) or path.parents
    for directory in candidates:
        package = _package(directory)
        if package is not None:
            member = path.relative_to(directory).as_posix()
            if member in package["members"]:
                return directory / package["archive"]["path"], member
            return None
    return None


def _archive(path):
    path = Path(path)
    stat = path.stat()
    package = _package(path.parent)
    if package is None:
        raise ValueError(f"Archive lacks package manifest: {path}")
    expected = package["archive"]
    package_stat = (path.parent / "package.json").stat()
    key = (str(path.absolute()), stat.st_size, stat.st_mtime_ns, expected["sha256"],
           package_stat.st_size, package_stat.st_mtime_ns)
    if key not in _ARCHIVES:
        if stat.st_size != expected["size"] or _file_hash(path) != expected["sha256"]:
            raise ValueError(f"Archive SHA256/size contradicts package manifest: {path}")
        handle = zipfile.ZipFile(path, "r")
        if set(handle.namelist()) != set(package["members"]):
            handle.close()
            raise ValueError(f"Archive member coverage contradicts package manifest: {path}")
        for info in handle.infolist():
            original = PurePosixPath(info.filename)
            if original.is_absolute() or ".." in original.parts or info.is_dir():
                handle.close()
                raise ValueError(f"Unsafe/non-file member in result archive: {info.filename}")
            if info.file_size != package["members"][info.filename]["size"]:
                handle.close()
                raise ValueError(f"Archive member length contradicts manifest: {info.filename}")
        _ARCHIVES[key] = handle
    return _ARCHIVES[key]


@contextmanager
def open_binary(path):
    path = Path(path)
    location = archive_for(path)
    if location is None and path.is_file():
        with path.open("rb") as stream:
            yield stream
        return
    if location is None:
        raise FileNotFoundError(path)
    archive, member = location
    with _archive(archive).open(member) as stream:
        yield stream


@contextmanager
def open_text(path, encoding="utf-8", errors="strict", newline=None):
    with open_binary(path) as binary:
        with io.TextIOWrapper(binary, encoding=encoding, errors=errors, newline=newline) as stream:
            yield stream


def read_bytes(path):
    with open_binary(path) as stream:
        return stream.read()


def read_text(path, encoding="utf-8", errors="strict"):
    with open_text(path, encoding=encoding, errors=errors) as stream:
        return stream.read()


def exists(path):
    return Path(path).exists() or archive_for(path) is not None


def size(path):
    path = Path(path)
    location = archive_for(path)
    if location is None and path.is_file():
        return path.stat().st_size
    if location is None:
        raise FileNotFoundError(path)
    archive, member = location
    return _archive(archive).getinfo(member).file_size


def sha256(path):
    digest = hashlib.sha256()
    with open_binary(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(directory, n):
    filename = f"input_n{n}.bin"
    if not re.fullmatch(r"input_n\d+\.bin", filename):
        raise ValueError("Invalid matrix dimension for input fingerprint")
    directory = Path(directory)
    path = directory / filename
    package = _package(directory)
    if package is None and path.is_file():
        return {"sha256": sha256(path), "size": path.stat().st_size, "validation": "bytes"}
    if package is None or filename not in package.get("verified_omitted_inputs", {}):
        raise FileNotFoundError(path)
    # Reading the package also verifies that the associated retained evidence is
    # intact, but this cannot rehash matrix bytes which have been removed.
    _archive(directory / package["archive"]["path"])
    fingerprint = package["verified_omitted_inputs"][filename]
    return {"sha256": fingerprint["sha256"], "size": fingerprint["size"],
            "validation": "packaging_manifest"}
