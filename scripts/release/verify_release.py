#!/usr/bin/env python3
"""Verify release checksums, archives, dataset references, and symlink safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sums(sum_file: Path) -> None:
    for line in sum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        path = sum_file.parent / name.strip()
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")
        print(f"OK sha256 {path}")


def check_archive(path: Path) -> None:
    with tarfile.open(path, "r:*") as tar:
        names = set()
        json_members = []
        for member in tar.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive contains a link: {member.name}")
            names.add(member.name)
            if pure.name in {"transforms_train.json", "transforms_test.json"}:
                json_members.append(member)
        for member in json_members:
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read {member.name}")
            data = json.load(stream)
            root = PurePosixPath(member.name).parent
            for frame in data.get("frames", []):
                image = root / frame["file_path"]
                candidates = {image.as_posix()}
                if not image.suffix:
                    candidates |= {(root / f"{frame['file_path']}{suffix}").as_posix() for suffix in (".png", ".jpg", ".jpeg")}
                if not candidates & names:
                    raise RuntimeError(f"Missing image referenced by {member.name}: {frame['file_path']}")
        print(f"OK archive {path}: {len(names)} members, {len(json_members)} split JSON files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if path.name == "SHA256SUMS":
            check_sums(path)
        else:
            check_archive(path)


if __name__ == "__main__":
    main()
