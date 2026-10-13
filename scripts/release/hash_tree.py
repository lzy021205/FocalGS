#!/usr/bin/env python3
"""Compute a reproducible aggregate SHA256 for a vendored source tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", type=Path)
    args = parser.parse_args()
    root = args.tree.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    records: list[tuple[str, int, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        checksum = file_sha256(path)
        size = path.stat().st_size
        line = f"{checksum}  {relative.as_posix()}\n"
        aggregate.update(line.encode("utf-8"))
        records.append((relative.as_posix(), size, checksum))

    print(f"tree: {root}")
    print(f"files: {len(records)}")
    print(f"bytes: {sum(size for _, size, _ in records)}")
    print(f"aggregate_sha256: {aggregate.hexdigest()}")


if __name__ == "__main__":
    main()
