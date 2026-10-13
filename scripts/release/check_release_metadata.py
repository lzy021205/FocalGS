#!/usr/bin/env python3
"""Check frozen release metadata, companion checksums, and project-page links."""

from __future__ import annotations

import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
AUTHORS = ["Zeyu Liu", "Chao Wang", "Jiahao Wu", "Jie Liang", "Ronggang Wang"]
TITLE = "FocalGS: Adaptive Focal 3D Gaussian Splatting for Continuous Zoom Rendering"
VENUE = "SIGGRAPH Asia 2026 Conference Papers"
DOI = "10.1145/3829340.3842186"


class LocalReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src", "data-full", "poster"} and value:
                self.references.add(value)


def read_sums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            result[Path(name.strip()).name] = digest
    return result


def require_text(path: Path, values: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for value in values:
        if value not in text:
            raise RuntimeError(f"Missing expected metadata in {path}: {value}")


def main() -> None:
    cff = ROOT / "CITATION.cff"
    readme = ROOT / "README.md"
    page = ROOT / "project_page/index.html"
    required = [TITLE, VENUE, DOI, *AUTHORS]
    for path in (cff, readme, page):
        require_text(path, required)

    forbidden = ("RELEASE BLOCKED", "author list pending", "TAPS list is still required")
    for path in (cff, readme, page):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker.lower() in text.lower():
                raise RuntimeError(f"Stale metadata marker in {path}: {marker}")

    master_sums = read_sums(ROOT / "release/SHA256SUMS")
    companion_dirs = (
        ROOT / "release/real/experimental_archive",
        ROOT / "release/real/postpaper_clean_benchmark",
    )
    for directory in companion_dirs:
        stats = json.loads((directory / "DATASET_STATS.json").read_text(encoding="utf-8"))
        version = (directory / "VERSION").read_text(encoding="utf-8").strip()
        if stats["version"] != version:
            raise RuntimeError(f"Version mismatch in {directory}")
        companion_sums = read_sums(directory / "SHA256SUMS")
        if companion_sums.get(stats["archive"]) != stats["sha256"]:
            raise RuntimeError(f"Companion checksum mismatch in {directory}")
        if master_sums.get(stats["archive"]) != stats["sha256"]:
            raise RuntimeError(f"Master checksum mismatch for {stats['archive']}")

    json.loads((ROOT / "release/PUBLISH_MANIFEST.json").read_text(encoding="utf-8"))

    parser = LocalReferenceParser()
    parser.feed(page.read_text(encoding="utf-8"))
    for reference in sorted(parser.references):
        parsed = urlsplit(reference)
        if parsed.scheme or reference.startswith("#"):
            continue
        target = page.parent / parsed.path
        if not target.exists():
            raise FileNotFoundError(f"Broken project-page reference: {reference}")

    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "github/main",
            "release/github-integration-20261013",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "diff", "--quiet", "main", "release/github-integration-20261013"],
        cwd=ROOT,
        check=True,
    )
    print(
        "OK release metadata, companion checksums, project-page links, "
        "Git ancestry, and integration-tree equality"
    )


if __name__ == "__main__":
    main()
