#!/usr/bin/env python3
"""Build the unified post-paper strict 24--85 mm real-data release.

Toy is filtered from the historical Toy split; Plant and Bear come from the
later clean strict splits. The script copies only JSON-referenced images and
the COLMAP sparse binary triplet. It never edits the research source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT = Path("release_staging/real_scene_archives")
ALLOWED_FOCALS = {24.0, 35.0, 50.0, 70.0, 85.0}

SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
PUBLIC_FRAME_FIELDS = {
    "file_path",
    "transform_matrix",
    "camera_angle_x",
    "focal_length_mm",
    "f_stop",
    "focal_dist_m",
}
PUBLIC_TOP_FIELDS = {"camera_angle_x"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_path(source: Path, file_path: str) -> Path:
    path = source / file_path
    if path.exists():
        return path
    if not path.suffix:
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"JSON reference does not resolve: {source} / {file_path}")


def load_splits(source: Path) -> dict[str, dict]:
    result = {}
    for name in ("transforms_train.json", "transforms_test.json"):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def sanitize_strict(data: dict) -> dict:
    public = {key: value for key, value in data.items() if key in PUBLIC_TOP_FIELDS}
    public["frames"] = [
        {key: value for key, value in frame.items() if key in PUBLIC_FRAME_FIELDS}
        for frame in data.get("frames", [])
        if float(frame["focal_length_mm"]) in ALLOWED_FOCALS
    ]
    return public


def copy_scene(source: Path, destination: Path, *, scene: str, source_note: str) -> dict:
    splits = load_splits(source)
    destination.mkdir(parents=True)
    refs: dict[str, Path] = {}
    split_counts = {}
    focal_counts = {}

    for json_name, data in splits.items():
        split_name = json_name.removeprefix("transforms_").removesuffix(".json")
        out_data = sanitize_strict(data)
        frames = out_data.get("frames", [])
        split_counts[split_name] = len(frames)
        focal_counts[split_name] = dict(
            sorted(
                Counter(str(int(float(frame["focal_length_mm"]))) for frame in frames).items()
            )
        )
        for frame in frames:
            source_image = image_path(source, frame["file_path"])
            relative = Path(frame["file_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe image reference: {relative}")
            refs[relative.as_posix()] = source_image
        (destination / json_name).write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for relative, source_image in sorted(refs.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target, follow_symlinks=True)

    sparse_out = destination / "sparse/0"
    sparse_out.mkdir(parents=True)
    for name in SPARSE_FILES:
        source_file = source / "sparse/0" / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        shutil.copy2(source_file, sparse_out / name, follow_symlinks=True)

    return {
        "scene": scene,
        "source_note": source_note,
        "splits": split_counts,
        "focal_counts": focal_counts,
        "unique_images": len(refs),
    }


def make_archive(source: Path, archive: Path) -> None:
    tar_command = [
        "tar",
        "--sort=name",
        "--mtime=UTC 2026-11-01",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-C",
        str(source.parent),
        "-cf",
        "-",
        source.name,
    ]
    with archive.open("wb") as output:
        tar_process = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
        assert tar_process.stdout is not None
        gzip_process = subprocess.Popen(
            ["gzip", "-n"], stdin=tar_process.stdout, stdout=output
        )
        tar_process.stdout.close()
        gzip_status = gzip_process.wait()
        tar_status = tar_process.wait()
    if tar_status or gzip_status:
        raise subprocess.CalledProcessError(
            tar_status or gzip_status, tar_command if tar_status else ["gzip", "-n"]
        )


def verify_tree(root: Path) -> None:
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise RuntimeError(f"Candidate contains symbolic links: {links[:3]}")
    for scene in root.rglob("transforms_train.json"):
        scene_root = scene.parent
        for json_path in (scene_root / "transforms_train.json", scene_root / "transforms_test.json"):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for frame in data.get("frames", []):
                image_path(scene_root, frame["file_path"])
        for name in SPARSE_FILES:
            if not (scene_root / "sparse/0" / name).is_file():
                raise FileNotFoundError(scene_root / "sparse/0" / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--research-root",
        type=Path,
        default=Path("../deblurrDyGS"),
        help="read-only historical research tree",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    research_root = args.research_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {output}")

    release_root = output / "payload"
    summary = {}
    strict_base = (
        research_root
        / "data/real_clean_baseline_20260820/focalgs_ready/real_valid_split/datasets"
    )
    sources = {
        "toy": research_root / "data/toy_final_resize",
        "plant": strict_base / "plant_strict_valid_24_85",
        "bear": strict_base / "bear_strict_valid_24_85",
    }
    source_notes = {
        "toy": (
            "Derived from the historical Toy experimental archive by retaining "
            "the existing split and the 24, 35, 50, 70, and 85 mm frames; this "
            "is not a later clean Toy capture."
        ),
        "plant": (
            "Copied from the later clean Plant strict validation split prepared "
            "after the paper experiments."
        ),
        "bear": (
            "Copied from the later clean Bear strict validation split prepared "
            "after the paper experiments."
        ),
    }

    for scene, source in sources.items():
        summary[scene] = copy_scene(
            source,
            release_root / scene,
            scene=scene,
            source_note=source_notes[scene],
        )

    (output / "BUILD_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    verify_tree(release_root)

    archives = []
    for scene in ("toy", "plant", "bear"):
        archive = output / f"{scene}.tar.gz"
        make_archive(release_root / scene, archive)
        archives.append(archive)
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(archive)}  {archive.name}\n" for archive in archives),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    for archive in archives:
        print(f"{sha256(archive)}  {archive}  {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
