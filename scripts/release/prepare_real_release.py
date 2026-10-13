#!/usr/bin/env python3
"""Build portable real-scene release candidates from the read-only archive.

The script intentionally separates the recoverable paper-experiment datasets
from the later strict 24--85 mm benchmark. It copies only JSON-referenced
images and the COLMAP sparse binary triplet. It never edits the source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT = Path("release_staging/real_candidate_20261013")

SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
PUBLIC_FRAME_FIELDS = {
    "file_path",
    "transform_matrix",
    "camera_angle_x",
    "camera_angle_y",
    "focal_length_mm",
    "f_stop",
    "focal_dist_m",
    "width",
    "height",
    "fl_x",
    "fl_y",
    "cx",
    "cy",
    "k1",
    "k2",
    "p1",
    "p2",
    "colmap_camera_model",
    "colmap_camera_id",
    "colmap_image_id",
    "resize_scale",
    "focus_distance_source",
    "valid_split_role",
    "viewpoint_id",
}
PUBLIC_TOP_FIELDS = {
    "camera_angle_x",
    "camera_angle_y",
    "camera_model_runtime_note",
    "colmap_camera_model",
    "width",
    "height",
    "fl_x",
    "fl_y",
    "cx",
    "cy",
    "valid_split_note",
}


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


def sanitize_strict(data: dict, scene: str) -> dict:
    public = {key: value for key, value in data.items() if key in PUBLIC_TOP_FIELDS}
    public["release_dataset_label"] = "post-paper strict 24--85 mm benchmark"
    public["release_metadata_note"] = (
        "Raw filenames, camera serial metadata, capture timestamps, exposure, ISO, "
        "lens-model strings, and EXIF focus bounds were removed for release."
    )
    public["scene"] = scene
    public["frames"] = [
        {key: value for key, value in frame.items() if key in PUBLIC_FRAME_FIELDS}
        for frame in data.get("frames", [])
    ]
    return public


def copy_scene(source: Path, destination: Path, *, sanitize: bool, scene: str) -> dict:
    splits = load_splits(source)
    destination.mkdir(parents=True)
    refs: dict[str, Path] = {}
    split_counts = {}
    focal_counts = {}

    for json_name, data in splits.items():
        out_data = sanitize_strict(data, scene) if sanitize else data
        frames = out_data.get("frames", [])
        split_counts[json_name] = len(frames)
        focal_counts[json_name] = dict(
            sorted(Counter(str(frame["focal_length_mm"]) for frame in frames).items())
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
        "source_archive_name": source.name,
        "split_counts": split_counts,
        "focal_counts": focal_counts,
        "unique_referenced_images": len(refs),
    }


def file_manifest(root: Path) -> list[dict]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def write_manifest(root: Path) -> None:
    records = file_manifest(root)
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "format": "FocalGS real-data portable candidate",
                "files": records,
                "file_count": len(records),
                "total_bytes": sum(item["bytes"] for item in records),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for item in records:
            stream.write(f"{item['sha256']}  {item['path']}\n")


def make_archive(source: Path, archive: Path, arcname: str) -> None:
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        tar.add(source, arcname=arcname, recursive=True)


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

    paper_root = output / "paper_experiment"
    strict_root = output / "postpaper_strict_24_85"
    archive_root = output / "archives"
    archive_root.mkdir(parents=True)
    summary = {"paper_experiment": {}, "postpaper_strict_24_85": {}}

    paper_sources = {
        "toy": research_root / "data/toy_final_resize",
        "plant": research_root / "data/plant_new",
        "bear": research_root / "data/bear_newest",
    }
    strict_base = (
        research_root
        / "data/real_clean_baseline_20260820/focalgs_ready/real_valid_split/datasets"
    )
    strict_sources = {
        "plant": strict_base / "plant_strict_valid_24_85",
        "bear": strict_base / "bear_strict_valid_24_85",
    }

    for scene, source in paper_sources.items():
        summary["paper_experiment"][scene] = copy_scene(
            source, paper_root / scene, sanitize=False, scene=scene
        )
    for scene, source in strict_sources.items():
        summary["postpaper_strict_24_85"][scene] = copy_scene(
            source, strict_root / scene, sanitize=True, scene=scene
        )

    (output / "BUILD_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(paper_root)
    write_manifest(strict_root)
    verify_tree(paper_root)
    verify_tree(strict_root)

    archives = [
        (paper_root, archive_root / "focalgs_real_paper_experiment_candidate.tar.gz", "focalgs_real_paper_experiment"),
        (strict_root, archive_root / "focalgs_real_postpaper_strict_24_85.tar.gz", "focalgs_real_postpaper_strict_24_85"),
    ]
    for source, archive, arcname in archives:
        make_archive(source, archive, arcname)
    with (archive_root / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for _, archive, _ in archives:
            stream.write(f"{sha256(archive)}  {archive.name}\n")

    print(json.dumps(summary, indent=2))
    for _, archive, _ in archives:
        print(f"{sha256(archive)}  {archive}  {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
