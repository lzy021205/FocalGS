# FocalGS Real-Scene Data

This document describes the two offline real-data release candidates prepared
on 2026-10-13. They are separate datasets and must remain separate downloads.

## Paper-experiment archival candidate

Archive: `focalgs_real_paper_experiment_candidate.tar.gz`

SHA256:
`f739ce473f1d167c715daa1b64232f0905f089f5936028d4b21ac3abca7933fa`

| Scene | Recoverable historical source | Train/test | Focal settings (mm) | Image size |
| --- | --- | ---: | --- | --- |
| Toy | `toy_final_resize` | 72 / 18 | 24, 35, 50, 70, 85, 105 | 800 x 533 |
| Plant | `plant_new` | 71 / 5 | 24, 35, 50, 70, 85, 87, 105 | 800 x 533 |
| Bear | `bear_newest` | 64 / 4 | 24, 35, 50, 70, 85, 105 | 800 x 533 |

These versions are linked to the available final-run scripts, logs, output
configuration, and 30,000-iteration checkpoints. They are the most reliable
recoverable versions used by the pre-paper experiments. They are not an exact
implementation of the paper prose that describes 15 viewpoints, five focal
lengths, and two held-out viewpoints. See
`docs/REAL_DATA_PROVENANCE.md` and
`docs/REAL_PROTOCOL_DISCLOSURE_DRAFT.md` before publishing results.

## Optional post-paper benchmark

Archive: `focalgs_real_postpaper_strict_24_85.tar.gz`

SHA256:
`1d12603d2616625373379ea7755bf446a48f3f3a6c72b602b0fe66cd15a53b88`

| Scene | Train/test | Focal settings (mm) | Label |
| --- | ---: | --- | --- |
| Plant | 58 / 14 | 24, 35, 50, 70, 85 | post-paper strict benchmark |
| Bear | 57 / 13 | 24, 35, 50, 70, 85 | post-paper strict benchmark |

This benchmark was derived after the paper experiments from later clean data.
It must not replace or be described as the paper-experiment dataset. There is
no corresponding strict Toy dataset.

## Layout and use

After extraction, each scene contains:

```text
<scene>/
  images/
  sparse/0/{cameras.bin,images.bin,points3D.bin}
  transforms_train.json
  transforms_test.json
```

Train with the same entrypoint used for synthetic data, selecting an approved
real-scene configuration:

```bash
python train.py -s /path/to/<scene> -m /path/to/output --configs <config.py>
```

The archives contain only JSON-referenced PNGs and the sparse binary triplet.
They contain no CR3 originals, camera-body or lens serial numbers, embedded
image metadata, databases, pickle files, caches, or symbolic links. The strict
JSON files also omit original filenames, timestamps, exposure/ISO strings, lens
strings, and raw EXIF focus bounds.

## Camera model

Plant and Bear were registered with per-image COLMAP `OPENCV` cameras. The
candidate preserves focal lengths and principal points; the strict candidate
also preserves `k1`, `k2`, `p1`, and `p2`. The frozen FocalGS loader constructs
a centered pinhole projection from `camera_angle_x` and the decoded image size,
so it does not apply the stored principal-point offsets or distortion
coefficients. Published results and downstream users must be told this.

## Verification

```bash
python scripts/release/verify_release.py /path/to/SHA256SUMS
python scripts/release/verify_release.py \
  /path/to/focalgs_real_paper_experiment_candidate.tar.gz \
  /path/to/focalgs_real_postpaper_strict_24_85.tar.gz
```

Downloads remain disabled until patent, capture-rights, privacy, and coauthor
protocol review are approved.
