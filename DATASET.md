# FocalGS Dataset

This document describes the frozen dataset release prepared on 2026-10-13 and
published through the FocalGS Hugging Face dataset repository.

## Synthetic release candidate

The candidate contains seven scenes. Each scene has 49 camera viewpoints and 9
focal settings per viewpoint. Nine viewpoints, including all their focal
settings, are held out for testing.

Each public archive uses the short path `synthetic/<scene>/<scene>.tar`. The
seven frozen archive byte streams were not rebuilt when these names were
shortened, so their existing SHA256 values remain unchanged.

| Scene | Train blur | Train sharp | Test sharp | Image size | Focal range (mm) | f-number | Focus distance (m) | Candidate bytes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Tanabata | 360 | 360 | 81 | 600 x 400 | 2.5--20 | 0.6 | 7.027--7.033 | 416,008,225 |
| Cozyroom | 360 | 360 | 81 | 600 x 400 | 2.5--20 | 2.0 | 1.839--1.848 | 311,293,463 |
| Pool | 360 | 360 | 81 | 600 x 400 | 3--20 | 1.0 | 2.963--2.968 | 300,322,181 |
| Factory | 360 | 360 | 81 | 600 x 400 | 7--21 | 0.5 | 3.603--3.605 | 431,981,318 |
| Building | 360 | 360 | 81 | 800 x 340 | 40--200 | 0.02 | 952.916--956.687 | 431,844,771 |
| Livingroom | 360 | 360 | 81 | 800 x 450 | 20--90 | 2.0 | 6.663--6.684 | 236,646,648 |
| Ship | 360 | 360 | 81 | see below | 40--160 | 2.0 | 5.393--5.401 | 477,027,150 |

The uncompressed candidate payload is about 2.43 GiB. The table reports the
extracted scene payload and does not include the final archive container
overhead.

Ship has 800 x 384 defocused training images and 800 x 395 sharp train/test
images. This is intentional and has not been resized. FocalGS reads each
image's actual dimensions and derives vertical FOV from the JSON horizontal
FOV and aspect ratio. The release preserves this documented camera-model behavior.

## Scene layout

Each extracted scene has this layout:

```text
<scene>/
  images/
    <801 JSON-referenced PNG files>
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  transforms_train.json
  transforms_test.json
  transforms_train_sharp.json
```

Only JSON-referenced images are included. `database.db`, pickle files, caches,
unreferenced blurry-test intermediates, text-format COLMAP duplicates, and
generated top-level point clouds are excluded. The loader creates
`points3d.ply` from `sparse/0/points3D.bin` on first use, so the extracted scene
directory must be writable for that first load.

## Split semantics

- `transforms_train.json` references defocused training supervision.
- `transforms_test.json` references held-out all-in-focus evaluation images.
- `transforms_train_sharp.json` references all-in-focus counterparts for the
  training viewpoints. They are used for visualization and diagnostics, not
  the canonical training loss.

Every frame supplies a camera-to-world `transform_matrix`, horizontal
`camera_angle_x`, `focal_length_mm`, `f_stop`, `focal_dist_m`, and `file_path`.
The sparse COLMAP model supplies the initial point cloud.

## Integrity verification

Verify all downloaded archives from the repository root:

```bash
sha256sum --check SHA256SUMS
# or
python verify_downloads.py .
```

Before publication, offline validation checked every JSON path, every 4 x 4
finite pose matrix, every PNG decode, the sparse binary triplet, and the
absence of published symbolic links. Internal upload inventories and
validation logs are retained by the project rather than distributed as part of
the user-facing dataset.

## Training, rendering, and evaluation

Use the matching canonical scene configuration:

```bash
python train.py \
    -s /path/to/extracted/<scene> \
    -m /path/to/output/<scene> \
    --configs configs/focalgs_canonical/<scene>.py

python render.py \
    -s /path/to/extracted/<scene> \
    -m /path/to/output/<scene> \
    --iteration 30000 --skip_train --skip_video

python metrics.py -m /path/to/output/<scene> --iteration 30000
```

The rendering command produces clear test views for comparison with sharp
ground truth. Add `--render_blur` only when physical defocus output is wanted.

## Real scenes

Toy, Plant, and Bear are separate short-named downloads:
`real/toy/toy.tar.gz`, `real/plant/plant.tar.gz`, and
`real/bear/bear.tar.gz`. Together they form the post-paper strict 24--85 mm
benchmark. Toy is a five-focal subset of the historical Toy archive; Plant and
Bear use the later clean strict splits. This benchmark is not the original
paper-experiment dataset. Actual counts, provenance, and camera limitations are
recorded in `REAL_DATASET.md` and `docs/REAL_DATA_PROVENANCE.md`.

## Redistribution status

Public redistribution of the seven frozen FocalGS rendered archives has been
authorized by the project release authority. This release authorization does
not establish ownership of upstream Blender assets or assign a new license to
them. Tanabata, Cozyroom, Pool, and Factory appear to derive from scene assets
also used by Deblur-NeRF, while parts of the exact upstream asset-license chain
could not be recovered from the research archive. See
`docs/SYNTHETIC_DATA_PROVENANCE.md` for the evidence recorded for each scene.
