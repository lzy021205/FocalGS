# FocalGS Dataset

This document describes the frozen dataset release prepared on 2026-10-13 and
published through the FocalGS Hugging Face dataset repository.

## Synthetic release candidate

The candidate contains seven scenes. Each scene has 49 camera viewpoints and 9
focal settings per viewpoint. Nine viewpoints, including all their focal
settings, are held out for testing.

Each public archive uses the short path `synthetic/<scene>/<scene>.tar`. The
archives were repackaged once on 2026-11-01 to remove internal documentation and
inventories while preserving the validated images, runtime JSON files, and
COLMAP sparse model.

| Scene | Train blur | Train sharp | Test sharp | Image size | Focal range (mm) | f-number | Focus distance (m) | Archive bytes | SHA256 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Tanabata | 360 | 360 | 81 | 600 x 400 | 2.5--20 | 0.6 | 7.027--7.033 | 416,450,560 | `bf6e5f474788064f27a9cebc3b80239757d0797668d7d415abb22c23c178a806` |
| Cozyroom | 360 | 360 | 81 | 600 x 400 | 2.5--20 | 2.0 | 1.839--1.848 | 311,736,320 | `524e953584d6a99febbf343dc067c70bb5530b1058e9af16c2ae07c50afad086` |
| Pool | 360 | 360 | 81 | 600 x 400 | 3--20 | 1.0 | 2.963--2.968 | 300,779,520 | `740c5c3c025b61a6c3e243c4c53b58899c9b9148ddf29863dccaa5d26bcf2e8a` |
| Factory | 360 | 360 | 81 | 600 x 400 | 7--21 | 0.5 | 3.603--3.605 | 432,424,960 | `d6440f51d3887cce6b156d59f8a2bede40c0a5a982c226d848c8e98dacdead76` |
| Building | 360 | 360 | 81 | 800 x 340 | 40--200 | 0.02 | 952.916--956.687 | 432,291,840 | `3273459df1ad6210d1fc13cb613e7f72b0a940a2546d90fe66d1f9cacd0a6297` |
| Livingroom | 360 | 360 | 81 | 800 x 450 | 20--90 | 2.0 | 6.663--6.684 | 237,086,720 | `f7969e9a7fe53212d8f86a1150f3a8cea30cb6522a37fcee549edeb636e8e377` |
| Ship | 360 | 360 | 81 | see below | 40--160 | 2.0 | 5.393--5.401 | 477,470,720 | `de5c253de4abd695ebf1bedc535f9b3b31ebaaa8a3554f6a2a8a549141ccaaf8` |

The seven uncompressed tar archives total 2,608,240,640 bytes (about 2.43 GiB).

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

Compute a downloaded archive's digest and compare it with the table above:

```bash
sha256sum synthetic/<scene>/<scene>.tar
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
