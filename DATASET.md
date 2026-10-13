# FocalGS Dataset

This document describes the offline dataset release candidate prepared on
2026-10-13. Download URLs will be added only after the patent and redistribution
reviews are complete.

## Synthetic release candidate

The candidate contains seven scenes. Each scene has 49 camera viewpoints and 9
focal settings per viewpoint. Nine viewpoints, including all their focal
settings, are held out for testing.

| Scene | Train blur | Train sharp | Test sharp | Image size | Focal range (mm) | f-number | Focus distance (m) | Candidate bytes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Tanabata | 360 | 360 | 81 | 600 x 400 | 2.5--20 | 0.6 | 7.027--7.033 | 416,008,225 |
| Cozyroom | 360 | 360 | 81 | 600 x 400 | 2.5--20 | 2.0 | 1.839--1.848 | 311,293,463 |
| Pool | 360 | 360 | 81 | 600 x 400 | 3--20 | 1.0 | 2.963--2.968 | 300,322,181 |
| Factory | 360 | 360 | 81 | 600 x 400 | 7--21 | 0.5 | 3.603--3.605 | 431,981,318 |
| Building | 360 | 360 | 81 | 800 x 340 | 40--200 | 0.02 | 952.916--956.687 | 431,844,771 |
| Livingroom | 360 | 360 | 81 | 800 x 450 | 20--90 | 2.0 | 6.663--6.684 | 236,646,648 |
| Ship | 360 | 360 | 81 | see below | 40--160 | 2.0 | 5.393--5.401 | 477,027,150 |

The uncompressed candidate payload is about 2.43 GiB. Sizes include the
per-scene manifests but not the final archive container overhead.

Ship has 800 x 384 defocused training images and 800 x 395 sharp train/test
images. This is intentional and has not been resized. FocalGS reads each
image's actual dimensions and derives vertical FOV from the JSON horizontal
FOV and aspect ratio. See `docs/IMPLEMENTATION_NOTES.md` for the camera-model
implication.

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
  FILELIST.tsv
  MANIFEST.sha256
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

After extraction, verify a scene from its root:

```bash
sha256sum --check MANIFEST.sha256
```

The release root also contains `SYNTHETIC_PAYLOAD.sha256`, `SCENE_SIZES.tsv`,
and `VALIDATION_REPORT.json`. The offline validation checks every JSON path,
every 4 x 4 finite pose matrix, every PNG decode, the sparse binary triplet,
and the absence of published symbolic links.

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

Toy, Plant, and Bear are not included in the synthetic candidate. Their
recoverable paper-experiment versions, later clean datasets, and strict
24--85 mm validation splits are distinct datasets. The version evidence and
camera limitations are recorded in `docs/REAL_DATA_PROVENANCE.md`.

## Redistribution status

The candidate is technically staged but is not cleared for public upload.
Scene-asset provenance and redistribution terms are not yet complete. In
particular, Tanabata, Cozyroom, Pool, and Factory appear to derive from scene
assets also used by Deblur-NeRF, while the exact source and license chain for
all seven rendered scenes has not been established from the local archive.
No ownership or redistribution grant is asserted here. See
`docs/SYNTHETIC_DATA_PROVENANCE.md` and `docs/RELEASE_CHECKLIST.md` before
publishing any archive.
