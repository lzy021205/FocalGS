# FocalGS Real-Scene Data

The public real-data release contains one benchmark:
`FocalGS-Real-PostPaper-Strict-24-85-v2`.

Archive: `focalgs_real_postpaper_strict_24_85_v2.tar.gz`

SHA256:
`5d73b83e8409dabc948bde03be3d38d72b2fcea18787c4be8c391d2f89ff96a1`

Download directory: `real/postpaper_strict_24_85/`

## Actual contents

| Scene | Source used for this release | Train/test | Focal settings (mm) | Image size |
| --- | --- | ---: | --- | --- |
| Toy | strict subset derived from the historical Toy experimental archive | 60 / 15 | 24, 35, 50, 70, 85 | 800 x 533 |
| Plant | later clean strict split | 58 / 14 | 24, 35, 50, 70, 85 | 800 x 534 |
| Bear | later clean strict split | 57 / 13 | 24, 35, 50, 70, 85 | 800 x 534 |

Toy was normalized to the same five-focal release protocol by removing only
the 105 mm frames from its existing train/test split. No image was resized,
reassigned, synthesized, or re-rendered. Unlike Plant and Bear, Toy does not
come from a later clean registration, so the release labels this provenance
inside both JSON files. This unified benchmark was prepared after the paper
experiments and must not be described as the original paper-experiment data.

The focal counts reflect successful registrations rather than a repaired
rectangular grid. Toy has 12/3 train/test images at each focal setting. Plant
has train counts 12/12/12/12/10 and test counts 3/3/3/3/2; Bear has train
counts 13/13/13/10/8 and test counts 3/3/3/3/1, ordered by 24/35/50/70/85 mm.

## Layout and use

After extraction, each scene contains:

```text
focalgs_real_postpaper_strict_24_85_v2/
  toy|plant|bear/
    images/
    sparse/0/{cameras.bin,images.bin,points3D.bin}
    transforms_train.json
    transforms_test.json
  MANIFEST.json
  SHA256SUMS
```

Train with an approved real-scene configuration:

```bash
python train.py -s /path/to/<scene> -m /path/to/output --configs <config.py>
```

The archive contains 217 JSON-referenced PNGs, six split JSON files, and nine
COLMAP sparse binary files. It contains no CR3 originals, serial numbers,
embedded image metadata, databases, pickle files, caches, or symbolic links.

## Camera model

Plant and Bear use per-image COLMAP `OPENCV` camera records; their public JSON
preserves focal lengths, principal points, and `k1`, `k2`, `p1`, and `p2`.
Toy preserves the camera fields available in its historical JSON and sparse
model. The frozen FocalGS loader builds a centered pinhole projection from
`camera_angle_x` and the decoded image size, so it does not apply stored
principal-point offsets or distortion coefficients.

## Verification

From the downloaded dataset-repository root:

```bash
python verify_downloads.py .
```

The release was also extracted and loaded scene by scene with the frozen
FocalGS reader. It returned 60/15 Toy, 58/14 Plant, and 57/13 Bear train/test
cameras and successfully decoded all three sparse point clouds.
