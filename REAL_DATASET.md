# FocalGS Real-Scene Data

The public real-data release is version
`FocalGS-Real-PostPaper-Strict-24-85-v3`. Each scene is a separate download with
the same short naming convention as common 3DGS datasets:

| Scene | Download | SHA256 | Bytes | Train/test | Image size |
| --- | --- | --- | ---: | ---: | --- |
| Toy | `real/toy/toy.tar.gz` | `b4f676b8cbfc4f6f494fdd24919684f4c98907e4c05615069d85f910f676a1c6` | 46,593,274 | 60 / 15 | 800 x 533 |
| Plant | `real/plant/plant.tar.gz` | `d7ebfe603cf7653fb38e11d7b15e7ba5726a1ff5cde6fe727f8715ed4977fa43` | 43,554,352 | 58 / 14 | 800 x 534 |
| Bear | `real/bear/bear.tar.gz` | `47ca2a673a1f8cd88060bec68611c82d8fdff82e126ef7e38071e2902f72d746` | 40,799,891 | 57 / 13 | 800 x 534 |

All three scenes use the focal settings 24, 35, 50, 70, and 85 mm.

Toy was normalized to the same five-focal release protocol by removing only
the 105 mm frames from its existing train/test split. No image was resized,
reassigned, synthesized, or re-rendered. Unlike Plant and Bear, Toy does not
come from a later clean registration, so its JSON records this provenance.
This benchmark was prepared after the paper experiments and must not be
described as the original paper-experiment data.

The focal counts reflect successful registrations rather than a repaired
rectangular grid. Toy has 12/3 train/test images at each focal setting. Plant
has train counts 12/12/12/12/10 and test counts 3/3/3/3/2; Bear has train
counts 13/13/13/10/8 and test counts 3/3/3/3/1, ordered by 24/35/50/70/85 mm.

## Layout and use

Each archive extracts directly to one scene directory:

```text
<scene>/
  images/
  sparse/0/{cameras.bin,images.bin,points3D.bin}
  transforms_train.json
  transforms_test.json
```

Train with an approved real-scene configuration:

```bash
python train.py -s /path/to/<scene> -m /path/to/output --configs <config.py>
```

The three archives contain 217 JSON-referenced PNGs, six split JSON files, and
nine COLMAP sparse binary files. They contain no CR3 originals, serial numbers,
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

The three scene archives were extracted and loaded with the frozen FocalGS
reader. It returned 60/15 Toy, 58/14 Plant, and 57/13 Bear train/test cameras
and successfully decoded all three sparse point clouds.
