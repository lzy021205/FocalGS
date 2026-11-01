# FocalGS Real-Scene Data

The public real-data release is the **Post-paper Strict 24--85 mm Benchmark**.
Each scene is a separate download with the same short naming convention as
common 3DGS datasets:

| Scene | Download | SHA256 | Bytes | Train/test | Image size |
| --- | --- | --- | ---: | ---: | --- |
| Toy | `real/toy/toy.tar.gz` | `0936000b69a4f44728521b6f39aa1a8c13858667c7114f62c28a7b1e4829f44e` | 46,592,545 | 60 / 15 | 800 x 533 |
| Plant | `real/plant/plant.tar.gz` | `a3e459b04983da8671e31e0302221acf40842a7080512643f55cdd8e6b5454db` | 43,545,496 | 58 / 14 | 800 x 534 |
| Bear | `real/bear/bear.tar.gz` | `35afe36b11b45fea3af91e89f9bf5a2cc1461a8344d02409914052f9fcdfb19b` | 40,792,230 | 57 / 13 | 800 x 534 |

All three scenes use the focal settings 24, 35, 50, 70, and 85 mm.

Toy was normalized to the same five-focal release protocol by removing only
the 105 mm frames from its existing train/test split. No image was resized,
reassigned, synthesized, or re-rendered. Unlike Plant and Bear, Toy does not
come from a later clean registration. This benchmark was prepared after the
paper experiments and must not be described as the original paper-experiment
data.

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

Plant and Bear retain their registered COLMAP `OPENCV` cameras, including
principal points and distortion coefficients, in `sparse/0/cameras.bin`. The
portable runtime JSON files contain only the fields consumed by FocalGS:
`file_path`, `transform_matrix`, `camera_angle_x`, `focal_length_mm`, `f_stop`,
and `focal_dist_m`. The frozen loader builds a centered pinhole projection from
`camera_angle_x` and the decoded image size, so it does not apply the sparse
model's principal-point offsets or distortion coefficients.

## Verification

Compute a downloaded archive's digest and compare it with the table above:

```bash
sha256sum real/<scene>/<scene>.tar.gz
```

The three scene archives were extracted and loaded with the frozen FocalGS
reader. It returned 60/15 Toy, 58/14 Plant, and 57/13 Bear train/test cameras
and successfully decoded all three sparse point clouds.
