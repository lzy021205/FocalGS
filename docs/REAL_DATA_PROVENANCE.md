# Real-Scene Version Provenance

Updated: 2026-11-01

The public real-data release is the unified post-paper strict 24--85 mm
benchmark. It is not the dataset used for the paper experiments. The historical
experiment evidence is retained here so that the two protocols are not
confused.

## Public benchmark

Version: `FocalGS-Real-PostPaper-Strict-24-85-v3`

| Scene | Public-source lineage | Train/test | Included focals (mm) | Resolution |
| --- | --- | ---: | --- | --- |
| Toy | historical Toy archive, filtered without changing the split | 60 / 15 | 24, 35, 50, 70, 85 | 800 x 533 |
| Plant | later clean strict split | 58 / 14 | 24, 35, 50, 70, 85 | 800 x 534 |
| Bear | later clean strict split | 57 / 13 | 24, 35, 50, 70, 85 | 800 x 534 |

No later clean Toy registration was found. To give Toy the same public focal
set, the release retains the 24, 35, 50, 70, and 85 mm frames from its
historical split and excludes its 105 mm frames. This produces 12 train and 3
test images at each focal setting. No image, pose, or train/test assignment was
changed. The Toy JSON explicitly records this different source lineage.

Plant and Bear retain the later clean strict splits. Their focal grids are
uneven because only successfully registered images are included. Plant has
58/14 train/test images; Bear has 57/13.

The public files are `real/toy/toy.tar.gz`, `real/plant/plant.tar.gz`, and
`real/bear/bear.tar.gz`. Together they have 217 PNGs, six split JSON files, and
nine sparse COLMAP binaries. They have no symbolic links, raw CR3 files,
embedded image metadata, databases, pickle files, caches, or unrelated
intermediates. All three archives were extracted and loaded with the frozen
FocalGS reader. Exact sizes and SHA256 values are in `REAL_DATASET.md`.

## Historical paper-experiment evidence

No archived real-scene version exactly implements the paper statement of 15
viewpoints at five focal lengths with two held-out viewpoints for each scene.
The strongest recoverable pre-paper lineages are:

| Scene | Historical source | Registered/used | Actual split | Focal evidence |
| --- | --- | ---: | ---: | --- |
| Toy | `toy_final_resize` | 90 | 72 / 18 | six settings: 24, 35, 50, 70, 85, 105 mm |
| Plant | `plant_new` | 76 | 71 / 5 | uneven registered subset including 24--105 mm and one 87 mm entry |
| Bear | `bear_newest` | 68 | 64 / 4 | uneven registered subset including 24--105 mm |

Scripts, logs, configuration snapshots, results, and 30,000-iteration
checkpoints associate those versions with the recoverable experiments. The
public strict benchmark must not be described as a replacement or exact
reconstruction of this experiment archive. The historical archive is not part
of the current public download.

## Camera model and distortion

Plant and Bear use per-image COLMAP `OPENCV` cameras. Their strict JSON preserves
`fl_x`, `fl_y`, `cx`, `cy`, `k1`, `k2`, `p1`, and `p2`. Toy preserves the
fields available in its historical JSON and sparse model. The frozen FocalGS
runtime builds a centered pinhole projection from `camera_angle_x` and actual
image dimensions. It does not apply principal-point offsets or OPENCV
distortion coefficients.

## Privacy and release boundary

Published PNGs have no embedded EXIF or PNG metadata. Raw Plant/Bear captures
contain capture timestamps and camera/lens serial information, so CR3 files are
excluded. Public JSON removes raw filenames, timestamps, exposure/ISO strings,
lens strings, serial-related metadata, and EXIF focus bounds. The release
contains sanitized images and camera data only.
