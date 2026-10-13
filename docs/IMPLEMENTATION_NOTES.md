# FocalGS Implementation Notes

These notes describe the frozen release implementation. They record known
differences between the submitted paper text, the checked-in code, and the
available datasets. They are documentation of the release state, not proposed
algorithm changes.

## Focal coordinate

The paper defines a linearly normalized focal coordinate:

```text
tau_paper = clip((f - f_min) / (f_max - f_min), 0, 1)
```

The trusted JSON/COLMAP loader in `scene/dataset_readers.py` instead uses a
log-normalized physical focal length:

```text
tau_code = clip((log(f) - log(f_min)) /
                (log(f_max) - log(f_min)), 0, 1)
```

The code obtains `f_min` and `f_max` from all frames in
`transforms_train.json`, `transforms_test.json`, and
`transforms_train_sharp.json`. A single-focal dataset maps to zero. The value is
stored as `Camera.time` and expanded per anchor as the fourth coordinate of the
canonical `SpaceTimeHashingField` query. Other inherited readers contain older
FOV-based conventions and are not the trusted FocalGS dataset path.

## Adaptive feature fusion

The paper describes two soft weights for the static and focal-dependent
features. The frozen implementation predicts one sigmoid gate and uses its
complement:

```text
f = w * f_residual + (1 - w) * f_static
```

In source terminology, `dynamic_mask` is `w`, `dy_feat` is the
focal-conditioned residual, and `sta_feat` is the static intrinsic feature.
This complementary gate is the release implementation and has been checked by
a finite forward/backward sanity test. The previous additive expression was a
completed stale ablation and is not a supported release mode.

## Anchor growing threshold

The paper states a focal-aware growing threshold of `0.0004`. The frozen
canonical configurations use `densify_grad_threshold = 0.0002`, matching the
mature experiment configuration recovered from training records. The release
does not silently change this value to match the prose. Reproductions should
use the scene files under `configs/focalgs_canonical/` and report the code
threshold.

## Data protocol

The synthetic release candidate matches the paper split: 49 viewpoints per
scene, with 40 train viewpoints and 9 held-out test viewpoints; each viewpoint
contains 9 focal settings. Training uses 360 defocused images, testing uses 81
sharp images, and 360 additional sharp train-view correspondences are included
for visualization and diagnostics.

The recoverable real-scene experiment directories do not exactly match the
paper statement of 15 viewpoints, 5 focal lengths per viewpoint, and 2 held-out
viewpoints. The strongest pre-paper lineage evidence is:

| Scene | Recoverable experiment data | Train/test | Observed focal values |
| --- | --- | ---: | --- |
| Toy | `toy_final_resize` | 72 / 18 | 24, 35, 50, 70, 85, 105 mm |
| Plant | `plant_new` | 71 / 5 | 24, 35, 50, 70, 85, 87, 105 mm |
| Bear | `bear_newest` | 64 / 4 | 24, 35, 50, 70, 85, 105 mm |

These are preserved as paper-experiment candidates because historical training
configuration, logs, checkpoints, and results point to them. They are not
rewritten into the protocol stated in the paper. Later clean Plant/Bear data
and the strict 24--85 mm validation splits are separate post-paper benchmarks;
Toy has no corresponding clean benchmark.

## Real-camera intrinsics and distortion

Plant and Bear were reconstructed with per-image COLMAP `OPENCV` cameras.
Their metadata includes `fl_x`, `fl_y`, `cx`, `cy`, and, in the later clean
versions, `k1`, `k2`, `p1`, and `p2`. The frozen main loader constructs a
centered pinhole camera from `camera_angle_x` and the decoded image dimensions;
it does not apply the stored principal point or distortion coefficients during
rendering. This is a known real-data calibration limitation and must be stated
with any released real benchmark results.

Toy uses the same centered-FOV runtime path. Its current 800 x 533 images were
derived from a reconstruction whose archived camera text reports 600 x 400,
which is another calibration caveat.

## Ship image dimensions

Ship intentionally has 800 x 384 defocused training images and 800 x 395 sharp
train/test images. The loader reads the actual dimensions of every image and
derives vertical FOV from the frame's horizontal FOV and that image's aspect
ratio. Cameras may therefore have different heights without breaking scene
loading or the per-view rasterizer. Evaluation compares 800 x 395 test renders
with 800 x 395 sharp ground truth. The release candidate preserves these files
unchanged; resizing would alter the camera model and is not part of the release.

## Clear and defocused rendering

Training enables the physical defocus component when `use_deblur=True`.
Evaluation defaults to clear rendering against sharp ground truth.
`render.py --render_blur` explicitly enables the physical Circle of Confusion
path for inference. These modes use the same learned focal-conditioned Gaussian
representation.
