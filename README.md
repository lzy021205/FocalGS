# FocalGS: Adaptive Focal 3D Gaussian Splatting for Continuous Zoom Rendering

FocalGS reconstructs a static scene from multi-view, multi-focal captures and
renders continuous zoom views from a single trained 3D Gaussian representation.
The release code keeps the canonical training path focused on focal-conditioned
Gaussian attributes and physical defocus supervision.

Resources:

- Code: https://github.com/lzy021205/FocalGS
- Project page: included in `project_page/`; public URL will be activated after approval
- Paper: *SIGGRAPH Asia 2026 Conference Papers*, DOI
  [10.1145/3829340.3842186](https://doi.org/10.1145/3829340.3842186)
- Dataset: offline release candidates documented in `DATASET.md` and
  `REAL_DATASET.md`; public URLs will be inserted after approval

**Zeyu Liu, Chao Wang, Jiahao Wu, Jie Liang, Ronggang Wang**

## Overview

FocalGS uses a sparse anchor-based Gaussian representation. Each anchor stores a
static intrinsic feature and receives a focal-conditioned residual feature from a
multi-resolution hash field. A complementary adaptive fusion gate combines these
features before Gaussian attributes are decoded and rasterized.

The release supports two rendering modes from the same checkpoint:

- Clear all-in-focus rendering for evaluation against sharp ground truth.
- Physical depth-of-field rendering with the existing thin-lens Circle of
  Confusion model, enabled explicitly at inference time.

## Method Overview

The canonical implementation follows this path:

1. Camera focal conditioning
2. `SpaceTimeHashingField` residual feature query
3. Complementary adaptive fusion
4. Gaussian attribute decoder
5. Gaussian rasterization

Implementation details reflected in the release code:

- The focal condition is derived from per-frame `focal_length_mm`.
- The conditioning coordinate is log-normalized over the scene focal range.
- The static anchor feature is fused with the focal-conditioned residual feature
  as `f = w * f_residual + (1 - w) * f_static`.
- The adaptive fusion weight `w` is a sigmoid gate.
- Each anchor decodes `k = 10` local Gaussians by default.
- The decoder predicts local offset, opacity, RGB color, scaling, and rotation.
- RGB color is decoded directly by an MLP and does not use spherical harmonics
  in the canonical path.
- Physical defocus is modeled with camera focal length, aperture, and focus
  distance metadata.

## Installation

Clone the repository:

```bash
git clone https://github.com/lzy021205/FocalGS.git
cd FocalGS
```

The public baseline vendors the required CUDA extension sources under
`submodules/`, so no separate Git submodule initialization is required.

Create the conda environment. The local CUDA extensions are built during
environment creation, so point the build to a CUDA 11.8 toolkit first. If CUDA
is installed elsewhere, adjust `CUDA_HOME`.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

# Optional when GPU auto-detection is unavailable during extension builds,
# for example "8.9" on an NVIDIA L40S.
# export TORCH_CUDA_ARCH_LIST="8.9"

conda env create -f environment.yml
conda activate FocalGS
```

`environment.yml` includes conda GCC/G++ 11 compiler packages. On systems where
the system C++ compiler is missing or incompatible, set the compiler wrappers
before creating the environment:

```bash
export FOCALGS_ENV_PREFIX="$(conda info --base)/envs/FocalGS"
export CC="$FOCALGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$FOCALGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
```

Key packages and extensions in the current `environment.yml`:

- Python 3.9
- PyTorch 2.2.0 with CUDA 11.8 packages
- `numpy<2` and `opencv-python==4.11.0.86` for compatibility with the PyTorch
  extension stack
- `mmcv==1.6.0` for config file loading
- `open3d==0.19.0` for point-cloud and scene loading utilities
- `diff-gaussian-rasterization` from `submodules/diff-gaussian-rasterization`
- `simple-knn` from `submodules/simple-knn`
- `tiny-cuda-nn` PyTorch bindings from
  `submodules/tiny-cuda-nn/bindings/torch`

COLMAP is required if your dataset does not already contain a usable sparse
initialization under `sparse/0`.

The release environment and CUDA extensions require an NVIDIA GPU for training,
rendering, and the standard LPIPS evaluation path. Dataset manifests can be
verified on CPU.

## Dataset Format

The canonical synthetic data path uses JSON camera metadata plus a COLMAP sparse
initialization. A scene should be organized as:

```text
<scene>/
  images/
    <image files>
  sparse/
    0/
      cameras.bin or cameras.txt
      images.bin or images.txt
      points3D.bin or points3D.txt
  transforms_train.json
  transforms_test.json
  transforms_train_sharp.json
```

Split semantics:

- `transforms_train.json`: blurred training images used for supervision.
- `transforms_test.json`: sharp held-out images used for rendering and metrics.
- `transforms_train_sharp.json`: train-view sharp images for visualization.

Each frame entry should provide:

- `file_path`: image path or image filename.
- `transform_matrix`: camera-to-world pose.
- `camera_angle_x`: horizontal field of view in radians.
- `focal_length_mm`: lens focal length in millimeters.
- `f_stop`: f-number used to compute aperture diameter.
- `focal_dist_m`: focus distance in meters.

The DoF renderer uses:

- focal length from `focal_length_mm`
- aperture diameter computed as `focal_length_mm / 1000 / f_stop`
- focus distance from `focal_dist_m`

Fallback defaults exist in the loader for compatibility, but calibrated focal,
aperture, and focus metadata are required for faithful FocalGS behavior.

See `DATASET.md` for scene statistics, archive integrity checks, and the exact
synthetic split. See `docs/IMPLEMENTATION_NOTES.md` for paper/code differences
and the intentional Ship resolution mismatch. See
`docs/SYNTHETIC_DATA_PROVENANCE.md` for the scene-by-scene rights record.

## Training

Official synthetic training command:

```bash
python train.py \
    -s /path/to/dataset \
    -m /path/to/output \
    --configs configs/focalgs_canonical/<scene>.py
```

Available canonical scene config names:

- `tanabata.py`
- `cozyroom.py`
- `pool.py`
- `factory.py`
- `building.py`
- `livingroom.py`
- `ship.py`

Canonical synthetic training details:

- `iterations = 30000`
- `hash = True`
- `use_deblur = True`
- `lambda_dssim = 0.2`
- base scaling regularization weight `0.01`
- opacity regularization weight `0.05`, active after iteration `3000`
- anisotropy regularization off by default
- shifted-camera disparity loss off by default

The canonical release objective is:

```text
L_total =
    0.8 * L1
  + 0.2 * (1 - SSIM)
  + 0.01 * base_scaling_reg
  + 0.05 * opacity_reg       [iteration > 3000]
```

The regularization terms above are part of the release training recipe. They are
not presented as separate method contributions.

## Rendering

Render clear all-in-focus test images:

```bash
python render.py \
    -s /path/to/dataset \
    -m /path/to/model \
    --iteration 30000 \
    --skip_train \
    --skip_video
```

This is the default evaluation rendering mode. It disables physical CoC blur and
writes clear test renders under:

```text
/path/to/model/test/ours_30000/renders
```

Render physical DoF test images:

```bash
python render.py \
    -s /path/to/dataset \
    -m /path/to/model \
    --iteration 30000 \
    --skip_train \
    --skip_video \
    --render_blur
```

`--render_blur` enables inference-time physical DoF rendering using the camera
metadata described above. The DoF renders are written under:

```text
/path/to/model/test/ours_30000_blurry/renders
```

## Evaluation

After rendering clear test images, run:

```bash
python metrics.py \
    -m /path/to/model \
    --iteration 30000
```

`metrics.py` evaluates:

```text
/path/to/model/test/ours_30000/renders
```

against:

```text
/path/to/model/test/ours_30000/sharp_gt
```

and writes the aggregate results to:

```text
/path/to/model/results.json
```

## Canonical Synthetic Scenes

The public canonical synthetic configs cover:

- Tanabata
- Cozyroom
- Pool
- Factory
- Building
- Livingroom
- Ship

## Real-World Data

The original Toy/Plant/Bear experiment lineage does not exactly realize the
paper's stated 15-viewpoint x 5-focal protocol. Later clean Plant/Bear datasets
and strict 24--85 mm splits are separate post-paper benchmarks; no equivalent
clean Toy dataset was found. `docs/REAL_DATA_PROVENANCE.md` records the versions,
split counts, camera limitations, privacy review, and recommended release labels.
`REAL_DATASET.md` gives the portable archive layout, checksums, and verification
commands. The two real-data downloads are versioned separately as
**Experimental Archive** and **Post-paper Clean Benchmark** under
`release/real/`.

## Reproducibility Notes

The frozen release intentionally documents rather than rewrites known
paper/code differences:

- paper linear versus code log focal normalization;
- paper two-weight fusion notation versus the code's single complementary gate;
- paper `0.0004` versus canonical config `0.0002` anchor-growing threshold;
- paper real-data prose versus the recoverable experiment splits.

See `docs/IMPLEMENTATION_NOTES.md` for the exact behavior.

## Citation

The confirmed publication metadata is:

- Title: **FocalGS: Adaptive Focal 3D Gaussian Splatting for Continuous Zoom Rendering**
- Authors: **Zeyu Liu, Chao Wang, Jiahao Wu, Jie Liang, Ronggang Wang**
- Venue: **SIGGRAPH Asia 2026 Conference Papers**
- DOI: **10.1145/3829340.3842186**

The online publication date has not been confirmed and is intentionally
omitted.

```bibtex
@inproceedings{liu2026focalgs,
  title     = {FocalGS: Adaptive Focal 3D Gaussian Splatting for Continuous Zoom Rendering},
  author    = {Liu, Zeyu and Wang, Chao and Wu, Jiahao and Liang, Jie and Wang, Ronggang},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842186},
  publisher = {Association for Computing Machinery}
}
```

## Acknowledgements

This project builds on 3D Gaussian Splatting, Scaffold-GS style anchor-based
representations, `diff-gaussian-rasterization`, `simple-knn`, and
`tiny-cuda-nn`.

## License

See `LICENSE` and `THIRD_PARTY.md`. The repository inherits the
Gaussian-Splatting research-only, non-commercial terms; it is source-available
for research and evaluation and is not licensed for unrestricted commercial
use. Dataset assets have separate redistribution requirements described in
`DATASET.md`, `REAL_DATASET.md`, and `docs/RELEASE_CHECKLIST.md`.
