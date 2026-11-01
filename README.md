# FocalGS: Adaptive Focal 3D Gaussian Splatting for Continuous Zoom Rendering

**Zeyu Liu, Chao Wang, Jiahao Wu, Jie Liang, Ronggang Wang**

[Project page](https://lzy021205.github.io/FocalGS/) |
[Dataset](https://huggingface.co/datasets/Liuzeyu456/FocalGS-Dataset)

FocalGS reconstructs a static scene from multi-view, multi-focal images and
renders continuous zoom views from a single 3D Gaussian representation. It
combines static anchor features with focal-conditioned residual features using
adaptive fusion. Physical depth-of-field rendering is used for defocused-image
supervision during training.

## Installation

Clone the repository:

```bash
git clone https://github.com/lzy021205/FocalGS.git
cd FocalGS
```

The required CUDA extension sources are included under `submodules/`; no
additional Git submodule initialization is required. The released environment
uses Python 3.9, PyTorch 2.2.0, and CUDA 11.8.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

conda env create -f environment.yml
conda activate FocalGS
```

If GPU architecture detection is unavailable while building the extensions,
set it explicitly, for example:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

The environment includes GCC/G++ 11. If a compatible compiler is not selected
automatically, run:

```bash
export FOCALGS_ENV_PREFIX="$(conda info --base)/envs/FocalGS"
export CC="$FOCALGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$FOCALGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
```

An NVIDIA GPU is required for training and rendering. COLMAP is required only
when a dataset does not already contain a sparse initialization under
`sparse/0`.

## Dataset

Download the synthetic and real scenes from the
[FocalGS dataset repository](https://huggingface.co/datasets/Liuzeyu456/FocalGS-Dataset).
Extract each scene before use. A scene follows this structure:

```text
<scene>/
  images/
  sparse/0/
  transforms_train.json
  transforms_test.json
  transforms_train_sharp.json
```

- `transforms_train.json`: defocused training images.
- `transforms_test.json`: sharp test images.
- `transforms_train_sharp.json`: corresponding sharp training views.

The camera metadata includes the pose, field of view, focal length, f-number,
and focus distance required by FocalGS.

## Training

Train a scene with its configuration file:

```bash
python train.py \
    -s /path/to/dataset/<scene> \
    -m /path/to/output/<scene> \
    --configs configs/focalgs_canonical/<scene>.py
```

Available scene configurations are `tanabata`, `cozyroom`, `pool`, `factory`,
`building`, `livingroom`, and `ship`. The released configurations train for
30,000 iterations.

## Rendering

Render clear test views:

```bash
python render.py \
    -s /path/to/dataset/<scene> \
    -m /path/to/output/<scene> \
    --iteration 30000 \
    --skip_train \
    --skip_video
```

To render physical depth of field, add `--render_blur`:

```bash
python render.py \
    -s /path/to/dataset/<scene> \
    -m /path/to/output/<scene> \
    --iteration 30000 \
    --skip_train \
    --skip_video \
    --render_blur
```

## Evaluation

After rendering the clear test views, run:

```bash
python metrics.py \
    -m /path/to/output/<scene> \
    --iteration 30000
```

The aggregate metrics are written to `<model_path>/results.json`.

## Citation

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

This project builds on 3D Gaussian Splatting, Scaffold-GS,
`diff-gaussian-rasterization`, `simple-knn`, and `tiny-cuda-nn`.

## License

See [LICENSE](LICENSE). This repository is available for research and
evaluation under the terms stated there. Third-party components retain their
respective license files in the vendored source directories.
