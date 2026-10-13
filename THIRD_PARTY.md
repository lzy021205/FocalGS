# Third-Party Provenance

This document records provenance for source code and native extensions vendored
in the public FocalGS release tree. It does not change or replace the repository
`LICENSE`.

| Component | Origin | License | Modified by FocalGS | License file present |
| --- | --- | --- | --- | --- |
| FocalGS main code | FocalGS codebase built on 3D Gaussian Splatting and Scaffold-GS style anchor infrastructure | Gaussian-Splatting License | Yes | `LICENSE` |
| `submodules/diff-gaussian-rasterization` | [GraphDeco/Inria differential Gaussian rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization) from 3D Gaussian Splatting | Gaussian-Splatting License | Yes. This release contains the FocalGS physical DoF CUDA interface, including `use_deblur`, `lens_aperture`, `lens_focus_depth`, `lens_focal_meters`, `lens_focal_pixels`, and `grad_focus`. | `submodules/diff-gaussian-rasterization/LICENSE.md` |
| `submodules/diff-gaussian-rasterization/third_party/glm` | GLM, OpenGL Mathematics | Happy Bunny License or MIT License | No FocalGS-specific changes identified in this audit | `submodules/diff-gaussian-rasterization/third_party/glm/copying.txt` |
| `submodules/diff-gaussian-rasterization/third_party/stbi_image_write.h` | stb image write | MIT License or Public Domain / Unlicense | No FocalGS-specific changes identified in this audit | License text embedded in `submodules/diff-gaussian-rasterization/third_party/stbi_image_write.h` |
| `submodules/simple-knn` | [GraphDeco/Inria simple KNN extension](https://gitlab.inria.fr/bkerbl/simple-knn) from 3D Gaussian Splatting | Gaussian-Splatting License | No FocalGS-specific changes identified in this audit; exact upstream diff is unavailable because the copied source has no Git metadata. | `submodules/simple-knn/LICENSE.md` |
| `submodules/tiny-cuda-nn` | [NVIDIA tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) | BSD-3-Clause | No FocalGS-specific changes identified in this audit; exact upstream commit is unavailable because the copied source has no Git metadata. | `submodules/tiny-cuda-nn/LICENSE.txt` |
| `submodules/tiny-cuda-nn/dependencies/cutlass` | NVIDIA CUTLASS | BSD-3-Clause | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/cutlass/LICENSE.txt` |
| `submodules/tiny-cuda-nn/dependencies/fmt` | fmt | MIT License | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/fmt/LICENSE.rst` |
| `submodules/tiny-cuda-nn/dependencies/cmrc` | CMakeRC | MIT License | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/cmrc/LICENSE.txt` |
| `submodules/tiny-cuda-nn/dependencies/json` | nlohmann/json | MIT License | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/json/LICENSE.MIT` |
| `submodules/tiny-cuda-nn/dependencies/stbi` | stb | MIT License or Public Domain / Unlicense | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/stbi/LICENSE` |

Python packages installed from `environment.yml`, such as `mmcv`, `open3d`,
`lpips`, and `torchmetrics`, are package dependencies rather than vendored source
in this repository.

## Revision status

The vendored directories are ordinary tracked files rather than Git submodules,
and their original `.git` metadata is absent. Exact upstream commit IDs have not
been recovered for `diff-gaussian-rasterization`, `simple-knn`, or
`tiny-cuda-nn`. This is a release metadata blocker: identify the upstream
revisions from the source archive or perform and record a source diff before
public release. Do not infer a commit from directory names.

The root `LICENSE` is the Gaussian-Splatting research-only, non-commercial
license. It permits redistribution under its conditions but is not an
unrestricted open-source license. Dataset and scene-asset licenses are separate
from the code license and are tracked in `DATASET.md` and
`docs/RELEASE_CHECKLIST.md`.
