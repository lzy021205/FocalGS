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
| `submodules/simple-knn` | [GraphDeco/Inria simple KNN extension](https://gitlab.inria.fr/bkerbl/simple-knn), nearest upstream revision `86710c2d4b46680c02301765dd79e465819c8f19` | Gaussian-Splatting License | Eight paths match the upstream tree. `simple_knn.cu` only moves the existing `<cfloat>` include; the local `LICENSE.md` omits an unrelated trailing pytorch-ssim note. | `submodules/simple-knn/LICENSE.md` |
| `submodules/tiny-cuda-nn` | [NVIDIA tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), local source archive identifies revision `2e757bbe781db59c4980d389d7dccbf5edc09669` | BSD-3-Clause | The 123 common files differ in four build/documentation files. The vendored tree additionally includes generated RTC headers and expanded dependency trees, and omits upstream example/readme data. Treat the commit as the recovered baseline, not a byte-identical tree. | `submodules/tiny-cuda-nn/LICENSE.txt` |
| `submodules/tiny-cuda-nn/dependencies/cutlass` | NVIDIA CUTLASS | BSD-3-Clause | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/cutlass/LICENSE.txt` |
| `submodules/tiny-cuda-nn/dependencies/fmt` | fmt | MIT License | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/fmt/LICENSE.rst` |
| `submodules/tiny-cuda-nn/dependencies/cmrc` | CMakeRC | MIT License | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/cmrc/LICENSE.txt` |
| `submodules/tiny-cuda-nn/dependencies/json` | nlohmann/json | MIT License | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/json/LICENSE.MIT` |
| `submodules/tiny-cuda-nn/dependencies/stbi` | stb | MIT License or Public Domain / Unlicense | No FocalGS-specific changes identified in this audit | `submodules/tiny-cuda-nn/dependencies/stbi/LICENSE` |

Python packages installed from `environment.yml`, such as `mmcv`, `open3d`,
`lpips`, and `torchmetrics`, are package dependencies rather than vendored source
in this repository.

## Vendored content fingerprints

The aggregate digest is SHA256 over the concatenation of sorted lines
`<file-sha256>  <relative-path>\n`, excluding `.git` and `__pycache__`
directories. Reproduce it with `python scripts/release/hash_tree.py <tree>`.

| Tree | Files | Bytes | Aggregate SHA256 |
| --- | ---: | ---: | --- |
| `submodules/diff-gaussian-rasterization` | 1,546 | 20,009,636 | `edd3bd5282abd7bad26176e7b9916dcc53743ba2b56eafaa2ca3536a2285dd32` |
| `submodules/simple-knn` | 8 | 13,404 | `75dbab86c169d928d579733bd5def5ee73cb5429c9e944998018022382ca3851` |
| `submodules/tiny-cuda-nn` | 6,197 | 122,491,549 | `5e502ba3ef54755a359df6be4fed233ccb8db8fdca2a78c3d33d36fd0e6374c2` |

The recovered tiny-cuda-nn dependency archives identify these revisions:

- CUTLASS: `1eb6355182a5124639ce9d3ff165732a94ed9a70`
- fmt: `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa`
- CMakeRC: `952ffddba731fc110bd50409e8d2b8a06abbd237`

## Revision status

The vendored directories are ordinary tracked files rather than Git submodules,
and their original `.git` metadata is absent. The simple-knn comparison and
tiny-cuda-nn baseline above are reproducible, while the modified
`diff-gaussian-rasterization` parent revision remains unknown. The physical-DoF
CUDA changes make a byte-identical upstream match impossible; retain the source
repository, local fingerprint, license, and modification notice instead of
inventing an upstream commit.

The root `LICENSE` is the Gaussian-Splatting research-only, non-commercial
license. It permits redistribution under its conditions but is not an
unrestricted open-source license. Dataset and scene-asset licenses are separate
from the code license and are tracked in `DATASET.md` and
`docs/RELEASE_CHECKLIST.md`.
