# FocalGS Real Data: Post-paper Strict 24--85 mm

Version: `FocalGS-Real-PostPaper-Strict-24-85-v2`

This public benchmark contains Toy, Plant, and Bear with the five focal
settings 24, 35, 50, 70, and 85 mm. It was prepared after the paper experiments
and is not the paper's original real-data protocol.

Toy is a strict subset of the historical Toy experimental archive: only its
105 mm frames were removed, while its train/test assignments and image files
were preserved. Plant and Bear are the later clean strict splits.

Archive: `focalgs_real_postpaper_strict_24_85_v2.tar.gz`

Bytes: `130977079`

SHA256:
`5d73b83e8409dabc948bde03be3d38d72b2fcea18787c4be8c391d2f89ff96a1`

| Scene | Train | Test | Resolution | Focal settings (mm) |
| --- | ---: | ---: | --- | --- |
| Toy | 60 | 15 | 800 x 533 | 24, 35, 50, 70, 85 |
| Plant | 58 | 14 | 800 x 534 | 24, 35, 50, 70, 85 |
| Bear | 57 | 13 | 800 x 534 | 24, 35, 50, 70, 85 |

Each scene contains sanitized train/test JSON, JSON-referenced PNGs, and the
COLMAP sparse binary triplet. Raw captures, identifying metadata, databases,
caches, and symbolic links are excluded.

```bash
sha256sum --check SHA256SUMS
python scripts/release/verify_release.py \
  focalgs_real_postpaper_strict_24_85_v2.tar.gz
```
