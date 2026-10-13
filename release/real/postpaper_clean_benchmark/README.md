# FocalGS Real Data: Post-paper Clean Benchmark

Version: `FocalGS-Real-PostPaper-Clean-24-85-v1`

This frozen candidate contains the strict 24--85 mm Plant and Bear splits
derived from the later clean data. It is an optional post-paper benchmark. It
was not used for the paper experiments and must not be labeled or cited as the
paper's original real-data protocol.

Archive: `focalgs_real_postpaper_strict_24_85.tar.gz`

Bytes: `84299583`

SHA256:
`1d12603d2616625373379ea7755bf446a48f3f3a6c72b602b0fe66cd15a53b88`

## Actual image and split statistics

| Scene | Original captures | Later clean registrations | Included strict images | Train | Test | Resolution | Focal settings (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Plant | 90 | 79 | 72 | 58 | 14 | 800 x 533 | 24, 35, 50, 70, 85 |
| Bear | 96 | 74 | 70 | 57 | 13 | 800 x 533 | 24, 35, 50, 70, 85 |

There is no corresponding clean/strict Toy release candidate. Long-focal
registrations outside 24--85 mm were excluded by this post-paper benchmark
protocol rather than reassigned or repaired.

## Contents and verification

Each scene contains sanitized train/test JSON, JSON-referenced PNGs, and the
COLMAP sparse binary triplet. Raw filenames, CR3 files, serial-related metadata,
timestamps, exposure/ISO and lens strings, databases, caches, and symbolic
links are excluded.

```bash
sha256sum --check SHA256SUMS
python scripts/release/verify_release.py \
  focalgs_real_postpaper_strict_24_85.tar.gz
```
