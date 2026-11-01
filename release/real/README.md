# FocalGS Real Data: Post-paper Strict 24--85 mm

Version: `FocalGS-Real-PostPaper-Strict-24-85-v3`

The public release uses one short-named archive per scene:

- `real/toy/toy.tar.gz`
- `real/plant/plant.tar.gz`
- `real/bear/bear.tar.gz`

Toy is a strict subset of the historical Toy experimental archive: only its
105 mm frames were removed, while its train/test assignments and image files
were preserved. Plant and Bear are the later clean strict splits. These three
archives form a post-paper benchmark and are not the paper's original
real-data protocol.

| Scene | Train | Test | Resolution | Focal settings (mm) |
| --- | ---: | ---: | --- | --- |
| Toy | 60 | 15 | 800 x 533 | 24, 35, 50, 70, 85 |
| Plant | 58 | 14 | 800 x 534 | 24, 35, 50, 70, 85 |
| Bear | 57 | 13 | 800 x 534 | 24, 35, 50, 70, 85 |

Each archive contains one scene directory with sanitized train/test JSON,
JSON-referenced PNGs, and the COLMAP sparse binary triplet. Raw captures,
identifying metadata, databases, caches, and symbolic links are excluded.
