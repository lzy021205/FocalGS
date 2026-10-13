# Real-Scene Version Provenance

Date: 2026-10-13

This record separates the data used by the recoverable pre-paper experiments
from two later benchmark revisions. The original research tree was inspected
read-only; no data was rewritten and no model was retrained.

## Conclusion

The local archive does not contain a real-scene dataset that exactly implements
the paper statement of 15 viewpoints x 5 focal lengths with 2 held-out
viewpoints for each of Toy, Plant, and Bear. The strongest recoverable
pre-paper lineage is listed below. It is suitable for an archival
"data used by the available experiment" release with explicit caveats, but it
must not be described as an exact realization of the paper's stated protocol.

## Recoverable paper-experiment lineage

### Toy

- Historical data directory: `data/toy_final_resize`
- Direct final-run script: `run_toy_final.sh`
- Historical model output: `output/toy_final_resize_run`
- Checkpoints: iterations 15000 and 30000
- Run time: 2026-04-29 to 2026-04-30, before the paper PDF
- Split: 72 train / 18 test, 800 x 533
- Focals: 24, 35, 50, 70, 85, and 105 mm; 12 train and 3 test images per focal
- Metadata: `f_stop=16`; focus distances span 0.775--1.335 m
- Missing: `transforms_train_sharp.json`

This is the best-confirmed real lineage because a named final command, source
path, model path, logs, and 30k checkpoint agree. It contains 15 apparent
capture sweeps x 6 focal settings and holds out 3 sweeps, not the paper's
15 x 5 / 2-view statement. The archived COLMAP camera text reports 600 x 400
while the released PNGs are 800 x 533; runtime uses JSON FOV and actual PNG
dimensions, while the point cloud still comes from that reconstruction.

### Plant

- Historical data directory: `data/plant_new`
- Historical model output: `output/plant_new_experiment`
- Checkpoints: iterations 15000 and 30000
- Results file: 30k render/evaluation output exists
- Run time: 2026-05-11, before the paper PDF
- Split: 71 train / 5 test, 800 x 533
- Train focal counts: 24/35/50/70 mm x 14, 85 mm x 11, 87 mm x 1,
  105 mm x 3
- Test: one image each at 24, 35, 50, 70, and 85 mm
- Metadata: `f_stop=16`; focus distances include 1.03--2.99 m and an 11.12 m
  outlier
- Missing: `transforms_train_sharp.json`

Training configuration, logs, checkpoint timestamps, rendered output, and
results all point to `plant_new`. The paper's displayed Plant comparison was
extracted from the paper PDF, so it does not provide an independent filename or
checkpoint-level identity check. This version is therefore the strongest
candidate, not cryptographic proof of the exact figure source.

### Bear

- Historical data directory: `data/bear_newest`
- Historical model output: `output/bear_newest_experiment`
- Checkpoints: iterations 15000 and 30000
- Paired no-defocus experiment and results also exist
- Run time: 2026-05-11, before the paper PDF
- Split: 64 train / 4 test, 800 x 533
- Train focal counts: 24/35/50 mm x 15, 70 mm x 12, 85 mm x 5,
  105 mm x 2
- Test: one image each at 24, 35, 50, and 70 mm
- Metadata: `f_stop=4`
- Missing: `transforms_train_sharp.json`

This lineage is preferable to `bear_800_standard`: the latter patches failed
long-focal test registrations by borrowing a 24 mm camera and subsequently
labels some physical 85/105 mm images near 24.45 mm. It is not a safe paper
release source.

## Later clean datasets

The clean baseline was created on and after 2026-08-20 and is separate from the
pre-paper experiments.

| Scene | Train/test | Focal range | Notes |
| --- | ---: | --- | --- |
| Plant clean | 63 / 16 | 24--105 mm | 79 referenced registered images; per-frame OPENCV calibration and EXIF-derived fields retained |
| Bear clean | 58 / 16 | 24--105 mm | 74 referenced registered images; 105 mm registration coverage is weak |
| Toy clean | unavailable | -- | No Toy dataset exists in the clean `focalgs_ready` tree |

The clean Plant/Bear directories use symbolic-link image roots in the research
tree and cannot be copied as-is into a portable release. Their JSON includes
more metadata than the frozen runtime consumes.

## Strict 24--85 mm validation splits

These were derived after the clean baseline to avoid the least reliable
long-focal registrations and unstable viewpoint groups.

| Scene | Train/test | Focal range | Status |
| --- | ---: | --- | --- |
| Plant strict | 58 / 14 | 24, 35, 50, 70, 85 mm | post-paper validation benchmark |
| Bear strict | 57 / 13 | 24, 35, 50, 70, 85 mm | post-paper validation benchmark |
| Toy strict | unavailable | -- | no clean Toy source |

These splits are the most defensible current Plant/Bear validation data, but
they are not the data used by the April/May paper experiments and do not repair
the paper/data mismatch.

## Camera model and distortion

Plant and Bear use per-image COLMAP `OPENCV` cameras. The paper-candidate JSON
preserves `fl_x`, `fl_y`, `cx`, and `cy`; the later clean JSON additionally
preserves `k1`, `k2`, `p1`, `p2`, image size, camera IDs, and EXIF-derived
fields. The frozen FocalGS runtime instead builds a centered pinhole projection
from `camera_angle_x` and actual image dimensions. It does not apply principal
point offsets or OPENCV distortion. Any real-scene release and result table must
state this limitation.

## Privacy and metadata

Sample PNGs from all three paper-candidate datasets contain no embedded EXIF or
PNG metadata. The inspected Plant and Bear CR3 originals contain camera body,
lens serial numbers, and capture timestamps; no GPS fields were returned for
the inspected samples. Owner, artist, and copyright fields were empty. Raw CR3
files must not be published without a full metadata scrub and a capture-rights
review. Later clean JSON exposes source CR3 filenames, exposure, ISO, lens model,
and focus metadata; those fields require a deliberate public-metadata decision.

The real scenes may also reveal the capture environment. Before release, the
authors must confirm ownership/consent for the depicted objects and locations.

## Recommended release treatment

1. Preserve `toy_final_resize`, `plant_new`, and `bear_newest` as an explicitly
   labeled archival paper-experiment candidate if reproducing available
   checkpoints is the priority.
2. Publish the strict Plant/Bear 24--85 mm data only as a separate post-paper
   validation benchmark.
3. Do not substitute clean or strict data for the paper candidate in filenames,
   tables, or download labels.
4. Keep `bear_800_standard` and raw CR3 files out of the public package.
5. Mark the paper's 15 x 5 protocol as unresolved unless an additional original
   dataset archive is located.
