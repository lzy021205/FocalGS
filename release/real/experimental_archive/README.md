# FocalGS Real Data: Experimental Archive

Version: `FocalGS-Real-Experimental-Archive-v1`

This frozen candidate contains the recoverable Toy, Plant, and Bear datasets
that correspond to the available pre-paper training commands, logs,
configuration snapshots, rendered results, and 30,000-iteration checkpoints.
It is an experimental archive, not a reconstruction of the protocol stated in
the paper prose.

Archive: `focalgs_real_paper_experiment_candidate.tar.gz`

Bytes: `139512358`

SHA256:
`f739ce473f1d167c715daa1b64232f0905f089f5936028d4b21ac3abca7933fa`

## Actual image and split statistics

| Scene | Capture/source evidence | Successfully registered and included | Train | Test | Resolution | Included focal settings (mm) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Toy | 96 source PNGs; 16 apparent six-image capture sequences | 90 | 72 | 18 | 800 x 533 | 24, 35, 50, 70, 85, 105 |
| Plant | 90 CR3 captures; 15 six-focal capture sequences | 76 | 71 | 5 | 800 x 533 | 24, 35, 50, 70, 85, 87, 105 |
| Bear | 96 CR3 captures; 16 six-focal capture sequences | 68 | 64 | 4 | 800 x 533 | 24, 35, 50, 70, 85, 105 |

Toy raw capture records are unavailable, so its capture-sequence count is an
inference from filenames and ordering. Plant and Bear counts are supported by
the retained raw-capture inventory. Registration failures make the Plant and
Bear focal grids uneven.

The paper describes 15 viewpoints at five focal lengths with two held-out
viewpoints. These archived experiments do not exactly match that description.
Do not alter images or splits to make them appear to match it. Publish this
candidate only with the approved protocol-disclosure note.

## Contents and verification

Each scene contains JSON-referenced PNGs, `transforms_train.json`,
`transforms_test.json`, and `sparse/0/{cameras.bin,images.bin,points3D.bin}`.
No raw CR3, database, pickle, cache, embedded image metadata, or symbolic link
is included.

```bash
sha256sum --check SHA256SUMS
python scripts/release/verify_release.py \
  focalgs_real_paper_experiment_candidate.tar.gz
```
