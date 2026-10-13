# Offline Release Checklist

Date: 2026-10-13

This checklist is the publication gate for the code, project page, and data.
The repository and archives remain offline until the patent restriction is
lifted and every blocking item is signed off.

## Ready to publish after the patent hold

- Frozen Full FocalGS source and seven canonical 30k scene configurations.
- Installation, training, clear rendering, physical-DoF rendering, and metrics
  commands.
- Implementation notes for focal normalization, complementary fusion,
  densification threshold, and data-protocol differences.
- Seven-scene synthetic staging tree with only JSON-referenced images, three
  camera JSONs, COLMAP binary sparse models, file lists, and SHA256 manifests.
- Read-only GitHub history audit and a non-force integration procedure that
  preserves public commit `0ad544d258b936bbdd67911d65130648db56d6df`.
- Project-page structure and media references. Quantitative Results and
  Controllable DoF sections remain intentionally absent.

## Technically ready but awaiting redistribution authorization

- Synthetic archives for Tanabata, Cozyroom, Pool, Factory, Building,
  Livingroom, and Ship.
- Rendered dataset images and COLMAP models derived from the underlying scene
  assets.
- Project-page paper figures and comparison crops until the authors confirm
  conference/paper figure reuse terms.

Tanabata, Cozyroom, Pool, and Factory appear to use scene assets also associated
with Deblur-NeRF, but the local archive does not establish a complete source
asset and license chain. Building, Livingroom, and Ship also lack a locally
verified source/license record. No public data upload is allowed until a human
records each asset source, creator, license, required attribution, modification
permission, and redistribution permission.

## Files that require an explicit paper/data difference note

- `docs/IMPLEMENTATION_NOTES.md`: linear paper focal coordinate versus log code
  coordinate; two paper weights versus one complementary code gate; paper
  `0.0004` versus release config `0.0002` growing threshold.
- Ship: 800 x 384 defocused train images versus 800 x 395 sharp train/test
  images. Preserve without resizing.
- Real paper-candidate datasets: recoverable split and focal counts do not match
  the paper's 15 viewpoints x 5 focal settings / 2 held-out viewpoints prose.
- Plant/Bear: OPENCV intrinsics and distortion exist in metadata, while the
  frozen main runtime uses centered `camera_angle_x` projection and ignores
  distortion.
- Later clean and strict Plant/Bear benchmarks are post-paper data and must be
  labeled accordingly.

## Do not publish yet

- Any Git push, release, public archive upload, GitHub visibility change, or
  project-page deployment while the patent hold remains active.
- Raw Plant/Bear CR3 photos: inspected samples contain body/lens serial numbers
  and capture timestamps. A complete privacy scrub is required.
- Real-scene images until capture ownership, location/object consent, and the
  intended paper-candidate versus strict-benchmark release policy are approved.
- `bear_800_standard`, whose patched missing registrations can assign physical
  85/105 mm frames approximately 24.45 mm camera metadata.
- Research checkpoints and historical experiment outputs unless a separate
  model-release manifest is approved.
- `fast_forward/` working/review media and internal project audit files.

## Blocking decisions before publication

1. Patent counsel/author approval to disclose code, data, paper, and page.
2. Final public author list, affiliations, paper URL/DOI, year/venue metadata,
   and the preferred BibTeX citation. `CITATION.cff` is deliberately marked
   incomplete until these are supplied.
3. Written redistribution clearance for every synthetic scene asset and for
   generated derivatives.
4. Real-data policy: whether to publish the recoverable paper-experiment
   candidates, the post-paper strict benchmark, or both as clearly separated
   downloads.
5. Full privacy/rights review for real images and removal of raw camera serial,
   timestamps, and any unwanted source-filename metadata.
6. Upstream revision identification for vendored code where the copied source
   lacks Git metadata, followed by a final third-party license review.
7. Final archive URLs and checksums inserted into README, DATASET.md, and the
   project page.

## Final technical gate

- Verify every archive with `sha256sum --check` after upload and after a clean
  download.
- Extract each archive into a new directory and run the offline loader preflight.
- On a supported NVIDIA GPU, run one known-checkpoint render and a small metrics
  smoke test in the documented release environment.
- Confirm `git diff` contains no core-method change and no untracked private
  material.
- Re-fetch GitHub `main`; integrate only if the audited public head is still an
  ancestor of the proposed release commit. Never force push.
