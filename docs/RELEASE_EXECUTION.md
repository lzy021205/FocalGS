# Release Execution Plan

This plan is prepared for use only after patent and rights approval. It does
not authorize a push, upload, or visibility change.

## 1. Code

The local `release/github-integration-20261013` branch descends from the current
public GitHub `main` commit. Immediately before publication:

```bash
git fetch https://github.com/lzy021205/FocalGS.git \
  main:refs/remotes/github/main
git merge-base --is-ancestor \
  refs/remotes/github/main release/github-integration-20261013
git diff --check refs/remotes/github/main..release/github-integration-20261013
```

Review the exact diff and push only a normal fast-forward. If the ancestry check
fails, integrate the new public commits locally and repeat review. Never force
push or replace the repository.

## 2. Data records

Use separate versioned records so that rights and scientific labels remain
independent:

1. **FocalGS synthetic dataset**: seven frozen scene archives plus their
   checksums and `DATASET.md`; publish only the scenes cleared in the signed
   asset-rights matrix.
2. **FocalGS paper-experiment real data**: one Toy/Plant/Bear archival candidate
   with `REAL_DATASET.md` and the protocol disclosure.
3. **FocalGS post-paper strict benchmark**: a separate optional Plant/Bear
   record whose title and description include "post-paper strict 24--85 mm".

Zenodo is the recommended archival host because each record can be versioned
and assigned a citable DOI. A Hugging Face Dataset mirror may be added for
convenient browsing, but it should point to the same version and checksums.
Do not use GitHub release assets for the 2.6 GB synthetic payload.

Keep every host record as an unpublished private draft until its own rights gate
passes. A Zenodo embargo still exposes record metadata after publication, so use
it before patent clearance only if counsel explicitly approves that disclosure.
Do not reserve a public URL in repository documentation by guessing it; insert
the host-generated URL and DOI only after the private draft has been reviewed.

## 3. Upload and download verification

Upload the archives without recompression. Compare host/downloaded bytes with
`release/SHA256SUMS`, then run:

```bash
sha256sum --check SHA256SUMS
python scripts/release/verify_release.py SHA256SUMS
python scripts/release/verify_release.py /path/to/downloaded/archive.tar.gz
```

Extract into a new directory, confirm there are no links, and run the exact
FocalGS loader preflight. For code, create the documented environment and run a
known-checkpoint render/metrics smoke test on an approved GPU before announcing
the release.

## 4. Publication order

Publish the code first or at the same time as the paper record. Activate each
dataset record only after its independent rights review. Add the resulting
paper, code, synthetic-data, paper-real-data, and post-paper-benchmark links to
README and the project page in one reviewed metadata commit.
