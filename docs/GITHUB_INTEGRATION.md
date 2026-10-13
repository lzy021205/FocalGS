# GitHub Integration Plan

Read-only audit date: 2026-10-13

Official repository: `https://github.com/lzy021205/FocalGS`

The public `main` branch currently contains one commit:

```text
0ad544d258b936bbdd67911d65130648db56d6df  Initial commit
```

That commit adds a two-line README. The local release repository has an
independent root commit and no configured remote. A direct push from local
`main` would therefore be non-fast-forward and must not be used.

## Safe offline preparation

After the release files are approved locally:

```bash
git fetch https://github.com/lzy021205/FocalGS.git \
    main:refs/remotes/github/main

git worktree add /tmp/FocalGS-github-integration github/main
```

Copy the approved local release tree into that worktree while excluding `.git`,
`release_staging`, internal audit material, private data, checkpoints, and
unapproved media. Replace the public two-line README with the reviewed release
README, then commit on a local branch:

```bash
git -C /tmp/FocalGS-github-integration switch -c release/offline-candidate
git -C /tmp/FocalGS-github-integration add --all
git -C /tmp/FocalGS-github-integration commit
```

This creates a normal descendant of the existing public commit. Before any
future publication, fetch again and verify ancestry:

```bash
git fetch https://github.com/lzy021205/FocalGS.git \
    main:refs/remotes/github/main
git merge-base --is-ancestor \
    refs/remotes/github/main release/offline-candidate
```

If the check succeeds, pushing the reviewed candidate to `main` will be a
fast-forward update. If the public branch changed, merge or rebase those public
changes locally and review the result. Do not use `--force`, delete the remote
branch, create a replacement repository, or rewrite the public initial commit.

No remote was added, no remote branch was changed, and no push was performed
during offline release preparation.
