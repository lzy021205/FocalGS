# GitHub Integration Plan

Read-only audit date: 2026-10-13

Official repository: `https://github.com/lzy021205/FocalGS`

The public `main` branch was rechecked read-only on 2026-10-13 and contains one
commit:

```text
0ad544d258b936bbdd67911d65130648db56d6df  Initial commit
```

That commit adds a two-line README. The local development `main` has an
independent root, so it must never be pushed directly. A separate local branch,
`release/github-integration-20261013`, was created from the public commit and
contains the reviewed release tree as normal descendant commits. No remote
write occurred.

## Safe offline preparation

Keep release-only commits synchronized onto the integration branch with normal
local cherry-picks, then inspect the resulting tree. Immediately before the
approved publication window:

```bash
git fetch https://github.com/lzy021205/FocalGS.git \
    main:refs/remotes/github/main

git merge-base --is-ancestor \
    refs/remotes/github/main release/github-integration-20261013
git diff --check \
    refs/remotes/github/main..release/github-integration-20261013
```

If ancestry succeeds, pushing that branch to `main` will be a fast-forward
update. If public `main` changed, merge or rebase those public changes into the
integration branch locally and review again. Do not use `--force`, delete the
remote branch, create a replacement repository, or rewrite the public initial
commit.

No remote was added, no remote branch was changed, and no push was performed
during offline release preparation.
