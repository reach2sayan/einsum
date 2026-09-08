This is a personal project, but contributions are welcome! I want to learn, so please comment. I don't promise to implement all suggestions, but I will surely think them through and through.

## The Windows leg

`MSVC 2022 (release)` and `python bindings (Windows)` in
`.github/workflows/action.yml` carry `continue-on-error: true`. That is
deliberate and temporary: MSVC had never compiled this tree -- `cmake/` carried
MSVC branches nobody had run -- and the cheapest way to find out what it thinks
is to let it say so on every push without blocking anyone.

Ending that period has to be a deliberate act, because `continue-on-error` masks
a job's conclusion to `success` whether its steps passed or failed. In the run
list a working MSVC job and a broken one look identical; only the step-level
conclusions tell them apart. So read the steps, not the badge, before deciding
it is green.

To flip it to blocking, in one change:

1. Delete the two `continue-on-error: true` lines.
2. Add the two contexts to `required_status_checks` in
   `.github/rulesets/main.json` and re-import the ruleset (below).
3. Consider `COMPILE_WARNING_AS_ERROR` for MSVC in
   `cmake/EinsumCompileOptions.cmake`, currently `if (NOT MSVC)`-guarded. A
   blocking leg that does not fail on warnings drifts back to noisy within a
   release.

## Branch protection

`.github/rulesets/main.json` is the ruleset `main` is protected with, kept in
the repository so the required checks and the workflow job names that produce
them can be changed in the same commit. It is not applied automatically --
import it after changing it:

*Settings -> Rules -> Rulesets -> New ruleset -> Import a ruleset*, or

```
gh api -X POST repos/:owner/:repo/rulesets --input .github/rulesets/main.json
# updating an existing one:
gh api repos/:owner/:repo/rulesets --jq '.[] | "\(.id) \(.name)"'
gh api -X PUT repos/:owner/:repo/rulesets/<id> --input .github/rulesets/main.json
```

Two entries in it are load bearing:

- **The `context` strings are job `name:` values** from
  `.github/workflows/action.yml`. Renaming a job silently stops its check being
  required -- the ruleset then waits on a check that no longer reports.
- **`DeployKey` is on the bypass list.** `main` requires a pull request, and the
  weekly release pushes its version bump directly. That push is made over SSH
  with `RELEASE_DEPLOY_KEY`, and a deploy-key push also fires Release's tag
  trigger, which a push made with the workflow's own token could not do.

## Releases

| Workflow | Trigger | What it does |
|---|---|---|
| `Trial release` | manual | Builds the whole wheel matrix and publishes nothing. Run it before tagging by hand, and after any change to `pyproject.toml`, `cmake/`, or the pins. |
| `Release` | tag `v*.*.*` | Checks the tag against `project(EinsteinSummation VERSION ...)`, builds the wheels, cuts the GitHub release. |
| `Weekly release` | Mondays 06:00 UTC, or manual | Bumps the patch number, commits, tags and pushes, which fires `Release`. Does nothing if `main` has not moved since the last tag, and refuses if `CMake` is not green on `HEAD`. |

`Release` and `Trial release` both call `.github/workflows/wheels.yml`, so a dry
run builds wheels the same way a tag does -- a trial that built them differently
would not be a trial of anything.

Setup the weekly release needs once: a deploy key with write access on the
repository, its private half stored as the `RELEASE_DEPLOY_KEY` secret.

## Verifying locally

```
scripts/verify_all.sh              # every preset, plus a clang release build
scripts/verify_all.sh release asan # just those
```

Linux only, and so is `python3 scripts/vendor_headers.py --check`. Nothing here
compiles MSVC; the Windows jobs are the only place that happens.
