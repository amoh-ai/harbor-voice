# Contributing

harbor-voice welcomes contributions. This guide covers how to set up, what
every change is checked against, and how to add a provider or a benchmark.

## Understand what you contribute

Like [harbor](https://github.com/harbor-framework/harbor), we follow
Ghostty's golden rule: you must understand the code you contribute. Use coding
agents as much as you like, but write your own PR description, explain the
approach and its tradeoffs, and say which agents you used and how.

## Set up

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker for the
tests that run the benchmarks' own code.

```bash
git clone https://github.com/amoh-ai/harbor-voice
cd harbor-voice
uv sync          # harbor-voice, both adapters, every extra, dev tools
```

From a checkout, each trial installs harbor-voice and the adapter into its
environment from these source folders, so a change runs as it is. An install
from PyPI installs the same released versions instead.

## Before you open a PR

```bash
uv run pytest
uv run ruff check src adapters scripts tests
uv run ruff format --check src adapters scripts tests
uv run ty check
```

- **Tests that need Docker** and the benchmarks' images (e.g.
  `eva-base:039efa0a`) skip without them. If you touch an adapter's call path,
  its sidecar or its goldens, run them with Docker up.
- **Format only the files you changed**; don't reformat the whole repository.
- **CI** runs the same checks on Linux and Windows.

## Ground rules

- **No secrets.** Keys come from the environment. Generated tasks hold only
  `${VAR}` templates, and a test enforces it.
- **No audio in JSON.** Trajectories reference audio by content hash.
- **Copied code stays verbatim.** `adapters/tau-voice/src/tau_voice/tau2_audio/`
  and `adapters/eva/src/eva_adapter/vendored/` are copies of the originals;
  regenerate them, don't edit them.
- **Copied code keeps its license.** Code copied from another project keeps
  that project's LICENSE file beside it.
- **harbor is a pinned dependency** (`harbor==0.25.0`). Import it, never copy
  it, and bump the pin in a PR of its own.
- **A benchmark's results must not move by accident.** A change to an
  adapter's call path (EVA's bridge, tau-voice's step loop, the transports
  they use) has to keep its goldens passing. If it could change scores, say so
  in the PR: the adapter's parity may need to be rerun.

## Adding a provider

A provider implements the transport contract
(`src/harbor_voice/transport/base.py`): four methods, plus a capability
declaration for everything optional.

1. Put it under `src/harbor_voice/providers/<vendor>/` and add it to that
   package's `register_all`. A new vendor also goes in `_VENDORS`
   (`src/harbor_voice/trial.py`).
2. Test it offline, against a fake socket; no test calls a real API.
3. Verify what it declares with `harbor_voice.conformance.run_conformance`,
   from a test against a live session (the module's docstring shows how).

## Adding a benchmark

A benchmark is an adapter: its own package under `adapters/<name>/`, listed in
the workspace (`pyproject.toml`). It has:

- **a converter** that writes the benchmark's tasks as harbor tasks;
- **a call driver**, named in each task's `voice.toml` (`[call] driver`),
  which runs the call the way the benchmark's own harness does
  (`src/harbor_voice/driver.py` is the contract);
- **usually a sidecar** that runs the benchmark's own code for the task, its
  tools and its grade. Its `/health` reports an `API_VERSION`, which the
  driver checks before a call: bump it on both sides when a route or a
  request model changes (`tests/adapters/test_sidecar_api.py` fails until
  you do);
- **tests and fixtures** under `tests/adapters/<name>/`;
- **a `PARITY.md`** showing the adapter reproduces the original harness, in
  harbor's parity format (`parity_experiment.json`).

To release it with harbor-voice, add its package and driver module to
`DRIVER_DISTRIBUTIONS` (`src/harbor_voice/install.py`) and to `PACKAGES` in
`scripts/check_dist.py` and the publish loop in `.github/workflows/release.yml`,
and give harbor-voice an extra that pins it. Files
the installed commands read (a metric script, a setup spec) go inside the
package; what only converting needs (templates, the sidecar) stays beside it.

[tau-voice](adapters/tau-voice) and [EVA](adapters/eva) are the two worked
examples.

## Publishing a dataset version

For maintainers: this pushes images and tags a release. A version is one
commit in each repo. The task folders go in
[harbor-voice-datasets](https://github.com/amoh-ai/harbor-voice-datasets),
tagged `v<version>`, and their entries go in `registry.json` here, every task
pinned to that commit. A published entry never changes: a fix is a new
version. `scripts/hub.py` does the publishing; run everything from one
harbor-voice commit.

**1. Images.** Push what `ghcr.io/amoh-ai` lacks and record the digests:

```bash
uv run python scripts/hub.py images --dry-run   # what would be pushed
uv run python scripts/hub.py images             # build, push, record
```

- Commit the records it writes, `adapters/*/images.json`.
- A tag the registry already serves is never pushed again.
- `--force` rebuilds first; an image built without the OCI labels needs it.

**2. Tasks.** Convert into a clone of the datasets repo beside this one,
with tau2-bench at its pinned commit (`2174a60`) beside it too. Delete a
dataset's folder before converting it again, so the version holds only what
the converter writes now. On Windows, the clone needs `git config
core.longpaths true`.

```bash
D=../harbor-voice-datasets/datasets
uv run tau-voice --tau2-root ../tau2-bench --registry ghcr.io/amoh-ai \
    --output-dir $D/tau-voice-control --complexity control
uv run tau-voice --tau2-root ../tau2-bench --registry ghcr.io/amoh-ai \
    --output-dir $D/tau-voice-regular --complexity regular
uv run tau-voice --tau2-root ../tau2-bench --registry ghcr.io/amoh-ai \
    --output-dir $D/tau-voice-control-test --complexity control --split test
uv run tau-voice --tau2-root ../tau2-bench --registry ghcr.io/amoh-ai \
    --output-dir $D/tau-voice-regular-test --complexity regular --split test
uv run tau-voice --tau2-root ../tau2-bench --registry ghcr.io/amoh-ai \
    --output-dir $D/tau-voice-parity-control --complexity control --split parity \
    --user-llm gpt-4.1-mini-2025-04-14
uv run tau-voice --tau2-root ../tau2-bench --registry ghcr.io/amoh-ai \
    --output-dir $D/tau-voice-parity-regular --complexity regular --split parity \
    --user-llm gpt-4.1-mini-2025-04-14
uv run eva-adapter convert --registry ghcr.io/amoh-ai --out $D/eva
uv run eva-adapter sample --per-domain 20 --seed 300 -o eva-parity.json
uv run eva-adapter convert --registry ghcr.io/amoh-ai --out $D/eva-parity \
    --records-from eva-parity.json
rm $D/eva/job.yaml $D/eva-parity/job.yaml eva-parity.json
```

- With `--registry ghcr.io/amoh-ai`, every image is pinned by digest. A
  converter refuses to write anything if an image of this checkout isn't
  recorded: run step 1 first.
- The parity datasets are the samples their `PARITY.md` ran: the 40 tasks of
  `parity_tasks.json` with the run's caller model, and EVA's 60 records.
- `eva-adapter convert` also writes a `job.yaml` with local paths; the
  datasets repo's CI rejects it.

**3. Release the tasks.** In the datasets repo, review the diff, then:

```bash
git add -A && git commit -m "v0.1.0" && git push
git tag v0.1.0 && git push origin v0.1.0
```

Its CI (the task checks, then `harbor lint`) must pass. Merge pull requests
there with a merge commit: harbor fetches a task by its commit, so a commit a
squash or a rebase drops breaks every version pinned to it.

**4. Register.** Here, once the tag is on the remote:

```bash
uv run python scripts/hub.py register --datasets-repo ../harbor-voice-datasets \
    --commit v0.1.0 --version 0.1.0 \
    --dataset tau-voice-control "tau-voice: all 278 tau2 tasks, control speech" \
    --dataset tau-voice-regular "tau-voice: all 278 tau2 tasks, regular speech" \
    --dataset tau-voice-control-test "tau-voice: tau2's test split (100), control speech" \
    --dataset tau-voice-regular-test "tau-voice: tau2's test split (100), regular speech, the leaderboard's condition" \
    --dataset tau-voice-parity-control "tau-voice: the 40-task parity sample, control speech" \
    --dataset tau-voice-parity-regular "tau-voice: the 40-task parity sample, regular speech" \
    --dataset eva "EVA: all 213 records, clean audio" \
    --dataset eva-parity "EVA: the 60-record parity sample, clean audio"
uv run python scripts/hub.py check --datasets-repo ../harbor-voice-datasets
```

- `register` refuses, writing nothing, if the commit isn't on the remote's
  `main` and tagged, if a name and version are already published, if a file
  is stray, or if an image is unpinned or unrecorded.
- Commit `registry.json`. CI checks that entries are only ever added.
- Once the images are public, `check --anonymous` also pulls each one
  without credentials.

## Releasing

harbor-voice and its two adapters (`harbor-voice-tau-voice-adapter`,
`harbor-voice-eva-adapter`) are released together, at one version: a trial
installs a driver at harbor-voice's own version.

1. Set the version in `pyproject.toml`, `adapters/tau-voice/pyproject.toml`
   and `adapters/eva/pyproject.toml`, and in harbor-voice's `tau-voice` and
   `eva` extras. Run `uv lock`.
2. Check what would be published:

   ```bash
   uv build --all-packages --out-dir dist
   uv run python scripts/check_dist.py dist --tag v<version>
   ```

   It fails on a file outside the packages' own code, metadata and licenses,
   on anything that looks like a credential, a local path or an email
   address, and on versions that disagree. CI runs it on every PR.
3. Merge to `main`. Optionally, run the Release workflow by hand first: it
   publishes to TestPyPI.
4. Tag the merge commit and push the tag:

   ```bash
   git tag v<version> && git push origin v<version>
   ```

   The Release workflow tests, builds, checks and publishes all three
   packages to PyPI with trusted publishing, then creates the GitHub release.
   A version on PyPI can't be replaced, only yanked.

## Issues and PRs

- **Issues** are for bugs and feature requests. Describe the problem; you
  don't need to propose a fix.
- **PRs:** when you open one from a fork, enable "Allow edits by
  maintainers", so we can make small fixes directly.
- **Interfaces and core logic:** open an issue to discuss the design before
  you change them. That includes the transport contract, ATIF-voice and the
  driver contract.

By contributing, you agree that your contribution is licensed under the
project's [Apache-2.0 license](LICENSE).
