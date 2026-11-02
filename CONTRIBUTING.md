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
uv sync          # harbor-voice, both adapters, every extra, dev tools
```

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
3. Run `harbor-voice conformance <provider>` to verify what it declares.

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

[tau-voice](adapters/tau-voice) and [EVA](adapters/eva) are the two worked
examples.

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
