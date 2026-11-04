# harbor-voice

**Evaluate voice agents on phone-call benchmarks, on
[harbor](https://github.com/harbor-framework/harbor).**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Built on harbor](https://img.shields.io/badge/built%20on-harbor-105864)](https://harborframework.com/docs)

harbor-voice evaluates voice agents on simulated phone calls. You can use
harbor-voice to:

- Run voice benchmarks like [τ-Voice](adapters/tau-voice/README.md) and
  [EVA](adapters/eva/README.md) against speech-to-speech agents.
- Test agents under realistic callers: accents, background noise,
  interruptions.
- Run calls in parallel, locally or in the cloud.
- Keep every call: the recording, a trajectory and the grade.
- Add your own voice benchmark.

## Benchmarks

- **[tau-voice](adapters/tau-voice/README.md)**: τ-Voice, the voice mode of
  Sierra's [tau3-bench](https://github.com/sierra-research/tau2-bench).
  Customer-service calls about airline, retail and telecom tasks.
- **[EVA](adapters/eva/README.md)**: ServiceNow's
  [EVA](https://github.com/ServiceNow/eva). Enterprise calls in airline, IT
  service and medical HR.

More voice benchmarks are on the way.

Each adapter's README covers its tasks, agents, caller conditions and parity
with the original harness. Support for more agents on both benchmarks is on
the way; other agents run on the [library](docs/library.md)'s built-in call
loop.

## Install

You need Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/amoh-ai/harbor-voice
cd harbor-voice
uv sync   # harbor-voice, both adapters, all extras, dev tools
```

Run harbor-voice from this checkout: each trial installs harbor-voice and the
adapter into its environment from these source folders. harbor is a pinned
dependency (`harbor==0.25.0`).

## Run a benchmark

Each benchmark needs its API keys in your environment, and a one-time setup
of its callers in your ElevenLabs account: see the
[tau-voice](adapters/tau-voice/README.md) and [EVA](adapters/eva/README.md)
READMEs. Every call is a live session with paid APIs, so start with one call.
Published datasets run without converting anything:

```bash
uv run tau-voice job-config --hub-dataset tau-voice-control@0.1.0 \
    --task-names airline-0-d9c07603 --trials 1 -n 1 \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    -o one.yaml
uv run harbor-voice jobs start -c one.yaml
```

Then a full job: τ-Voice's leaderboard condition, 3 trials per task with 4
calls at a time, or all of EVA:

```bash
uv run tau-voice job-config --hub-dataset tau-voice-regular-test@0.1.0 \
    --trials 3 -n 4 --model openai.live/gpt-live-1 \
    --agent-config '{"backend_model": "gpt-6-luna"}' -o tau.yaml
uv run eva-adapter job-config --hub-dataset eva@0.1.0 \
    --model openai.realtime/gpt-realtime-1.5 -o eva.yaml
uv run harbor-voice jobs start -c tau.yaml
```

- **The agent.** `openai.live/gpt-live-1` is OpenAI Live: `gpt-live-1` speaks
  and listens, and the `backend_model` behind it reasons and calls tools. For
  OpenAI Realtime, use `--model openai.realtime/gpt-realtime-1.5` with no
  `--agent-config`.
- **Concurrency.** Calls run in real time, a few minutes each, so `-n` sets a
  job's wall time. Keep it within your providers' limits on concurrent
  sessions (ElevenLabs' especially), or calls are refused.

| Dataset | Tasks | What |
|---|---|---|
| `tau-voice-control` | 278 | every task, `control` caller |
| `tau-voice-regular` | 278 | every task, `regular` caller |
| `tau-voice-control-test` | 100 | tau3-bench's `test` split, `control` caller |
| `tau-voice-regular-test` | 100 | the `test` split, `regular` caller: the leaderboard's condition |
| `tau-voice-parity-control` | 40 | the parity sample, `control` caller |
| `tau-voice-parity-regular` | 40 | the parity sample, `regular` caller |
| `eva` | 213 | every task, clean audio |
| `eva-parity` | 60 | the parity sample, 20 per domain |

[`registry.json`](registry.json) lists the datasets. Their tasks are in
[harbor-voice-datasets](https://github.com/amoh-ai/harbor-voice-datasets) and
their images on `ghcr.io/amoh-ai`, both pinned, so a version always runs the
same thing. Without `@VERSION`, harbor picks the latest. For a condition or
subset that isn't published, convert the tasks yourself: each adapter's
README shows how.

## Running on cloud

The orchestrator stays on your machine. The backend creates one environment
per trial and deletes it afterwards. Choose it with `job-config --env <type>`:

- **Daytona.** Set `DAYTONA_API_KEY` and pass `--env daytona`.
- **EC2.** harbor launches a VM per trial and reaches it over SSH; your
  account's settings go in the job config.

Setup for both, and prebuilt images for faster startup:
[docs/cloud.md](docs/cloud.md).

## What a run leaves behind

```
<trial>/
├── result.json             harbor's trial result: reward, timings, agent
├── agent/
│   ├── trajectory.json     the call, in ATIF-voice
│   └── call/
│       ├── outcome.json
│       └── <adapter>/      each attempt: the recording, the benchmark's own record and grade
└── verifier/               the reward (reward.txt for tau-voice, reward.json for EVA), result.json
```

## CLI

| Command | What it does |
|---|---|
| `harbor-voice jobs start` / `trials start` | harbor's runners, with the agent defaulting to harbor-voice's |
| `harbor-voice status <job>` | a running job: counts, reward so far, each trial's phase |
| `harbor-voice show <trajectory>` | summarise a call: turns, timeline, validity |
| `harbor-voice listen <trajectory> --out call.wav` | write a call's recording to a WAV |
| `tau-voice` | convert tasks; `job-config`, `voices`, `build-image`; for parity, `sample` and `compare` |
| `eva-adapter` | `convert`, `job-config`, `agents`, `build-image`; for parity, `sample` and `compare` |

`harbor view`, harbor's viewer, shows a call as text. The library's own
commands are in [docs/library.md](docs/library.md).

## Development

```
src/harbor_voice/     the library: providers, the call loop, ATIF-voice, metrics, the CLI
adapters/tau-voice/   the tau-voice adapter (its own package)
adapters/eva/         the EVA adapter (its own package)
tests/                the library's tests (unit/) and each adapter's (adapters/)
```

```bash
uv run pytest
uv run ruff check src adapters tests && uv run ruff format src adapters tests
uv run ty check
```

Tests that need Docker and the benchmarks' images skip without them; CI runs
the rest on Linux and Windows. To contribute, or to add a benchmark, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0; see [LICENSE](LICENSE). harbor-voice builds on harbor and
depends on it as a package. The adapters run tau2-bench and EVA in
containers at pinned commits and don't redistribute them, except the parts
copied into the repo, which are under the MIT License: see
[tau2-bench's](adapters/tau-voice/src/tau_voice/tau2_audio/LICENSE) and
[EVA's](adapters/eva/src/eva_adapter/vendored/LICENSE) license files.
