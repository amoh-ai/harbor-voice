# harbor-voice

**Harbor's infrastructure, extended for voice agents.** Run voice benchmarks
as live phone calls, one [Harbor](https://github.com/harbor-framework/harbor)
trial per call, on your laptop or across thousands of cloud sandboxes.

[![PyPI](https://img.shields.io/pypi/v/harbor-voice)](https://pypi.org/project/harbor-voice/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://github.com/amoh-ai/harbor-voice/blob/main/pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/amoh-ai/harbor-voice/blob/main/LICENSE)
[![Built on harbor](https://img.shields.io/badge/built%20on-harbor-105864)](https://harborframework.com/docs)

![harbor-voice: benchmarks run as Harbor trials, one per call. Inside each, a simulated caller built in six layers talks to the agent under test over a live call loop, with the task's tools and grader beside it. Every call leaves an ATIF-voice trajectory, the recording and a grade, used as an evaluation or as an RL rollout, on any of Harbor's backends.](https://raw.githubusercontent.com/amoh-ai/harbor-voice/main/docs/assets/cover.png)

Voice benchmarks each ship their own runner, their own simulated caller and
their own output format. harbor-voice runs them all the same way. Each call
is a Harbor trial: an isolated environment with a live, full-duplex audio
session between a simulated caller and the agent under test, the task's
tools and grader, and a timed trajectory of everything that happened.

Read the launch post:
[Introducing harbor-voice](https://amoh.ai/blogs/harbor-voice).

## Highlights

- **Voice benchmarks, out of the box.** [τ-Voice](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/README.md)
  (278 tasks) and [EVA](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/eva/README.md)
  (213 tasks), published as pinned datasets. Both pass Harbor's parity check
  against their own harnesses.
- **Real calls, not text turns.** Audio streams both ways in 20 ms frames, in
  real time. The agent can be interrupted, talk over the caller, and call
  tools mid-sentence.
- **A caller in six layers.** Persona, speech, room, line, conversation and
  the agent's tools. Each layer is seeded, and the trajectory records what it
  was asked to do and what it actually did.
- **One record per call.** ATIF-voice: valid ATIF for any ATIF reader, with
  the call's timeline, both sides of each turn, the conditions and the
  validity of the measurement.
- **Any agent.** Speech-to-speech models from OpenAI, Google, Amazon, xAI,
  ElevenLabs and Deepgram, or a cascade you build from speech to text, an
  LLM and text to speech. Self-hosted servers too.
- **Anywhere Harbor runs.** Docker on your machine, or Daytona, Modal, EC2,
  GKE, Vercel, Prime Intellect and Docker Sandboxes.
- **Evaluation or RL.** The same trial gives a score, or a rollout
  (trajectory and reward) for a trainer.
- **Bring your own benchmark.** An adapter supplies the tasks, the caller and
  the grader.

## Contents

- [Install](#install)
- [Try it without API keys](#try-it-without-api-keys)
- [Run a benchmark](#run-a-benchmark)
- [How a call runs](#how-a-call-runs)
- [The six layers](#the-six-layers)
- [ATIF-voice](#atif-voice)
- [Parity](#parity)
- [Providers](#providers)
- [Running on cloud](#running-on-cloud)
- [Evaluation and RL rollouts](#evaluation-and-rl-rollouts)
- [What a run leaves behind](#what-a-run-leaves-behind)
- [CLI](#cli)
- [Development](#development)
- [License](#license)

## Install

You need Python 3.12+ and Docker.

**From PyPI:**

```bash
pip install "harbor-voice[tau-voice,eva]"
```

The `tau-voice` and `eva` extras add each benchmark's commands; the library
alone is `pip install harbor-voice`. Each trial installs the same versions
into its environment.

**From source**, to convert tasks or change the code (needs
[uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/amoh-ai/harbor-voice
cd harbor-voice
uv sync   # harbor-voice, both adapters, all extras, dev tools
```

Each trial then installs harbor-voice and the adapter from these source
folders, so a change runs as it is. Prefix the commands below with `uv run`.

Optional extras, if you depend on harbor-voice as a library:

| Extra | For |
|---|---|
| `live` | OpenAI Live, over WebRTC |
| `aws` | Amazon Nova Sonic and the Bedrock providers |
| `google` | Gemini Live on Vertex AI |
| `daytona`, `ec2`, `gke`, `langsmith`, `modal`, `prime`, `sbx`, `vercel` | Harbor's cloud backend of that name |

## Try it without API keys

The library has offline providers that need no keys and no network. A
two-turn call on them takes a few seconds and writes the recording, the
metrics and a validity report:

```toml
# my-task/voice.toml
profile = "telephony"
seed = 7
duplex = "half"
max_turns = 2
task = "intake"

[agent]
pipeline = "cascade"
stt = "offline.scripted-stt"
llm = "offline.echo-llm"
tts = "offline.tone-tts"

[agent.config.stt]
turns = ["I need to move my appointment", "Tuesday works"]

[caller]
source = "script"
turns = ["I need to move my appointment", "Tuesday works"]
```

```python
import asyncio
from pathlib import Path

from harbor_voice.trial import VoiceTaskConfig, VoiceTrial

config = VoiceTaskConfig.from_toml(Path("my-task/voice.toml"))
outcome = asyncio.run(VoiceTrial(config, out_dir=Path("out")).run())
print(outcome.reward)  # the call's metrics, also in out/reward.json
Path("out/trajectory.json").write_text(outcome.trajectory.to_atif_json(), encoding="utf-8")
```

Then `harbor-voice show out/trajectory.json` summarises the call and
`harbor-voice listen out/trajectory.json --out call.wav` writes its
recording. More in [docs/library.md](https://github.com/amoh-ai/harbor-voice/blob/main/docs/library.md#the-built-in-call-loop).

## Run a benchmark

Each benchmark needs its API keys in your environment, and a one-time setup
of its callers in your ElevenLabs account: see the
[tau-voice](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/README.md) and [EVA](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/eva/README.md)
READMEs. Every call is a live session with paid APIs, so start with one call.
Published datasets run without converting anything:

```bash
tau-voice job-config --hub-dataset tau-voice-control@0.1.2 \
    --task-names airline-0-d9c07603 --trials 1 -n 1 \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    -o one.yaml
harbor-voice jobs start -c one.yaml
```

Then a full job: τ-Voice's leaderboard condition, 3 trials per task with 4
calls at a time, or all of EVA:

```bash
tau-voice job-config --hub-dataset tau-voice-regular-test@0.1.2 \
    --trials 3 -n 4 --model openai.live/gpt-live-1 \
    --agent-config '{"backend_model": "gpt-6-luna"}' -o tau.yaml
eva-adapter job-config --hub-dataset eva@0.1.2 \
    --model openai.realtime/gpt-realtime-1.5 -o eva.yaml
harbor-voice jobs start -c tau.yaml
```

- **The agent.** `openai.live/gpt-live-1` is OpenAI Live: `gpt-live-1` speaks
  and listens, and the `backend_model` behind it reasons and calls tools. For
  OpenAI Realtime, use `--model openai.realtime/gpt-realtime-1.5` with no
  `--agent-config`. On τ-Voice, `tau-voice job-config --cascade` makes a
  speech-to-text, LLM and text-to-speech cascade the agent instead.
- **Concurrency.** Calls run in real time, a few minutes each, so `-n` sets a
  job's wall time. Keep it within your providers' limits on concurrent
  sessions (ElevenLabs' especially), or calls are refused.

### Benchmarks

- **[tau-voice](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/README.md)**: τ-Voice, the voice mode of
  Sierra's [tau3-bench](https://github.com/sierra-research/tau2-bench).
  Customer-service calls about airline, retail and telecom tasks.
- **[EVA](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/eva/README.md)**: ServiceNow's
  [EVA](https://github.com/ServiceNow/eva). Enterprise calls in airline, IT
  service and medical HR.

Each adapter's README covers its tasks, agents, caller conditions and parity
with the original harness. To add a benchmark, see
[CONTRIBUTING.md](https://github.com/amoh-ai/harbor-voice/blob/main/CONTRIBUTING.md#adding-a-benchmark).

### Datasets

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

[`registry.json`](https://github.com/amoh-ai/harbor-voice/blob/main/registry.json) lists the datasets. Their tasks are in
[harbor-voice-datasets](https://github.com/amoh-ai/harbor-voice-datasets) and
their images on `ghcr.io/amoh-ai`, both pinned, so a version always runs the
same thing. Published versions: 0.1.0, 0.1.1 and 0.1.2, the latest. The
datasets repo's README says what changed in each and which images each pins.
From 0.1.1 the images are built for x86 and ARM; each adapter's README lists
the current tags. Without `@VERSION`, harbor picks the latest. For a condition or
subset that isn't published, convert the tasks yourself from a checkout: each
adapter's README shows how.

## How a call runs

```
harbor-voice jobs start          one Harbor trial per call, in its own environment
└── trial
    ├── main: harbor-voice
    │   ├── simulated caller     speaks as the task defines: its voices, room, line and timing
    │   └── call loop            audio both ways in real time, holds the floor, times every frame
    │         ⇅ audio, tool calls
    │       agent under test     a speech-to-speech model, or a cascade
    └── sidecar: the task        its tools, the caller's prompts and the grader
```

- **The call loop** moves 20 ms audio frames between the caller and the agent
  in real time, decides who holds the floor, and timestamps every frame. A
  transport is always full-duplex; half duplex is a policy on top.
- **The agent's tool calls** go through the call loop to the task's tools in
  the sidecar, which also grades the call at the end.
- **Harbor** does the rest: parallel trials, retries, resuming an interrupted
  job, and the backend each trial runs on.

The benchmark adapters bring their own caller and call loop (τ-Voice's and
EVA's, unchanged). A task that declares a `[caller]` instead runs on
harbor-voice's own built-in loop, with the six layers below.

## The six layers

The built-in caller is one simulator with one interface. Its turns come from
a script, an LLM playing a persona, or recordings. Each layer is seeded and
recorded with what it achieved, not just what it was asked for, and the clean
audio is kept beside what the agent heard.

| Layer | What it changes | For example |
|---|---|---|
| L1 Persona | who speaks | the voice, an accent, a persona brief |
| L2 Speech | how they speak | vocal tics, speaking rate, microphone muffling |
| L3 Environment | the room | a continuous background bed at a target SNR |
| L4 Channel | the line | μ-law at 8 kHz, frame loss in bursts, jitter |
| L5 Conversation | when they speak | interruptions, backchannels, speech not meant for the agent |
| L6 World | the agent's tools | latency, errors, outages (not the caller's audio) |

Named profiles set the layers for a condition (`harbor-voice profiles` lists
them, with the metrics each moves):

| Profile | The caller |
|---|---|
| `clean` | no perturbation: the baseline every robustness delta is measured against |
| `accented` | an accented caller on a clean line |
| `accented_noisy` | an accented caller in a noisy room |
| `background_noise` | a noisy room: a background bed at 15 dB SNR |
| `noisy_line` | a noisy room over a bad line |
| `telephony` | a phone line: μ-law 8 kHz with burst loss |
| `overlap` | a caller who interrupts and backchannels, on a clean line |
| `selectivity_probes` | sounds the agent should hear and keep talking through: a backchannel, a vocal tic, speech not directed at it |

`harbor-voice report <job>` pairs each call under a profile with the same
call under `clean`, so the change it reports comes from the caller's
conditions alone.

## ATIF-voice

![One call as ATIF-voice: the plain ATIF steps on the left; on the right, extra.voice with the call's timeline (caller, agent and tool lanes, a find_slots call that answered 6 seconds late), the first turn's said and heard text, the perturbation plan with each layer requested and achieved, and validity.](https://raw.githubusercontent.com/amoh-ai/harbor-voice/main/docs/assets/atif.png)

Harbor records agents in ATIF, a trajectory built from steps. A call needs
more: two parties can talk at once, interrupt each other, and call tools
while audio is still flowing. ATIF-voice (`ATIF-v1.8+voice.1`) is a genuine
ATIF trajectory, readable by any ATIF consumer, with the call carried in
`extra.voice`:

- **`timeline`**: every speech start and end, overlap, barge-in, tool call
  and transport gap, in seconds since the session opened.
- **`turns`**: both sides of each turn, what the caller said, what the agent
  heard and what it replied, with clock stamps and dropped frames.
- **`perturbation_plan`**: the seeds, and each layer's settings, requested
  and achieved.
- **`validity`**: whether the measurement holds. A call where the measurement
  or recording failed is marked invalid, never scored 0, so infrastructure
  failures stay apart from agent failures.
- **`usage`**: tokens, characters and billed seconds for the agent, the
  caller and the judges.
- **`artifacts`**: the recording, per party and mixed, referenced by content
  hash, never inlined.

The metrics read from this record: turn timing, overlap and barge-in, the
word error rate between what the caller said and what the agent heard, and
run validity. A metric abstains rather than scoring 0 when a call can't
support it (`harbor-voice metrics` lists them).

## Parity

To check harbor-voice, each benchmark ran through its own harness and
through harbor-voice on the same tasks, with the same agent, callers, voices,
seeds and grader, three runs a side. Harbor's parity criterion holds on every
row: all 8 for τ-Voice and all 12 for EVA.

| Benchmark | Score | Own harness | harbor-voice |
|---|---|---|---|
| τ-Voice, `control` caller | average reward, 40 tasks | 55.0 ± 1.4 | 53.3 ± 6.0 |
| τ-Voice, `regular` caller | average reward, 40 tasks | 48.3 ± 1.7 | 47.5 ± 1.4 |
| EVA | EVA-A pass rate, 60 records | 48.3 ± 4.4 | 46.1 ± 3.9 |
| EVA | EVA-X pass rate, 60 records | 65.6 ± 3.4 | 58.3 ± 3.3 |

Means ± SEM over 3 runs, in %. EVA-X is lower through harbor-voice, and the
gap is in turn-taking. The reports explain why, and give each domain, the
exact commands and how to reproduce it:
[τ-Voice PARITY.md](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/PARITY.md) and
[EVA PARITY.md](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/eva/PARITY.md).

## Providers

The agent under test is a speech-to-speech model, or a cascade of parts.
`harbor-voice caps` prints each provider's capabilities and which credentials
are missing.

| Kind | Providers |
|---|---|
| Speech to speech | `openai.realtime`, `openai.live`, `aws.nova-sonic`, `elevenlabs.agents`, `google.live`, `xai.realtime`, `deepgram.agent`, `selfhosted.realtime` |
| Speech to text, for cascades | `aws.nova-stt`, `aws.voxtral`, `deepgram.stt`, `cartesia.stt`, `sarvam.stt`, `fishaudio.stt`, `xai.stt`, `meta.stt`, `microsoft.stt`, `openrouter.stt`, `selfhosted.stt` |
| LLM, for cascades | `openai.llm`, `aws.bedrock-llm`, `litellm.llm`, `selfhosted.llm` |
| Text to speech, for cascades | `elevenlabs.tts`, `aws.nova-tts`, `deepgram.tts`, `cartesia.tts`, `sarvam.tts`, `fishaudio.tts`, `xai.tts`, `openrouter.tts`, `selfhosted.tts` |
| Omni (audio in, text out), in place of STT and LLM | `openrouter.omni`, `selfhosted.omni` |
| Offline, for tests | `offline.scripted-stt`, `offline.echo-llm`, `offline.scripted-llm`, `offline.tool-echo-llm`, `offline.tone-tts` |

`litellm.llm` reaches Gemini, Vertex AI, xAI, Bedrock, OpenRouter, and any
vLLM, SGLang or Ollama server. The `selfhosted.*` providers call your own
OpenAI-compatible servers (vLLM, vLLM-Omni, SGLang), so an open model, or one
you are training, can be the agent. Each provider's sign-in and options are
in [docs/library.md](https://github.com/amoh-ai/harbor-voice/blob/main/docs/library.md#providers).
To add one, see [CONTRIBUTING.md](https://github.com/amoh-ai/harbor-voice/blob/main/CONTRIBUTING.md#adding-a-provider).

## Running on cloud

The orchestrator stays on your machine. The backend creates one environment
per trial and deletes it afterwards. Choose it with `job-config --env <type>`:

- **Daytona.** Set `DAYTONA_API_KEY` and pass `--env daytona`.
- **EC2.** harbor launches a VM per trial and reaches it over SSH; your
  account's settings go in the job config.
- **Modal, Docker Sandboxes, GKE, Prime Intellect, Vercel.** `--env modal`,
  `sbx`, `gke`, `prime` or `vercel`, with that service's sign-in.

Setup for each, and prebuilt images for faster startup:
[docs/cloud.md](https://github.com/amoh-ai/harbor-voice/blob/main/docs/cloud.md).
Harbor's other integrations work the same way.

## Evaluation and RL rollouts

![The same trials serve both workflows: as an evaluation, rewards become scores compared across agents and conditions on the same seeded calls; as RL rollouts, trajectories and rewards go to your trainer, and the updated model comes back behind a provider for the next round of calls.](https://raw.githubusercontent.com/amoh-ai/harbor-voice/main/docs/assets/rollouts.png)

Each trial produces a trajectory and the task's own reward. As an evaluation,
the reward scores the agent. As an RL rollout, the trajectory and reward go
to a trainer: ATIF's steps can carry `prompt_token_ids`,
`completion_token_ids` and `logprobs`, and each voice turn can point at its
token span. Because the caller's conditions are seeded and recorded, hard
calls can be kept in training and replayed later.

harbor-voice does not train. To collect rollouts from your own model, serve
it behind a provider, such as `selfhosted.realtime` or `selfhosted.llm`.

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
| `harbor-voice caps` | provider capability matrices, and which credentials are missing |
| `harbor-voice profiles` | the named caller conditions, their layers, and the metrics each moves |
| `harbor-voice metrics` | the registered metrics, what they need, and which layers move them |
| `harbor-voice doctor` | preflight a configuration: what will report, abstain, or be blocked |
| `harbor-voice report <job>` | roll up built-in-loop trials and diff each condition against a baseline |
| `harbor-voice drive <task>` | run a task's call driver in this process; what harbor-voice's agent runs inside a trial's environment |
| `tau-voice` | `job-config`, `voices`; from a checkout, converting tasks, `build-image`, and parity's `sample` and `compare` |
| `eva-adapter` | `job-config`, `agents`, `judges`; from a checkout, `convert`, `build-image`, and parity's `sample` and `compare` |

`harbor view`, harbor's viewer, shows a call as text.

## Development

```
src/harbor_voice/     the library: providers, the call loop, ATIF-voice, metrics, the CLI
adapters/tau-voice/   the tau-voice adapter (its own package)
adapters/eva/         the EVA adapter (its own package)
tests/                the library's tests (unit/) and each adapter's (adapters/)
docs/                 the library, cloud backends, and this README's figures
```

From a source install:

```bash
uv run pytest
uv run ruff check src adapters scripts tests && uv run ruff format src adapters scripts tests
uv run ty check
```

Tests that need Docker and the benchmarks' images skip without them; CI runs
the rest on Linux and Windows. To contribute, add a provider or add a
benchmark, see [CONTRIBUTING.md](https://github.com/amoh-ai/harbor-voice/blob/main/CONTRIBUTING.md).
What changed in each release: [CHANGELOG.md](https://github.com/amoh-ai/harbor-voice/blob/main/CHANGELOG.md).
To report a vulnerability: [SECURITY.md](https://github.com/amoh-ai/harbor-voice/blob/main/SECURITY.md).

## Acknowledgements

harbor-voice is built on [Harbor](https://github.com/harbor-framework/harbor)
and runs Sierra's [tau2-bench](https://github.com/sierra-research/tau2-bench)
and ServiceNow's [EVA](https://github.com/ServiceNow/eva) through their own
code. Thanks to their authors.

## License

Apache-2.0; see [LICENSE](https://github.com/amoh-ai/harbor-voice/blob/main/LICENSE). harbor-voice builds on harbor and
depends on it as a package. The adapters run tau2-bench and EVA in
containers at pinned commits and don't redistribute them, except the parts
copied into the repo, which are under the MIT License: see
[tau2-bench's](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/src/tau_voice/tau2_audio/LICENSE) and
[EVA's](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/eva/src/eva_adapter/vendored/LICENSE) license files.
