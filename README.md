# harbor-voice

**Voice-agent evaluation on [harbor](https://github.com/harbor-framework/harbor).**
harbor-voice puts a voice agent on a phone call with a simulated caller,
records the call, and grades it with the benchmark's own grader. It runs
through harbor's job runner and environments, so the same job runs on your
laptop's Docker or in the cloud.

> **Status: pre-release (0.1.0).** Two benchmark adapters, tau-voice and EVA,
> run end to end through harbor's runner. Each has parity results against its
> original harness, from runs on Daytona.

## Benchmarks

| | [tau-voice](adapters/tau-voice) | [EVA](adapters/eva) |
|---|---|---|
| Benchmark | [tau2-bench](https://github.com/sierra-research/tau2-bench)'s full-duplex voice mode, at `2174a60` | ServiceNow's [EVA](https://github.com/ServiceNow/eva), at `039efa0a` |
| Tasks | 278: airline 50, retail 114, telecom 114 | 213 records: airline 50, itsm 80, medical_hr 83 |
| Agent under test | OpenAI Realtime, OpenAI Live | OpenAI Realtime |
| Caller | tau2's: an LLM for the words, ElevenLabs for the voice, tau2's phone line and audio effects | EVA's: two ElevenLabs Agents (one female, one male English caller) |
| Conditions | tau2's speech complexities: `control`, `regular` (the leaderboard's) and six ablations | clean, or EVA's audio perturbations: eight background noises at any SNR, connection degradation |
| Grade | tau2's evaluator | EVA's metrics and judges: EVA-A, EVA-X, overall |
| Parity | 40 tasks × 3 runs, OpenAI Live (`gpt-live-1`, backend `gpt-6-luna`), both modes: the criterion holds on all 8 rows ([PARITY.md](adapters/tau-voice/PARITY.md)) | 60 records × 3 runs, OpenAI Realtime (`gpt-realtime-1.5`): the criterion holds on all 12 rows ([PARITY.md](adapters/eva/PARITY.md)) |

In both, the benchmark's own code runs in a sidecar container: the task, its
tools, the caller's prompts and the grading. harbor-voice runs the call
itself, the way the original harness does: tau2's 0.2 s step loop, or EVA's
audio bridge. Each adapter's README covers its keys, options and outputs.

## How it fits harbor

harbor-voice adds no runner of its own. A voice task is a harbor task, and
the call runner is a harbor agent, `VoiceCallAgent`:

```
harbor-voice jobs start -c job.yaml          (harbor's job runner, mounted)
  └─ per trial, a harbor environment (docker | daytona | …)
       ├─ main     ← VoiceCallAgent installs harbor-voice and the adapter here,
       │             then runs the task's call driver: the benchmark's call loop
       └─ sidecar  ← the benchmark's own code, from the task's docker-compose.yaml
  └─ harbor's verifier grades it; harbor collects logs and trajectory.json
```

- **What you get from harbor:** concurrency, retries, resume, job metrics and
  its environment backends, exactly as for any harbor agent.
  `harbor-voice jobs` and `harbor-voice trials` *are* `harbor jobs` and
  `harbor trials`. When no agent is given, the agent defaults to
  `VoiceCallAgent`.
- **The agent under test is the run's choice, never the task's:**
  `-m provider/model`, e.g. `openai.live/gpt-live-1` (tau-voice's parity
  agent) or `openai.realtime/gpt-realtime-1.5` (EVA's).
- **Each benchmark brings its own call loop** (the `[call] driver` in a
  task's `voice.toml`). Turn-taking, audio pacing and when a call ends are
  part of a benchmark's definition, so parity needs its exact loop.

## Install

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker for local
runs.

```bash
git clone https://github.com/amoh-ai/harbor-voice
cd harbor-voice
uv sync          # harbor-voice, both adapters, every extra, dev tools
```

- **Run it from this checkout.** At trial start, `VoiceCallAgent` builds
  harbor-voice and the adapter from their source folders and installs them
  into the environment. An install from a package registry can't do that yet.
- **harbor is a pinned dependency** (`harbor==0.25.0`).

The extras, if you install the package on its own:

| Extra | For |
|---|---|
| `live` | OpenAI Live over WebRTC |
| `aws` | Amazon Nova Sonic and Bedrock providers |
| `daytona` | harbor's Daytona backend |
| `ec2` | harbor's EC2 backend |

## Quickstart: tau-voice

**1. Keys.** Set these in your environment (or source a `.env`). Tasks hold
only `${VAR}` templates, and harbor fills them from your environment at trial
start. The full table is in the
[tau-voice README](adapters/tau-voice/README.md).

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | the agent under test, the caller's LLM |
| `ELEVENLABS_API_KEY` | the caller's voice |
| `TAU2_VOICE_ID_<PERSONA>` | one ElevenLabs voice per tau2 persona, made by tau2's `setup_voices`: two for `control`, five more for `regular` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | tau2's hallucination reviewer, on Bedrock |

**2. Convert tasks** from a checkout of tau2-bench, which supplies the task
lists:

```bash
uv run tau-voice --tau2-root ../tau2-bench --output-dir datasets/tau-voice
# or a few: --domains airline --task-ids 0 1 2; a condition: --complexity regular
```

**3. Run one call** on local Docker. The agent is OpenAI Live, as in the
parity run: `gpt-live-1`, a speech frontend, delegating to a backend model
(`--agent-config`). For OpenAI Realtime, use
`--model openai.realtime/gpt-realtime-1.5` and no `--agent-config`.

```bash
uv run tau-voice job-config --dataset datasets/tau-voice \
    --task-names airline-0-d9c07603 --trials 1 -n 1 \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    -o one.yaml
uv run harbor-voice jobs start -c one.yaml
```

**4. Run a job.** `job-config` writes a harbor job that runs the tasks the way
tau2 runs its trials. It lists the agent once per tau2 trial seed, retries
infrastructure failures (not agent failures), and leaves errored trials out
of the mean:

```bash
uv run tau-voice job-config --dataset datasets/tau-voice --trials 3 -n 4 \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    -o job.yaml
uv run harbor-voice jobs start -c job.yaml
```

## Quickstart: EVA

**1. Keys and caller agents.** The full table, including the optional judge
settings, is in the [EVA README](adapters/eva/README.md).

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | the agent under test, EVA's GPT judges |
| `ELEVENLABS_API_KEY` | the caller |
| `EVA_EN_USER_F`, `EVA_EN_USER_M` | the IDs of EVA's two caller agents, in your ElevenLabs account |
| `EVA_MODEL_LIST` | EVA's judge deployments: `EVA_MODEL_LIST="$(cat adapters/eva/judges.json)"` |
| `GEMINI_API_KEY` | the audio judges (Gemini) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | the faithfulness judge, on Bedrock |

The caller agents don't exist until you create them. This writes two agents
to your ElevenLabs account, configured as EVA's setup defines them:

```bash
uv run eva-adapter agents show                      # what would be created (offline)
uv run eva-adapter agents create --agent en_f --yes # its ID goes in EVA_EN_USER_F
uv run eva-adapter agents create --agent en_m --yes # its ID goes in EVA_EN_USER_M
```

**2. Convert, run one call, run the job.** The agent is OpenAI Realtime
(`gpt-realtime-1.5`), as in the parity run; it's the only agent EVA's
adapter takes.

```bash
uv run eva-adapter convert --out datasets/eva  # also writes datasets/eva/job.yaml
uv run eva-adapter job-config --dataset datasets/eva \
    --task-names airline-1-1-2-17dc1179 --trials 1 -n 1 \
    --model openai.realtime/gpt-realtime-1.5 -o one.yaml
uv run harbor-voice jobs start -c one.yaml       # one call
uv run harbor-voice jobs start -c datasets/eva/job.yaml
```

A perturbed condition gets its own dataset folder, e.g.
`eva-adapter convert --out datasets/eva-coffee_shop-15db --background-noise coffee_shop --snr-db 15`.

## Running on cloud

The orchestrator stays on your machine. The backend creates one environment
per trial and deletes it afterwards. Choose the backend with `job-config --env
<type>` (both adapters), or `--env` on `trials start`. Settings that belong to
your account go in your own copy of the job config, never in the tasks.

- **Backends.** The tasks use docker-compose, because the benchmark runs in a
  sidecar. So they need a backend that supports it. The parity runs used
  Daytona.
- **Daytona.** Set `DAYTONA_API_KEY`. Each trial is a Docker-in-Docker sandbox
  sized from the task.

  ```bash
  uv run tau-voice job-config ... --env daytona -o job.yaml
  uv run harbor-voice jobs start -c job.yaml
  ```

- **EC2.** Sign in with any standard AWS credential (a profile, SSO,
  environment variables). harbor launches a VM per trial, connects over SSH
  and terminates it afterwards. It needs a key pair and a security group that
  allows SSH from your machine. Add your account's settings to the job
  config:

  ```yaml
  environment:
    type: ec2
    kwargs:
      region: us-east-1
      aws_profile: <your profile>
      ami_id: <an Ubuntu 24.04 AMI>
      instance_type: m7i-flex.large
      key_name: <your key pair>
      ssh_key_path: ~/.ssh/<your key>.pem
      security_group_ids: [<sg allowing SSH from you>]
  ```

- **Faster startup.** By default each environment builds the task's images,
  which takes minutes on a fresh cloud sandbox. Build them once and push them
  to a registry the backend can pull from, then convert against it:
  `tau-voice build-image --registry <prefix> --push` with
  `tau-voice --registry <prefix>`, or `eva-adapter build-image --registry
  <prefix> --push` with `eva-adapter convert --registry <prefix>`.

Calls run in real time, a few minutes each, so concurrency (`-n`) sets a
job's wall time. Keep it within your providers' concurrent-session limits
(ElevenLabs' especially), or calls get refused.

## What a trial leaves behind

```
<trial>/
├── result.json                  harbor's trial result: reward, timings, agent
├── agent/
│   ├── trajectory.json          ATIF-voice: the attempt that counts
│   └── call/
│       ├── outcome.json
│       └── <adapter>/…          per attempt: the stereo call (caller | agent),
│                                the benchmark's own record and grade, usage
└── verifier/                    reward.txt, result.json
```

- **The trajectory** is a harbor ATIF `Trajectory` with a voice payload: a
  timeline of speech, overlaps, barge-ins and tool calls, the turns with what
  was said and what was heard, and audio referenced by content hash, never
  inlined.
- **Reruns.** Both benchmarks rerun a call their own rules reject, up to
  their own limits: tau2 when its reviewer finds the *caller* hallucinated,
  EVA when its validation fails. Every attempt is kept, and the last one
  counts.
- **Reading a call:** `harbor-voice show <trajectory.json>` summarises it,
  and `harbor-voice listen <trajectory.json>` renders it to a WAV.
  `harbor view` shows the call as text; it has no audio or timeline view.

## The core library

`src/harbor_voice/` is usable beyond the two adapters.

- **Transports and providers.** A transport is always full-duplex audio
  frames, and half duplex is a policy on top. Optional operations
  (truncation, events, text) are declared per provider and checked, not
  probed. `harbor-voice conformance` tests a provider against that contract.

  | Kind | Providers |
  |---|---|
  | Speech to speech | `openai.realtime`, `openai.live`, `aws.nova-sonic`, `elevenlabs.agents` |
  | STT / LLM / TTS, for cascades | `aws.nova-stt`, `aws.voxtral` / `openai.llm`, `aws.bedrock-llm` / `elevenlabs.tts`, `aws.nova-tts` |
  | Offline, for tests | `offline.scripted-stt`, `offline.echo-llm`, `offline.scripted-llm`, `offline.tool-echo-llm`, `offline.tone-tts` |

- **A built-in call loop.** It has a seeded simulated caller (a script, a
  persona LLM or recorded clips), half- or full-duplex turn-taking, and
  perturbation layers, each recorded with what it achieved, not just what was
  asked for, with the clean audio kept for comparison:

  ```
  L1 PERSONA      who speaks          voice, persona brief
  L2 SPEECH       how they speak      disfluency, rate
  L3 ENVIRONMENT  the room            continuous background bed
  L4 CHANNEL      the wire            codec, loss, jitter, phone line
  L5 CONVERSATION when they speak     barge-in, backchannel, interjections
  L6 WORLD        the agent's tools   latency, errors, outages (not the caller's audio)
  ```

- **Voice metrics** that abstain rather than score 0 when a call can't support
  them: turn timing, overlap and barge-in, said-vs-heard WER, run validity.

**Not yet connected:**
- **The built-in loop can't run through harbor yet.** It runs from Python
  (`harbor_voice.trial.VoiceTrial`), but a task needs a `[call]` driver to
  run through harbor.
- **The adapters don't use it.** They run their benchmark's own loop and
  conditions instead.
- **Benchmark jobs aren't scored with the voice metrics.** They're graded by
  the benchmark only, and `harbor-voice report` can't roll them up.

## CLI

| Command | |
|---|---|
| `harbor-voice trials start` / `jobs start` | harbor's runners; the agent defaults to `VoiceCallAgent` |
| `harbor-voice status <job>` | a running job: counts, reward so far, each trial's phase |
| `harbor-voice show` / `listen <trajectory>` | summarise a recorded call; render it to a WAV |
| `harbor-voice caps` | provider capability matrices, and which credentials are missing |
| `harbor-voice conformance` | the provider conformance suite |
| `harbor-voice doctor` / `profiles` / `metrics` | the built-in loop's conditions and metrics: what a configuration will report, abstain on, or reject |
| `harbor-voice report <job>` | roll up built-in-loop trials and diff conditions against a baseline |
| `harbor-voice drive` | runs a task's call driver; what `VoiceCallAgent` runs inside `main` |
| `tau-voice …` | convert tasks; `build-image`, `job-config`; parity: `sample`, `compare` |
| `eva-adapter …` | `convert`, `agents`, `build-image`, `job-config`; parity: `sample`, `compare` |

## Limitations

- **Agents.** tau-voice runs OpenAI Realtime and OpenAI Live; EVA runs OpenAI
  Realtime only. There's no transport yet for testing your own deployed voice
  agent (a WebSocket endpoint, SIP or WebRTC).
- **Keys.** A benchmark needs keys from several vendors: OpenAI, ElevenLabs
  and AWS for tau-voice, and Gemini as well for EVA.
- **Source checkout only** (see Install).

## Repository

```
src/harbor_voice/        transports and providers, the built-in call loop and
                         layers, ATIF-voice, metrics, VoiceCallAgent, the CLI
adapters/tau-voice/      the tau-voice adapter (its own package)
adapters/eva/            the EVA adapter (its own package)
tests/unit/voice/        the package's tests
tests/adapters/          each adapter's tests and fixtures: eva/, tau-voice/
```

Development:

```bash
uv run pytest
uv run ruff check src adapters tests && uv run ruff format src adapters tests
uv run ty check
```

Some tests need Docker and the benchmarks' images, and skip without them. CI
runs the rest on Linux and Windows.

## License

Apache-2.0; see [LICENSE](LICENSE). harbor-voice builds on harbor and
depends on it as a package. The adapters run tau2-bench and EVA in
containers at pinned commits and don't redistribute them, except the parts
copied into the repo, which are under the MIT License: see
[tau2-bench's](adapters/tau-voice/src/tau_voice/tau2_audio/LICENSE) and
[EVA's](adapters/eva/src/eva_adapter/vendored/LICENSE) license files.
