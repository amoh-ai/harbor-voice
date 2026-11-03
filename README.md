# harbor-voice

harbor-voice is [harbor](https://github.com/harbor-framework/harbor) for
voice agents. It puts a voice agent on a phone call with a simulated caller,
records the call and grades it, on harbor's job runner and environments. You
can use harbor-voice to:

- Evaluate speech-to-speech agents such as OpenAI Realtime and OpenAI Live
  on voice benchmarks: [tau-voice](adapters/tau-voice) (tau3-bench's voice
  mode) and [EVA](adapters/eva).
- Run calls in parallel on harbor's backends, from your laptop's Docker to
  Daytona or EC2, with harbor's retries and resume.
- Test agents under realistic audio: accents, background noise, phone lines
  and connection loss, as each benchmark defines them.
- Keep every call: its audio, a timed trajectory in ATIF-voice (harbor's ATIF
  format adapted for voice), and the benchmark's own grade.
- Add your own voice benchmark as an adapter, like the two included.

## Core features

- Full-duplex and half-duplex calls
- Audio-native speech-to-speech agents (OpenAI Realtime, OpenAI Live, Amazon
  Nova Sonic, ElevenLabs Agents) and cascades (speech-to-text → LLM →
  text-to-speech), with more providers to come
- A simulated phone line: sample rates, μ-law/G.711, packet loss, jitter
- A simulated caller with seeded perturbation layers: persona, speech, room,
  line, turn-taking, tool faults
- ATIF-voice trajectories, and voice metrics that abstain rather than score 0

The benchmarks run full-duplex speech-to-speech calls with their own callers;
the rest runs on harbor-voice's built-in call loop (see
[The core library](#the-core-library)).

## Benchmarks

- **[tau-voice](adapters/tau-voice/README.md)**: τ-Voice, the full-duplex
  voice mode of [tau3-bench](https://github.com/sierra-research/tau2-bench).
  A simulated customer calls a service agent about airline, retail and telecom
  tasks (278), and tau3-bench's own evaluator grades the outcome.

- **[EVA](adapters/eva/README.md)**: ServiceNow's
  [EVA](https://github.com/ServiceNow/eva). Enterprise calls in airline, IT
  service and medical HR (213 scenarios), scored for accuracy (EVA-A) and
  experience (EVA-X) by EVA's own metrics and judges.

More voice benchmarks are on the way.

## Datasets

Converted tasks are published in
[harbor-voice-datasets](https://github.com/amoh-ai/harbor-voice-datasets) and
listed in this checkout's [`registry.json`](registry.json). Each version pins
its tasks to one commit and its images to digests on `ghcr.io/amoh-ai`, so a
version always runs the same bytes. `job-config --hub-dataset
NAME[@VERSION]` runs one without converting anything (with no version,
harbor picks the latest):

```bash
uv run tau-voice job-config --hub-dataset tau-voice-regular-test@0.1.0 \
    --trials 3 -n 4 --model openai.live/gpt-live-1 \
    --agent-config '{"backend_model": "gpt-6-luna"}' -o job.yaml
uv run eva-adapter job-config --hub-dataset eva@0.1.0 \
    --model openai.realtime/gpt-realtime-1.5 -o job.yaml
uv run harbor-voice jobs start -c job.yaml
```

| Dataset | Tasks | What |
|---|---|---|
| `tau-voice-control` | 278 | every tau2 task, `control` speech (clean, patient turn-taking) |
| `tau-voice-regular` | 278 | every tau2 task, `regular` speech (accents, noise, interruptions) |
| `tau-voice-control-test` | 100 | tau2's `test` split, `control` speech |
| `tau-voice-regular-test` | 100 | tau2's `test` split, `regular` speech: the τ-voice leaderboard's condition |
| `tau-voice-parity-control` | 40 | the [parity](adapters/tau-voice/PARITY.md) sample, `control` speech |
| `tau-voice-parity-regular` | 40 | the parity sample, `regular` speech |
| `eva` | 213 | every EVA record, clean audio |
| `eva-parity` | 60 | the [parity](adapters/eva/PARITY.md) sample, 20 per domain |

You still need the keys and the one-time ElevenLabs setup in the quickstarts
below. To pick a few tasks, add `--task-names` with their folder names.

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
| `TAU2_VOICE_ID_<PERSONA>` | one ElevenLabs voice per tau2 persona, made once by `tau-voice voices create --yes` (tau2's `setup_voices`, without a tau2 checkout): two for `control`, five more for `regular` |
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

- **Backends.** Local Docker, Daytona and EC2 today, with more to come. The
  benchmark runs in a sidecar, so a backend needs docker-compose support.
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
  and `harbor-voice listen <trajectory.json> --out call.wav` writes its
  recording, found in the trial folder by its hash.
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

  More providers are on the way.

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

## CLI

| Command | |
|---|---|
| `harbor-voice trials start` / `jobs start` | harbor's runners; the agent defaults to `VoiceCallAgent` |
| `harbor-voice status <job>` | a running job: counts, reward so far, each trial's phase |
| `harbor-voice show` / `listen <trajectory>` | summarise a recorded call; write its recording to a WAV |
| `harbor-voice caps` | provider capability matrices, and which credentials are missing |
| `harbor-voice conformance` | the provider conformance suite |
| `harbor-voice doctor` / `profiles` / `metrics` | the built-in loop's conditions and metrics: what a configuration will report, abstain on, or reject |
| `harbor-voice report <job>` | roll up built-in-loop trials and diff conditions against a baseline |
| `harbor-voice drive` | runs a task's call driver; what `VoiceCallAgent` runs inside `main` |
| `tau-voice …` | convert tasks; `build-image`, `job-config`, `voices`; parity: `sample`, `compare` |
| `eva-adapter …` | `convert`, `agents`, `build-image`, `job-config`; parity: `sample`, `compare` |

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
runs the rest on Linux and Windows. To contribute, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0; see [LICENSE](LICENSE). harbor-voice builds on harbor and
depends on it as a package. The adapters run tau2-bench and EVA in
containers at pinned commits and don't redistribute them, except the parts
copied into the repo, which are under the MIT License: see
[tau2-bench's](adapters/tau-voice/src/tau_voice/tau2_audio/LICENSE) and
[EVA's](adapters/eva/src/eva_adapter/vendored/LICENSE) license files.
