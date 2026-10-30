# harbor-voice

**Voice-agent evaluation on [harbor](https://github.com/harbor-framework/harbor).**
harbor-voice puts a voice agent on a phone call with a simulated caller,
records the call as a trajectory, and grades it, using harbor's own job
runner and environments. The same command runs a call on your laptop's
Docker or on Daytona or EC2.

> Status: pre-release (0.1.0). The tau-voice adapter runs end to end on
> Docker and Daytona; parity results against the original tau2-bench are in
> progress.

## What it does

- **Calls, not chat turns.** Speech-to-speech agents (e.g. OpenAI Realtime,
  Amazon Nova Sonic) and cascades (STT → LLM → TTS), in half or full duplex,
  over a simulated wire: sample rates, μ-law/G.711, packet loss, jitter.
- **A caller you can perturb, layer by layer.** A seeded simulator with six
  layers between the caller and what the agent hears, each recorded with what
  it *achieved*, not just what was asked for:

  ```
  L1 PERSONA      who speaks            voice, persona brief
  L2 SPEECH       how they speak        disfluency, rate
  L3 ENVIRONMENT  the room              continuous background bed
  L4 CHANNEL      the wire              codec, loss, jitter, phone line
  L5 CONVERSATION when they speak       barge-in, backchannel, interjections
  L6 WORLD        what the tools do     latency, errors, outages
  ```
  The clean reference is always kept, so a degraded run can be compared with
  the same call undegraded.
- **ATIF-voice trajectories.** Every call is written as a genuine harbor ATIF
  `Trajectory` with a voice payload: a continuous timeline (speech, overlaps,
  barge-ins, tool calls), turns with what was said and what came back, and
  audio by content hash, never inlined.
- **Voice metrics** that abstain rather than score 0 when a call cannot
  support them: turn timing, overlap and barge-in, said-vs-heard WER, and
  run validity.
- **Benchmark adapters.** [tau-voice](adapters/tau-voice) (tau2-bench's
  full-duplex voice mode: airline, retail, telecom, 278 tasks) converts
  into harbor tasks that run unchanged on any harbor backend.

## How it fits harbor

harbor-voice adds no runner of its own. A voice task is a harbor task, and
the call runner is a harbor agent, `VoiceCallAgent`:

```
harbor-voice jobs start -c job.yaml          (harbor's job runner, mounted)
  └─ per trial, a harbor environment (docker | daytona | ec2 | …)
       ├─ main  ← VoiceCallAgent installs harbor-voice here and runs the call
       └─ benchmark sidecar(s) from the task's docker-compose.yaml
  └─ harbor's verifier grades it; harbor collects logs and trajectory.json
```

So harbor's concurrency, retries, resume, metrics, `harbor view` and every
environment backend work as they do for any harbor agent. `harbor-voice jobs`
and `harbor-voice trials` *are* `harbor jobs` and `harbor trials`; when no
agent is given, the agent defaults to `VoiceCallAgent`. The voice agent under
test is the run's choice (`-m`), never the task's.

## Install

Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker for local runs.

```bash
git clone https://github.com/amoh-ai/harbor-voice
cd harbor-voice
uv sync                      # harbor-voice + the tau-voice adapter + dev tools
```

harbor is a pinned dependency (`harbor==0.25.0`). Optional extras:

| Extra | For |
|---|---|
| `aws` | Amazon Nova Sonic / Bedrock providers |
| `daytona` | harbor's Daytona backend |
| `ec2` | harbor's EC2 backend |

## Quickstart: tau-voice

**1. Keys.** Put these in your environment (or a `.env` you source). Tasks
hold only `${VAR}` templates, and harbor fills them from your environment at
trial start.

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | the agent under test (OpenAI Realtime), tau2's caller LLM |
| `ELEVENLABS_API_KEY` | the caller's voice |
| `TAU2_VOICE_ID_MATT_DELANEY`, `TAU2_VOICE_ID_LISA_BRENNER` | ElevenLabs voice IDs for tau2's personas |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | tau2's reviewer model on Bedrock (`TAU_VOICE_REVIEW_MODEL` overrides it) |

**2. Convert tasks.** This needs a checkout of
[tau2-bench](https://github.com/sierra-research/tau2-bench), which supplies
the task lists:

```bash
uv run tau-voice --tau2-root ../tau2-bench --output-dir datasets/tau-voice
# or a few: --domains airline --task-ids 0 1 2
```

Each task folder is a harbor task: `task.toml`, `instruction.md`,
`environment/` (the `main` image and a tau2 sidecar that runs tools, the
caller's LLM and grading), and `tests/test.sh`.

**3. Run one call** (local Docker):

```bash
harbor-voice trials start -p datasets/tau-voice/airline-0-d9c07603 \
  -m openai.realtime/gpt-realtime-1.5
```

**4. Run a job.** `job-config` writes a harbor job that runs tasks the way
tau2 runs its trials: one agent entry per tau2 seed, three retries for
infrastructure failures only, and a mean that leaves errored trials out:

```bash
uv run tau-voice job-config --dataset datasets/tau-voice \
  --model openai.realtime/gpt-realtime-1.5 --trials 3 -n 4 -o job.yaml
harbor-voice jobs start -c job.yaml
```

A call runs in real time (a few minutes each), so concurrency (`-n`) is the
lever for large runs.

## Running on cloud

The orchestrator stays on your machine; the backend creates one environment
per trial and deletes it afterwards. Choose the backend with
`job-config --env <type>` (or `--env` on `trials start`). Settings that belong
to your account go in your own copy of the job config, never in the tasks.

**Daytona.** Set `DAYTONA_API_KEY` and install the `daytona` extra. Each
trial is a Docker-in-Docker sandbox sized from the task (2 CPU, 4 GB, 10 GB).

```bash
uv run tau-voice job-config ... --env daytona -o job.yaml
harbor-voice jobs start -c job.yaml
```

**EC2.** Install the `ec2` extra and sign in with any standard AWS credential
(a profile, SSO, `aws login`, environment variables). harbor launches a VM per
trial, connects over SSH and terminates it afterwards. It needs a key pair and
a security group that allows SSH from your machine. Add your account's
settings to the job config:

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

**Faster startup.** By default each environment builds the task's images (on
Daytona that is about 2 minutes per trial). `tau-voice build-image` builds them
once, and tasks converted with `--prebuilt` reference those images instead.
For cloud backends, push them to a registry the backend can pull from.

## What a trial leaves behind

```
<trial>/
├── result.json                 harbor's trial result (reward, timings, agent)
├── agent/
│   ├── trajectory.json         ATIF-voice: the call that counts
│   └── call/
│       ├── outcome.json
│       └── tau_voice/attempt-N/
│           ├── both.wav        the call, stereo: caller | agent
│           ├── steps.json      per-tick record of both sides
│           ├── trajectory.json
│           ├── tau2_state.json tau2's own record of the call
│           └── result.json     tau2's grade
└── verifier/{reward.txt, result.json}
```

tau2 reruns a call when its reviewer finds that the *simulated caller*
hallucinated (up to three times, seed + 1000·n); every attempt is kept and
the last one counts.

## CLI

| Command | |
|---|---|
| `harbor-voice trials start` / `jobs start` | harbor's runners; agent defaults to `VoiceCallAgent` |
| `harbor-voice doctor` | preflight: what a configuration will report, abstain on, or reject |
| `harbor-voice profiles` | named perturbation conditions and the layers they activate |
| `harbor-voice caps` | provider capability matrices |
| `harbor-voice metrics` | registered metrics and what they need |
| `harbor-voice show` / `listen` | summarise a recorded call, render its timeline to WAV |
| `harbor-voice report` | roll a job up and diff profiles against the baseline |
| `harbor-voice conformance` | provider conformance suite |
| `harbor-voice drive` | runs a task's call driver; what `VoiceCallAgent` runs inside `main` |
| `tau-voice …` | convert tasks, `build-image`, `job-config` |

Agents are named `provider/model`, e.g. `openai.realtime/gpt-realtime-1.5` or
`aws.nova-sonic/<model>`.

## Repository

```
src/harbor_voice/     transports and providers, the caller simulator and layer
                      stack, ATIF-voice, metrics, VoiceCallAgent, the CLI
adapters/tau-voice/   the tau-voice adapter (its own package)
adapters/eva/         the EVA adapter (its own package)
tests/, adapters/tau-voice/tests/, adapters/eva/tests/
```

Development:

```bash
uv run pytest
uv run ruff check src adapters tests && uv run ruff format src adapters tests
uv run ty check
```

## License

Apache-2.0; see [LICENSE](LICENSE). harbor-voice builds on harbor and depends
on it as a package. The tau-voice adapter runs tau2-bench in a container at a
pinned commit and does not redistribute it.
