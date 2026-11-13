# tau-voice on harbor-voice

τ-Voice is the full-duplex voice mode of
[tau3-bench](https://github.com/sierra-research/tau2-bench), Sierra's
benchmark for customer-service agents. A simulated customer phones the agent
about a problem in one of three domains: airline, retail or telecom. The agent
talks the caller through it and acts with the domain's tools (looking up a
reservation, changing an order, …) while following the domain's policy. Both
sides can speak at once, as on a real call, and the caller can sound like a
real one: accents, background noise, a phone line, interruptions and
backchannels (tau3-bench's speech complexities). tau3-bench's own evaluator
then grades whether the task was done.

This adapter runs τ-Voice's 278 tasks (airline 50, retail 114, telecom 114)
as harbor tasks, at tau3-bench's commit `2174a60`. tau3-bench's own code holds
each task, its tools, the caller's prompts and the grade, in a sidecar;
harbor-voice runs the call step for step as tau3-bench's harness does. How it
compares with tau3-bench's own `tau2 run`: [PARITY.md](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/PARITY.md).

## Run it

From a checkout of harbor-voice (`uv sync`). With `pip install
"harbor-voice[tau-voice]"`, `voices` and `job-config` work the same without
`uv run`; converting tasks needs the checkout.

```bash
# Tasks: one per tau2 task in the base split (airline 50, retail 114,
# telecom 114), from a tau2-bench checkout. A few: --domains airline
# --task-ids 0 1 2; a speech complexity: --complexity regular.
uv run tau-voice --tau2-root ../tau2-bench --output-dir datasets/tau-voice

# One trial, locally (Docker), with the parity run's agent: OpenAI Live,
# gpt-live-1 delegating to a backend model:
uv run tau-voice job-config --dataset datasets/tau-voice \
    --task-names airline-0-d9c07603 --trials 1 -n 1 \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    -o one.yaml
uv run harbor-voice jobs start -c one.yaml

# A job, run the way tau2 runs its trials:
uv run tau-voice job-config --dataset datasets/tau-voice --trials 3 -n 4 \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    -o job.yaml
uv run harbor-voice jobs start -c job.yaml
```

For OpenAI Realtime instead: `--model openai.realtime/gpt-realtime-1.5` and
no `--agent-config`.

`job-config` lists the agent once per tau2 trial seed (tau2's run seed 300
gives 626729, 373753, 361454, ...), retries infrastructure failures three
times, and its metric (`src/tau_voice/metrics/tau2_mean.py`) leaves errored
trials out of the mean, as tau2 does. `--env daytona` (or any harbor
backend) runs the trials remotely.

Environment (names only; a task holds `${VAR}` templates, filled at trial
start):

| Variable | What for |
|---|---|
| `OPENAI_API_KEY` | the agent under test; the caller's LLM; tau2's listener decisions |
| `ELEVENLABS_API_KEY` | the caller's voice |
| `TAU2_VOICE_ID_<PERSONA>` | one ElevenLabs voice per tau2 persona: `MATT_DELANEY`, `LISA_BRENNER` (`control`), `MILDRED_KAPLAN`, `ARJUN_ROY`, `WEI_LIN`, `MAMADOU_DIALLO`, `PRIYA_PATIL` (`regular`). `tau-voice voices create --yes` makes them once, as tau2's `setup_voices` does (same prompts, settings and seed), and prints these lines; `tau-voice voices list` prints them again |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | tau2's hallucination reviewer, Claude Opus 4.5 on Bedrock |
| `TAU_VOICE_REVIEW_MODEL` | optional: another reviewer model |

## One call

| Part | What it is | Where |
|---|---|---|
| Agent under test | a speech-to-speech model: OpenAI Realtime (`openai.realtime/...`) or OpenAI Live (`openai.live/gpt-live-1` and a backend) | its provider; harbor-voice holds the connection |
| Caller | tau2's turn policy (when to speak), an LLM (what to say, `--user-llm`), ElevenLabs TTS (`eleven_v3`, the persona's voice), tau2's phone line and effects | harbor-voice, in `main` |
| Sidecar | tau2 as a service: the task, its database and tools, the caller's prompts, the listener's decisions, grading | the `tau2` container |

- **The step clock.** A call is a sequence of 0.2 s steps (tau2's tick):
  each carries 0.2 s of the caller's 8 kHz μ-law audio and up to 0.2 s of
  the agent's, with tau2's one-step lag between them. Call time is steps ×
  0.2 s; a call is cut off at 1200 s (`max_steps`). The caller's thinking
  (its LLM, decisions, TTS, tools) happens between steps and never counts as
  call time, as in tau2's default.
- **Nothing of tau2's prompting is ported:** the sidecar builds every caller
  request with tau2's own code. tau2's audio effects are vendored verbatim
  (`tau_voice.tau2_audio`, MIT) and checked against tau2 byte for byte.
- **Ends:** `user_stop` (the caller hangs up), `agent_stop` (the agent
  transfers to a human), `max_steps`, `too_many_errors`.
- **Hallucination reruns**, as tau2's: after each attempt the reviewer checks
  whether the *caller* invented facts; if so the task reruns with seed +
  1000·n, up to 3 times, and the last attempt counts.
- **Reward:** tau2's evaluator, through the sidecar (`tests/test.sh`, harbor's
  verifier).

Out, per attempt, under the trial's `agent/call/tau_voice/attempt-<n>/`:
`summary.json` (call time, termination, steps, the caller's calls, effects,
usage counts for every model), `steps.json` (every step), `result.json`
(tau2's grade and the hallucination verdict), `tau2_state.json` (the step
record in tau2's format), `trajectory.json` (ATIF-voice) and `both.wav`
(stereo: the caller as the agent heard it, the agent as the caller heard it).

## Speech complexity

`--complexity` is one of tau2's presets, for every task converted; convert
each into its own folder (task folder names do not carry it).

| Preset | Personas | Audio effects | Behaviour |
|---|---|---|---|
| `control` | American: Matt Delaney, Lisa Brenner | none; G.711 phone line only | waits its turn |
| `regular` (the leaderboard's) | five accents: Mildred Kaplan, Arjun Roy, Wei Lin, Mamadou Diallo, Priya Patil | background noise (SNR 15 dB), bursts, muffling, frame drops, phone line | interrupts and backchannels (decided by `gpt-4.1`, tau2's fixed decision model), vocal tics, out-of-turn phrases |

The six ablations (`control_audio`, `control_accents`, `control_behavior`
and their pairs) are `control` plus one or two of `regular`'s groups. tau2
seeds a call's effects with `hash(task.id)`; the sidecar image pins
`PYTHONHASHSEED=0` so that seed is the same in every process.

## Agents

The adapter takes these two agents; more are on the way.

| | OpenAI Realtime (`openai.realtime`) | OpenAI Live (`openai.live`) |
|---|---|---|
| Model | one speech model, e.g. `gpt-realtime-1.5` | `gpt-live-1`, a speech frontend delegating to a backend (e.g. `gpt-6-luna`) that reasons and calls tools |
| Wire | WebSocket, 8 kHz μ-law | WebRTC (harbor-voice's `live` extra) |
| tau2's adapter reproduced | its discrete-time OpenAI adapter | `DiscreteTimeOpenAILiveAdapter` |
| The caller barges in | the agent's utterance is truncated at what played | recorded only; the frontend decides whether to stop |
| While the caller thinks | nothing is sent, so the pause does not exist for the agent | the WebRTC track keeps sending silence |

OpenAI Live takes its backend as the agent's s2s settings:
`job-config --model openai.live/gpt-live-1 --agent-config '{"backend_model":
"gpt-6-luna"}'`. Its prompts are tau2's, rendered by tau2 in the sidecar.
tau2's own Live provider sends an `OpenAI-Alpha` preview header that the
documented API does not take; the sidecar image removes that one line, so
both harnesses call the same API.

## Images

`main` is a slim Python with uv (harbor-voice is installed into it at trial
start); the sidecar is tau2 at the commit plus `sidecar/server.py` and
`sidecar/evaluate.py`. Tags carry a digest of what goes into each image.
Both are built for `linux/amd64` and `linux/arm64`, so they run without
emulation on x86 backends and on Apple silicon (Docker Sandboxes on a Mac).

- **Default:** each task builds both images where it runs, as harbor tasks
  do; after the first trial the build is a cache hit.
- **Prebuilt:** `tau-voice build-image` builds both once; tasks converted
  with `--prebuilt` name them instead of building. For remote backends,
  `build-image --registry <prefix> --push`, then convert with the same
  `--registry`, so every trial pulls the same images.

## Parity

Harbor's job of the parity tasks against tau2's own `tau2 run` on the same
tasks (`original.tau2_command`, in the tau2 image), then `compare` writes
harbor's `parity_experiment.json`. The results, the exact tau2 command, what
is compared and how to reproduce it: [`PARITY.md`](https://github.com/amoh-ai/harbor-voice/blob/main/adapters/tau-voice/PARITY.md).

```bash
uv run tau-voice sample --tau2-root ../tau2-bench    # parity_tasks.json (40)
uv run tau-voice --tau2-root ../tau2-bench --output-dir datasets/parity --split parity
uv run tau-voice compare --harbor <job folder> ... --original <tau2 save folder> ... \
    --complexity control [--metrics metrics-control.json] --out parity_experiment.json
```

## Tests

In [`tests/adapters/tau-voice/`](https://github.com/amoh-ai/harbor-voice/tree/main/tests/adapters/tau-voice), with their
fixtures.

- Goldens recorded from tau2's own code in the sidecar image
  (`tau2_*_cases.py`): the agent's per-step handling for Realtime and
  Live (`test_agent_steps.py`), the phone line (`test_telephony.py`), the
  audio effects (`test_tau2_audio.py`), and the caller end to end, effects
  and behaviour (`test_tau2_caller_effects.py`,
  `test_tau2_caller_behavior.py`).
- The step loop (`test_stepped_call.py`), the caller's policy
  (`test_reactive_caller.py`), the trial and trajectory (`test_trial.py`,
  `test_tau_trajectory.py`), the converter and job (`test_converter.py`,
  `test_job.py`).
- Parity: the sample (`test_sample.py`), tau2's command
  (`test_original.py`), the statistics (`test_compare.py`).
