# EVA on harbor-voice

harbor-voice runs [EVA](https://github.com/ServiceNow/eva) (ServiceNow's
voice-agent benchmark) at commit `039efa0a`: EVA's own code decides the
prompts, tools, validation and scores, in a sidecar; harbor-voice runs the
call.

## Run it

```bash
# Tasks: one per EVA record (airline 50, itsm 80, medical_hr 83), and job.yaml
# for the reference system over EVA's k = 3 trials.
uv run eva-adapter convert --out datasets/eva

# One trial, locally (Docker):
uv run harbor-voice trials start -p datasets/eva/airline-1-1-2-17dc1179 \
    -m openai.realtime/gpt-realtime-1.5

# The job:
uv run harbor-voice jobs start -c datasets/eva/job.yaml
```

Environment (names only; nothing is written into a task):

| Variable | Where | What for |
|---|---|---|
| `OPENAI_API_KEY` | main, eva | the agent under test; EVA's gpt-5.2 judges |
| `ELEVENLABS_API_KEY` | main | the caller (ElevenLabs Agents) |
| `EVA_EN_USER_F`, `EVA_EN_USER_M` | main | EVA's two English caller agents (`eva-adapter agents`) |
| `EVA_MODEL_LIST` | eva | EVA's LiteLLM deployments for its judges, as EVA reads it |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `VERTEXAI_*`, `GOOGLE_APPLICATION_CREDENTIALS` | eva, optional | Gemini, the audio judges (whichever the model list uses) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION` | eva, optional | Bedrock, the faithfulness judge |
| `JUDGE_MODEL` | eva, optional | EVA's own judge override |
| `EVA_SIDECAR_METRIC_CONFIGS` | eva, optional | EVA's `metric_configs` as JSON (`EVA_METRIC_CONFIGS`' format), handed to EVA by the sidecar; empty is EVA's judges |
| any name given to `convert --sidecar-env` | eva, optional | whatever your own model list's deployments read (a proxy's URL and key, say) |

The required ones are `${VAR}` in the task, so a trial refuses to start
without them; the optional ones are `${VAR:-}`. EVA's own `EVA_*` settings are
not passed: an empty one fails EVA's config, and a set one changes its
protocol. `EVA_SIDECAR_METRIC_CONFIGS` is the sidecar's channel for the one
EVA setting a run may want, which judges grade (`sidecar/server.py`); EVA
ignores the name itself.

**The judges' model list** is [`judges.json`](judges.json):
`EVA_MODEL_LIST="$(cat adapters/eva/judges.json)"`. It names every judge
deployment EVA's metrics ask for, as EVA's published setup runs them, and
holds only `os.environ/…` references, never a key:

| Deployment | Judges | Through |
|---|---|---|
| `gpt-5.2` | conciseness, conversation progression (EVA adds `service_tier: flex`) | OpenAI, `OPENAI_API_KEY` |
| `gpt-5.2-medium` | user behavioural fidelity (validation) | OpenAI, `OPENAI_API_KEY`, `reasoning_effort: medium` |
| `gemini-3-flash-preview` | agent and user speech fidelity (audio), EVA's default | the Gemini API, `GEMINI_API_KEY` |
| `us.anthropic.claude-opus-4-6-v1` | faithfulness | Bedrock, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, region from `AWS_REGION` |

- EVA names `gpt-5.2-medium` but never defines it. This file reads the name
  as gpt-5.2 at medium reasoning effort.
- EVA's own setup reaches Gemini through Vertex; `gemini-3-flash-preview`
  here uses a Gemini API key. The model is the same.

**The audio judges.** A plain run uses EVA's `gemini-3-flash-preview`
through the Gemini API (`GEMINI_API_KEY`). To judge audio with another
model, `gemini-3.8-flash` on Vertex AI say (as the parity run did, PARITY.md
"How it was run"):

1. Add its deployment to your own copy of the list, reaching it however you
   like: `"model": "gemini/gemini-3.8-flash"` with
   `"api_key": "os.environ/GEMINI_API_KEY"`; `"vertex_ai/gemini-3.8-flash"`
   with `vertex_project`, `vertex_location` (`global`) and credentials the
   container can read (`GOOGLE_APPLICATION_CREDENTIALS`); or an
   OpenAI-compatible LiteLLM proxy, `"model": "openai/gemini-3.8-flash"` with
   `"api_base"` and `"api_key"` as `os.environ/<NAME>` references.
2. Convert with `--sidecar-env <NAME> ...` for every variable the deployment
   reads that the table above lacks, so it reaches the sidecar.
3. Set `EVA_SIDECAR_METRIC_CONFIGS` (and `EVA_METRIC_CONFIGS` for EVA's own
   CLI) to EVA's `metric_configs` for the audio metrics, e.g.
   `{"agent_speech_fidelity":{"audio_judge_model":"gemini-3.8-flash","judge_params":{"reasoning_effort":"low"}},"user_speech_fidelity":{...the same...}}`.

The sidecar refuses a trial whose judge deployment's `api_base` names an
empty variable: LiteLLM would otherwise send the judge to its provider's
default endpoint (OpenAI's, for an OpenAI-compatible proxy). On Vertex, keep
`reasoning_effort` at `low` or above: the model rejects thinking level
MINIMAL, EVA's default `"minimal"`.

## One trial

`eva_adapter.trial:drive` (the task's `[call] driver`), in `main`:

1. `POST /open` → EVA's session for the attempt (prompt, tools, greeting) and
   its caller (prompt, gender).
2. The agent is built from the run's `[agent]` with EVA's Realtime settings,
   and its `session.update` must equal the one EVA's own server rendered for
   this record, or the trial fails before anything connects.
3. The call: `RealtimeCall` between `openai.realtime` and the
   `elevenlabs.agents` caller (EVA's agent for the persona's gender); tools go
   to `POST /tool`.
4. `POST /record` (always, with the error if the call failed), then
   `POST /validate`; EVA's rerun rule decides, up to 3 attempts.
5. `POST /grade` if an attempt passed; `GET /files` either way.

Out, under the trial's `/logs/agent/call/eva/`: `summary.json` (each
attempt's end reason, validation and timings; usage for the agent, the caller
and each judge), `metrics.json` (EVA's, for the attempt that counts), `run/`
(EVA's run directory: `config.json`, `evaluation_summary.json`,
`records/<id>/trial_<k>/`, archived attempts, `judge_token_usage/`),
`attempt-<n>/attempt.json` and the caller agent's live config under
`elevenlabs/`. The ATIF-voice trajectory of the attempt that counts is at
`/logs/agent/trajectory.json`, audio by content hash.

**Reward**: `[call.config].reward`, `EVA-A_pass` (default), `EVA-X_pass` or
`EVA-overall_pass`, is harbor's `reward`; `reward.json` also holds all three
composites under their own names, and the job metric reports each one's
mean. A record no attempt validated has no reward: EVA scores
no such record, and the job metric (`template/metrics/eva_mean.py`) counts
it as unscored. EVA's run-level numbers (pass@k, pass^k, CIs) come from
`sidecar/aggregate.py` over the trials' `metrics.json`.

**Verifier**: the driver grades through the sidecar itself (the summary and
trajectory need the grade, and `main` cannot exec into the sidecar), and the
task's `tests/test.sh` still runs after it and is the reward harbor records,
as in tau-voice. It calls the same `/grade`, which costs nothing twice:
EVA's `MetricsRunner` computes only metrics missing from `metrics.json`.

**Trial index**: EVA seeds nothing per trial; its trials of a record are
`trial_0` … `trial_2`. The task's seed is that index, and the job lists the
call agent once per seed 0, 1, 2.

## Images

The sidecar is EVA's image at the commit plus `sidecar/server.py`; `main`
is a slim Python with uv. Tags carry the EVA commit and a digest of our files.

- **Default**: each task builds the sidecar where it runs, EVA included, from
  ServiceNow's public repo at the full commit (`EVA_BASE=eva-from-source`: EVA's
  Dockerfile stage for stage). Works anywhere, Daytona included, with no
  registry; but every fresh sandbox fetches and installs EVA (minutes).
- **For runs of any size**: build once and push, then convert against the
  registry, so every trial pulls the same bytes:

  ```bash
  uv run eva-adapter build-image --registry ghcr.io/<org> --push
  uv run eva-adapter convert --out datasets/eva --registry ghcr.io/<org>
  ```

  `build-image` builds `eva-base:039efa0a` with EVA's own Dockerfile from the
  repo URL if it is missing, the sidecar on it, and `main`.
- **Local dev**: `docker build -t eva-base:039efa0a <eva checkout>`, then
  `docker build adapters/eva/sidecar` builds on it (the Dockerfile's default).
- **Perturbed tasks** use a second `main`, `eva-main:perturbed-<digest>`:
  the same, plus EVA's numpy and audioop-lts pins and EVA's seven noise WAVs
  (+118 MB of layers: 61 MB of pins, 58 MB of WAVs). `build-image` builds and
  pushes it beside the clean one; clean tasks never use it.

## Parity

Harbor's job of the sample's records against EVA's own CLI on the same
records (`original.eva_command`, in EVA's image; twice for the noise floor),
then `compare` writes harbor's `parity_experiment.json`. The results, the
exact EVA command, what is compared and how to reproduce it:
[`PARITY.md`](PARITY.md).

```bash
uv run eva-adapter sample                           # parity_records.json (30), --smoke, --all
uv run eva-adapter compare --harbor <job folder> ... --original <EVA run dir> ... \
    [--original2 <EVA run dir> ...] --records adapters/eva/parity_records.json
```

A judge that failed during a run (a 429, say) leaves a trial graded but
without its composite (`judge_errored`, counted apart); EVA's own rerun of
failed metrics (`EVA_RERUN_FAILED_METRICS`) recomputes just those scores
from the recordings.

## Perturbations

EVA's audio perturbations, under EVA's own names (`EVA_PERTURBATION__*`),
run by EVA's own `AudioPerturbator`
([`vendored/perturbation.py`](src/eva_adapter/vendored/perturbation.py),
EVA's file at the commit but for where it finds the WAVs; MIT, see `vendored/LICENSE`)
where EVA's bridge runs it: on every chunk of the caller's audio before the
agent hears it (the recording keeps the clean copy, which EVA's
`user_speech_fidelity` judges), and as ambient noise whenever the caller is
quiet, the agent's turns included. A new one per attempt, as EVA builds one
per conversation.

| Flag | EVA's setting | Values |
|---|---|---|
| `--background-noise` | `EVA_PERTURBATION__BACKGROUND_NOISE` | `airport_gate`, `baby_crying`, `background_music`, `bad_connection_static`, `coffee_shop`, `loud_construction`, `nyc_street`, `road_noise` |
| `--snr-db` | `EVA_PERTURBATION__SNR_DB` | dB, EVA's default 15 (EVA checks nothing else; its UI offers 0–40); needs a noise |
| `--connection-degradation` | `EVA_PERTURBATION__CONNECTION_DEGRADATION=true` | μ-law codec, 3 % packet loss per 20 ms, a random gain (0.5–1.2) per chunk; stacks on a noise |

- `bad_connection_static` is Gaussian noise at the SNR; the other seven mix
  a WAV from EVA's repo (`assets/noise/`). The WAVs are not in this repo:
  a perturbed task's `main` fetches them from GitHub at the full commit id
  when it is built, and checks each one's SHA-256.
- **Not supported yet:** EVA's accents (they need ElevenLabs agents EVA's
  repo does not define) and behaviours (they change the caller's prompt and
  agent). `--accent` and `--behavior` are refused by name.
- **The randomness is EVA's.** Static, packet loss and gain draw from
  numpy's global generator, unseeded, as in EVA. Two runs of a condition
  never hear the same noise: compare them statistically, never trial by
  trial.
- **So is the arithmetic.** EVA scales noise by each chunk's float32 RMS, and
  numpy 2.3 changed how float32 sums over 8192 elements round, so a perturbed
  `main` installs EVA's numpy (2.2.6). A trial records the numpy it ran
  (`summary.json`, the trajectory's `perturbation_plan`).

**One condition per dataset.** Convert each condition into its own folder;
the condition is in every task's name (`airline-1-1-2-17dc1179-coffee_shop-15db`,
`eva/airline/1.1.2@coffee_shop-15db`), its `[metadata]` and
`[call.config.perturbation]`, so results of two conditions never share a
task; and `convert` refuses a folder that holds another condition's tasks,
so they never share a job's metric either (harbor pools a dataset folder).
The trajectory's `config_matrix.profile` is
`eva/<condition>`. Without the flags, `convert` writes today's clean tasks
byte for byte.

```bash
# A condition, with plain harbor-voice:
uv run eva-adapter convert --out datasets/eva-coffee_shop-15db \
    --background-noise coffee_shop --snr-db 15
uv run harbor-voice jobs start -c datasets/eva-coffee_shop-15db/job.yaml
#   one trial: harbor-voice trials start -p datasets/eva-coffee_shop-15db/<task> \
#              -m openai.realtime/gpt-realtime-1.5

# Static plus connection degradation, on Daytona with prebuilt images:
uv run eva-adapter build-image --registry ghcr.io/<org> --push   # adds eva-main:perturbed-*
uv run eva-adapter convert --out datasets/eva-static-10db-degraded --registry ghcr.io/<org> \
    --background-noise bad_connection_static --snr-db 10 --connection-degradation
uv run eva-adapter job-config --dataset datasets/eva-static-10db-degraded -n 3 \
    --env daytona -o job-static.yaml
uv run harbor-voice jobs start -c job-static.yaml

# Parity under a condition: both sides, one condition per comparison; EVA's side
# gets the same condition as its own EVA_PERTURBATION__* settings.
uv run eva-adapter compare --harbor <job folder> ... --original <EVA run dir> ... \
    --background-noise coffee_shop --snr-db 15
```

The conditions are every noise at any SNR, with or without connection
degradation, and connection degradation alone: EVA's own grid.

## Tests

- `test_eva_perturbation.py`: the vendored file against EVA's image; the
  goldens (EVA's perturbator in EVA's image on seeded inputs, 28 cases:
  every noise, SNRs, static, degradation, combinations, the ambient stream,
  a wrap of the noise file, silence, clipping) byte for byte in perturbed
  `main` and, chunk by chunk, here; the bridge's wire, recording and ambient
  noise; EVA's settings reading the original side's condition in EVA's
  image. The Docker tests skip without Docker or `eva-base:039efa0a`.
- `test_eva_sample.py`, `test_eva_original.py`, `test_eva_compare.py`: the
  samples (fixed, stratified, the checked-in files), EVA's command (no key
  in it), and `compare` on synthetic outputs of both sides in their real
  layouts (`fixtures/compare`): every status counted, the statistics, one
  condition and one audio judge at a time, judge errors, the CLI.
- `test_eva_trial.py`, `test_eva_convert.py`: the driver's attempt loop, the
  parity guard against EVA's recorded session, the trajectory, the tasks.
- `test_eva_e2e.py`: one trial through `drive_here` with the real sidecar
  (its judges faked), a fake OpenAI Realtime server and a fake ElevenLabs
  server on localhost; a validated attempt, and a rerun after EVA's gate
  rejects one. Skips without Docker or `eva-base:039efa0a`. No API is called.
- The goldens (`test_eva_sidecar_goldens.py`, `test_eva_call_golden.py`,
  `test_eva_realtime_session.py`) check each piece against EVA's own code.
