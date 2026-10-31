# EVA parity

Does harbor-voice's EVA adapter reproduce EVA itself? Both sides ran the same
60 records with the same agent, caller agents and judges, and the scores are
compared in harbor's `parity_experiment.json` format.

## The subset

60 of EVA's 213 records, **20 per domain**, 3 trials each (EVA's k): 180
trials a side. Drawn by `eva-adapter sample --per-domain 20 --seed 300`, in
EVA's dataset order; both sides run exactly these ids:

| Domain | Records |
|---|---|
| airline | `1.1.3`, `1.2.2`, `1.3.2`, `2.1.6`, `2.2.5`, `2.3.4`, `2.4.2`, `3.1.5`, `3.3.4`, `4.1.2`, `4.2.1`, `4.2.5`, `5.1.5`, `5.2.1`, `5.2.2`, `6.1.1`, `6.3.4`, `7.1.1`, `7.2.2`, `7.3.1` |
| itsm | `13`, `16`, `19`, `20`, `22`, `30`, `33`, `40`, `45`, `46`, `52`, `56`, `58`, `60`, `62`, `76`, `81`, `90`, `96`, `102` |
| medical_hr | `1.2`, `4.2`, `5.2`, `6.2`, `9.1`, `11.2`, `A2`, `A5`, `D1.2`, `D2.2`, `D4.1`, `D5.3`, `D6.1`, `D7.2`, `D8.2`, `T1.2`, `T2.2`, `T3.2`, `T6.1`, `T7.1` |

How the draw works:

- **Equal per domain**, since each domain row is reported.
- **Stratified within each domain by EVA's record families**, every family at
  least once, the rest in proportion to its size:
  - airline: the id's first number (7 families; EVA has no category there,
    but family 7 is its adversarial callers);
  - itsm: EVA's `scenario_context.category`, a band of the numeric ids
    (single 29, double 14, triple 14, quadruple 14, adversarial 9);
  - medical_hr: the id's prefix, also EVA's category (single 24, `A`
    adversarial 9, `D` double 29, `T` triple 21).
- **Distinct scenarios first** within a family (`D4.1` before `D4.2`).
- **Every label present.** A label that cuts across families must appear at
  least once: ITSM's `escalate_to_live_agent` (11 of 80 records, in every
  category). This draw has two, `20` and `58`.
- **Nothing hand-picked.** The family rule was checked against EVA's datasets
  at the commit, and a test checks that `sample` draws the same records.

## How it was run

| | Original | Harbor |
|---|---|---|
| Harness | EVA's own CLI at `039efa0a`, unchanged (`eva` in EVA's image) | a harbor-voice job of `adapters/eva` tasks |
| Agent | `EVA_FRAMEWORK=openai_realtime`, `EVA_MODEL__S2S=gpt-realtime-1.5`, EVA's defaults | `openai.realtime/gpt-realtime-1.5` with EVA's settings (checked against EVA's own `session.update` every attempt) |
| Caller | EVA's ElevenLabs simulator on `EVA_EN_USER_F` / `EVA_EN_USER_M` | the same two agents through `elevenlabs.agents` |
| Judges | `EVA_MODEL_LIST` ([`judges.json`](judges.json)); audio judges below | the same, in every trial's sidecar |
| Validation, reruns | EVA's, ≤ 3 attempts | EVA's code in the sidecar, ≤ 3 attempts |
| Trials | `EVA_NUM_TRIALS=3` | the agent once per seed 0, 1, 2 (EVA's trial index) |

What differs is the call loop only: harbor's port of EVA's bridge and
Realtime server, and harbor's transports.

- **Setup:** 2026-10-29, both sides on Daytona, at the same time. Each side
  ran in six shares of 10 records, two calls at once per share (24
  ElevenLabs calls at peak). Clean condition, no perturbation.
- **The original command**, one per domain (`original.eva_command`):

  ```bash
  cd /app && EVA_FRAMEWORK=openai_realtime EVA_MODEL__S2S=gpt-realtime-1.5 \
    EVA_DOMAIN=airline EVA_RECORD_IDS=1.1.3,1.2.2,... EVA_NUM_TRIALS=3 \
    EVA_MAX_CONCURRENT_CONVERSATIONS=2 EVA_MAX_RERUN_ATTEMPTS=3 \
    EVA_USER_SIMULATOR__PROVIDER=elevenlabs EVA_LANGUAGE=en \
    EVA_RUN_ID=<label>-original-airline EVA_OUTPUT_DIR=/work/output \
    EVA_METRIC_CONFIGS='{"agent_speech_fidelity":{"audio_judge_model":"gemini-3.8-flash","judge_params":{"reasoning_effort":"low"}},"user_speech_fidelity":{...the same...}}' \
    EVA_MODEL__S2S_PARAMS="$(python3 -c '... {"model": "gpt-realtime-1.5", "api_key": os.environ["OPENAI_API_KEY"]} ...')" \
    eva
  ```

  Everything else is EVA's default. EVA reads the agent's key only from
  `s2s_params.api_key`, so that value is assembled when the command runs and
  EVA's `config.json` redacts it.
- **Audio judges:** `gemini-3.8-flash` on Vertex AI, reasoning effort `low`,
  on both sides alike (`EVA_METRIC_CONFIGS` for EVA,
  `EVA_SIDECAR_METRIC_CONFIGS` for harbor). EVA's published judge is
  `gemini-3-flash-preview` at minimal through the Gemini API, whose request
  limits are too low for a run this size; Vertex rejects minimal on this
  model. Harbor-vs-EVA parity is unaffected, since both sides judge alike, but
  speech fidelity (and EVA-A, which needs it) is not comparable with EVA's
  published numbers. The other judges are EVA's.
- **Compared** (`eva-adapter compare`), per domain and pooled:
  - `EVA-A_pass`, `EVA-X_pass`, `EVA-overall_pass`: each run's mean over its
    graded trials (run k = EVA's trial k), mean ± sample SEM over the 3 runs;
  - **harbor's criterion:** the two sides' run ranges overlap;
  - the components, and diagnostics: validation, reruns, end reasons, call
    length, response speed. No trial is dropped silently: missing, errored and never-validated
    trials are counted and listed, and enter no mean.

### Reproducing it

```bash
# 1. The records.
uv run eva-adapter sample --per-domain 20 --seed 300 -o records.json

# 2. Harbor's side, one dataset and job per domain (any harbor backend).
uv run eva-adapter convert --out datasets/parity-airline --domains airline \
    --records 1.1.3 1.2.2 ... --trials 3
uv run eva-adapter job-config --dataset datasets/parity-airline --trials 3 -n 2 \
    --job-name parity-airline -o job-airline.yaml   # [--env daytona ...]
uv run harbor-voice jobs start -c job-airline.yaml

# 3. EVA's side: the command above in EVA's image, one domain per run.

# 4. Compare harbor's job folders against EVA's run directories.
uv run eva-adapter compare --harbor jobs/parity-airline ... \
    --original eva-output/<label>-original-airline ... \
    --records records.json -o parity_experiment.json
```

- The same `EVA_MODEL_LIST` and audio-judge config on both sides.
- Every call holds one ElevenLabs Agents conversation: the harbor jobs' `-n`
  plus EVA's `EVA_MAX_CONCURRENT_CONVERSATIONS`, over every domain run at
  once, must stay within the workspace's concurrent-call limit, or refused
  calls skew the rerun and validation counts.
- A trial holds the sidecar image (~3.6 GB) and each attempt's record; set
  the disk with `job-config --override-storage-mb` where the backend's
  default is smaller.

## Results

Pass rates in %, mean ± SEM over the 3 runs, per-run values in brackets:

| Scope | Metric | Original | Harbor | Ranges overlap |
|---|---|---|---|---|
| airline | EVA-A | 36.7 ± 7.3 [25, 50, 35] | 40.0 ± 5.0 [35, 35, 50] | yes |
| airline | EVA-X | 65.0 ± 5.0 [60, 75, 60] | 55.0 ± 7.6 [65, 40, 60] | yes |
| airline | EVA-overall | 30.0 ± 5.0 [25, 40, 25] | 28.3 ± 6.0 [25, 20, 40] | yes |
| itsm | EVA-A | 68.3 ± 8.3 [60, 85, 60] | 66.7 ± 6.0 [70, 55, 75] | yes |
| itsm | EVA-X | 61.7 ± 7.3 [75, 60, 50] | 55.0 ± 5.0 [60, 60, 45] | yes |
| itsm | EVA-overall | 45.0 ± 5.8 [45, 55, 35] | 40.0 ± 2.9 [35, 40, 45] | yes |
| medical_hr | EVA-A | 36.7 ± 4.4 [35, 30, 45] | 28.9 ± 2.0 [25, 31.6, 30] | yes |
| medical_hr | EVA-X | 68.3 ± 4.4 [75, 60, 70] | 66.1 ± 4.6 [75, 63.2, 60] | yes |
| medical_hr | EVA-overall | 23.3 ± 6.7 [30, 10, 30] | 11.8 ± 4.4 [20, 10.5, 5] | yes |
| overall | EVA-A | 47.2 ± 4.3 [40, 55, 46.7] | 45.2 ± 3.3 [43.3, 40.7, 51.7] | yes |
| overall | EVA-X | 65.0 ± 2.9 [70, 65, 60] | 58.6 ± 4.0 [66.7, 54.2, 55] | yes |
| overall | EVA-overall | 32.8 ± 1.5 [33.3, 35, 30] | 26.8 ± 1.8 [26.7, 23.7, 30] | yes |

**Harbor's criterion holds on all 12 rows.**

- **EVA-A** agrees closely: overall −2.0 points; per record, 37 the same, 10
  more on harbor, 13 more on the original (sign-flip p = 0.64).
- **EVA-X and EVA-overall** are lower on harbor (−6.4 and −6.0 points; p =
  0.26 and 0.23). The difference is in turn-taking
  (0.814 vs 0.847).
- **medical_hr EVA-A** is the weakest row: on its 20 records harbor never
  passed more often, the original did on 5 (p = 0.062).
- No second original run was made, so the gaps are not yet set against
  EVA-vs-EVA variation.

Components, overall (mean ± SEM):

| Component | Original | Harbor |
|---|---|---|
| task completion | 0.756 ± 0.049 | 0.771 ± 0.021 |
| faithfulness | 0.336 ± 0.023 | 0.335 ± 0.012 |
| agent speech fidelity | 0.993 | 0.991 |
| turn taking | 0.847 ± 0.015 | 0.814 ± 0.018 |
| conciseness | 0.727 | 0.727 |
| conversation progression | 0.692 | 0.698 |
