# tau-voice parity

Does harbor-voice's tau-voice adapter reproduce tau2-bench's own voice
harness? Both sides ran the same 40 tasks with the same agent, caller,
voices, seeds and grader, in two speech complexities, and the scores are
compared in harbor's `parity_experiment.json` format.

Sierra's published τ-voice numbers use voices that are not available outside
Sierra, so no one else can reproduce them. Parity here is harbor's notion:
the original harness and the harbor side, both run under identical settings.

## The subset

40 of the 100 tasks in tau2's `test` split (the split the τ-voice
leaderboard reports), 3 runs each: 120 trials a side, per mode. Drawn by
`tau-voice sample --tau2-root <tau2-bench>`; both sides run exactly these
ids, which are in [`parity_tasks.json`](parity_tasks.json):

| Domain | Sampled | Of `test` | Tasks |
|---|---|---|---|
| airline | 8 | 20 | `2`, `18`, `19`, `24`, `29`, `35`, `45`, `48` |
| retail | 16 | 40 | `17`, `32`, `33`, `38`, `51`, `56`, `60`, `61`, `64`, `65`, `71`, `97`, `100`, `101`, `102`, `108` |
| telecom | 16 | 40 | 16 ids of the form `[issue]faults[PERSONA:…]`, in the file |

How the draw works:

- **Random, with a fixed seed:** 300, tau2's default run seed; ids in split
  order.
- **Proportional per domain:** 8 of 20, 16 of 40, 16 of 40.
- **Nothing hand-picked.** A test checks that `sample` draws the checked-in
  file from tau2's `test` split at the pinned commit. The converter's
  `--split parity` reads it, and tau2's run takes the same ids as
  `--task-ids`.

## How it was run

| | Original | Harbor |
|---|---|---|
| Harness | tau2's own `tau2 run` at `2174a60`, unchanged, in the tau2 image the tasks' sidecar is built from | a harbor-voice job of `adapters/tau-voice` tasks, one per mode |
| Agent | OpenAI Live: `--audio-native-provider openai_live --audio-native-model gpt-live-1 --live-config '{"backend_model": "gpt-6-luna"}'` | `openai.live/gpt-live-1` with `backend_model = "gpt-6-luna"` (`job-config --agent-config`) |
| Caller | tau2's voice user simulator, `--user-llm gpt-4.1-mini-2025-04-14`, real-time generation off (`--no-realtime-generation`) | harbor-voice's caller: tau2's turn policy, effects and phone line, its prompts built by tau2's code in the sidecar; the same `user_llm` |
| Listener decisions (`regular`) | `gpt-4.1`, which tau2 hard-codes | the same, by tau2's code in the sidecar |
| Voices | seven ElevenLabs voices made by tau2's `setup_voices` | the same seven |
| Effect seed | `PYTHONHASHSEED=0`, so tau2's `hash(task.id)` seed is reproducible (upstream leaves it random per process) | the same, in each trial's sidecar |
| Reviewer | Claude Opus 4.5 on Bedrock (`--review-model`) | the same |
| Hallucination reruns | up to 3, seed + 1000·n, the last attempt counts | the same |
| Trials | `--num-trials 3 --seed 300`: trial seeds 626729, 373753, 361454 | the agent once per seed |

What differs is the call loop only: harbor-voice's port of tau2's step loop
and its Live adapter, and harbor's transports.

- **Modes:** `control` (clean speech, patient turn-taking) and `regular`
  (tau-voice's leaderboard condition: accented personas, noise, bursts,
  muffling, frame drops, vocal tics, out-of-turn phrases, LLM-decided
  interruptions and backchannels), each compared on its own.
- **Setup:** 2026-10-28, both sides at the same time; harbor's on its Daytona
  backend. The 40 tasks ran as four disjoint quarters (airline 2, retail 4,
  telecom 4 each), each with harbor `-n 2` per mode and tau2
  `--max-concurrency 2` per domain, pooled by `compare`.
- **Completeness:** every trial was graded on both sides, 120 per mode per
  side, none errored.
- **The original command**, one per domain and mode
  (`original.tau2_command`), in `/opt/tau2-bench` in the tau2 image:

  ```bash
  tau2 run --domain airline --audio-native --audio-native-provider openai_live \
    --live-config '{"backend_model": "gpt-6-luna"}' --no-realtime-generation \
    --audio-native-model gpt-live-1 --speech-complexity control \
    --task-split-name test --num-trials 3 --seed 300 --max-concurrency 2 \
    --hallucination-retries 3 \
    --review-model bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0 \
    --auto-resume --save-to <label>-control-airline \
    --task-ids 2 18 19 24 29 35 45 48 --user-llm gpt-4.1-mini-2025-04-14
  ```

  Keys come from its environment: `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`,
  the seven `TAU2_VOICE_ID_*`, and `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (and `AWS_REGION_NAME`, the same) for
  the reviewer. The image sets `PYTHONHASHSEED=0`.
- **Compared** (`tau-voice compare`), per mode, per domain and overall:
  - average reward (Pass^1): each run's mean over its graded tasks, mean ±
    sample SEM over the 3 runs, with the per-run values;
  - **harbor's criterion:** the two sides' run ranges overlap;
  - Pass^k for k = 1, 2, 3 (tau2's estimator); tau2's interaction metrics,
    computed by tau2's code on both sides (`sidecar/interaction_metrics.py`);
    and diagnostics: hallucination reruns, terminations, call length,
    errored trials, tokens. Errored trials are counted and enter no mean.

### Reproducing it

```bash
# 1. The tasks (writes parity_tasks.json; the checked-in file is this draw).
uv run tau-voice sample --tau2-root ../tau2-bench

# 2. Harbor's side, one dataset and job per mode (any harbor backend).
uv run tau-voice --tau2-root ../tau2-bench --output-dir datasets/parity-control \
    --split parity --complexity control --user-llm gpt-4.1-mini-2025-04-14
uv run tau-voice job-config --dataset datasets/parity-control \
    --model openai.live/gpt-live-1 --agent-config '{"backend_model": "gpt-6-luna"}' \
    --trials 3 -n 2 --job-name parity-control -o job-control.yaml   # [--env daytona]
uv run harbor-voice jobs start -c job-control.yaml
#    ... and the same for --complexity regular.

# 3. tau2's side: the command above in the tau2 image (`tau-voice build-image`
#    builds it), once per domain and mode; tau2 writes data/simulations/<save-to>.

# 4. tau2's interaction metrics for both sides, in the tau2 image:
#    python sidecar/interaction_metrics.py --complexity control \
#        --harbor <job folder> --original <tau2 save folder>... -o metrics-control.json

# 5. Compare, per mode.
uv run tau-voice compare --harbor jobs/parity-control --original <tau2 save folders> \
    --runs 3 --complexity control --metrics metrics-control.json \
    --out parity_experiment-control.json
```

- The same caller model, voices and reviewer on both sides.
- Every caller line is an ElevenLabs TTS request: harbor's `-n` per mode
  plus tau2's `--max-concurrency` per domain, over every run at once, must
  stay within the ElevenLabs workspace's concurrency, or refused requests
  are retried and the agent hears the retries as silence.

## Results

Average reward in %, mean ± SEM over the 3 runs, per-run values in brackets:

| Mode | Scope | Original | Harbor | Ranges overlap |
|---|---|---|---|---|
| control | airline | 62.5 ± 7.2 [75, 62.5, 50] | 50.0 ± 0.0 [50, 50, 50] | yes (touch at 50) |
| control | retail | 66.7 ± 5.5 [56.25, 75, 68.75] | 62.5 ± 10.8 [62.5, 43.75, 81.25] | yes |
| control | telecom | 39.6 ± 5.5 [50, 31.25, 37.5] | 45.8 ± 5.5 [37.5, 43.75, 56.25] | yes |
| control | overall | 55.0 ± 1.4 [57.5, 55, 52.5] | 53.3 ± 6.0 [50, 45, 65] | yes |
| regular | airline | 50.0 ± 7.2 [50, 37.5, 62.5] | 58.3 ± 4.2 [50, 62.5, 62.5] | yes |
| regular | retail | 54.2 ± 2.1 [56.25, 56.25, 50] | 43.8 ± 3.6 [50, 37.5, 43.75] | yes (touch at 50) |
| regular | telecom | 41.7 ± 2.1 [43.75, 37.5, 43.75] | 45.8 ± 4.2 [50, 50, 37.5] | yes |
| regular | overall | 48.3 ± 1.7 [50, 45, 50] | 47.5 ± 1.4 [50, 47.5, 45] | yes |

**Harbor's criterion holds on all 8 rows.**

- **Overall, the two sides agree in both modes:** −1.7 points in `control`
  and −0.8 in `regular`, inside the run-to-run spread.
- **Per task the disagreements are symmetric:** in `control`, 22 of 40 tasks
  have the same success count on both sides, 9 more on harbor, 9 more on the
  original (paired sign-flip p = 0.88); in `regular`, 28 the same, 5 and 7
  (p = 1.00).
- **Per domain the gaps are larger, up to 12.5 points** (airline `control`;
  retail `regular` 10.4), and two rows' ranges only touch. The domains have
  8–16 tasks each, and one side's runs swing by up to 37.5 points (harbor
  retail `control`).
- **Precision is the limit:** 40 tasks × 3 runs can detect a gap of about 8
  points (8.5 in `control`, 7.3 in `regular`).
- **One telecom task is lopsided** across both modes (harbor 5 of 6, the
  original 0 of 6); with 40 tasks one such task can arise by chance.
- **Not comparable to the τ-voice leaderboard** (gpt-live-1, 78.9%), which
  uses another backend (gpt-6 astra, medium reasoning), tau2's default
  caller and the full `test` split. The claim here is parity between the two
  harnesses under one setting.

Pass^k (tau2's estimator), overall:

| Mode | Side | Pass^1 | Pass^2 | Pass^3 |
|---|---|---|---|---|
| control | original | 0.550 | 0.425 | 0.350 |
| control | harbor | 0.533 | 0.408 | 0.350 |
| regular | original | 0.483 | 0.375 | 0.325 |
| regular | harbor | 0.475 | 0.383 | 0.350 |

tau2's interaction metrics, overall, computed by tau2's code on both sides:

| | control original | control harbor | regular original | regular harbor |
|---|---|---|---|---|
| Response latency L_R (s) | 3.94 | 4.04 | 3.86 | 3.63 |
| Yield latency L_Y (s) | 0.77 | 0.77 | 0.83 | 0.78 |
| Response rate R_R | 0.79 | 0.78 | 0.75 | 0.74 |
| Yield rate R_Y | 0.58 | 0.55 | 0.62 | 0.66 |
| Agent interruptions I_A | 1.81 | 1.96 | 1.67 | 1.56 |
| Selectivity S_BC / S_VT / S_ND | | | 0.87 / 0.34 / 0.54 | 0.83 / 0.38 / 0.55 |
