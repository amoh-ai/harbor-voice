# The harbor-voice library

`src/harbor_voice/` is usable beyond the two benchmark adapters: providers
for voice agents, a built-in call loop with a simulated caller, ATIF-voice
trajectories and voice metrics.

## Providers

A transport is always full-duplex audio frames; half duplex is a policy on
top. Optional operations (truncation, events, text) are declared per provider
and checked, not probed.

| Kind | Providers |
|---|---|
| Speech to speech | `openai.realtime`, `openai.live`, `aws.nova-sonic`, `elevenlabs.agents` |
| Speech to text, for cascades | `aws.nova-stt`, `aws.voxtral` |
| LLM, for cascades | `openai.llm`, `aws.bedrock-llm` |
| Text to speech, for cascades | `elevenlabs.tts`, `aws.nova-tts` |
| Offline, for tests | `offline.scripted-stt`, `offline.echo-llm`, `offline.scripted-llm`, `offline.tool-echo-llm`, `offline.tone-tts` |

More providers are on the way.

`harbor-voice caps` prints each provider's capabilities and which credentials
are missing. The conformance suite checks a provider against what it
declares: `harbor_voice.conformance.run_conformance`, run from a test against
a live session (the module's docstring shows how).

## The built-in call loop

The benchmark adapters bring their own caller and call loop. A task that
declares a `[caller]` instead runs on harbor-voice's own. It has a seeded
simulated caller (a script, a persona LLM or recorded clips), half- or
full-duplex turn-taking, and perturbation layers. Each layer is recorded with
what it achieved, not just what was asked for, and the clean audio is kept
for comparison.

```
L1 PERSONA      who speaks          voice, persona brief
L2 SPEECH       how they speak      disfluency, rate
L3 ENVIRONMENT  the room            continuous background bed
L4 CHANNEL      the wire            codec, loss, jitter, phone line
L5 CONVERSATION when they speak     barge-in, backchannel, interjections
L6 WORLD        the agent's tools   latency, errors, outages (not the caller's audio)
```

This loop runs from Python; harbor's runner takes only tasks with a `[call]`
driver, as the adapters' are. A cascade on the offline providers needs no
credentials:

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

`out/` also gets `measurement.json` and `validity.json`, and
`harbor-voice show out/trajectory.json` summarises the call.

## Voice metrics

Turn timing, overlap and barge-in, said-vs-heard WER and run validity. A
metric abstains rather than scoring 0 when a call can't support it.

## Commands

| Command | What it does |
|---|---|
| `harbor-voice caps` | provider capability matrices, and which credentials are missing |
| `harbor-voice profiles` | the named caller conditions, their layers, and the metrics each moves |
| `harbor-voice metrics` | the registered metrics, what they need, and which layers move them |
| `harbor-voice doctor` | preflight a configuration: what will report, abstain, or be blocked |
| `harbor-voice report <job>` | roll up built-in-loop trials and diff each condition against a baseline |
| `harbor-voice drive <task>` | run a task's call driver in this process; what harbor-voice's agent runs inside a trial's environment |

## Extras

`uv sync` installs every extra. If you depend on harbor-voice as a library,
these are optional:

| Extra | For |
|---|---|
| `live` | OpenAI Live, over WebRTC |
| `aws` | Amazon Nova Sonic and the Bedrock providers |
| `daytona` | harbor's Daytona backend |
| `ec2` | harbor's EC2 backend |
