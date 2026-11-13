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
| Speech to speech | `openai.realtime`, `openai.live`, `aws.nova-sonic`, `elevenlabs.agents`, `google.live`, `xai.realtime`, `deepgram.agent`, `selfhosted.realtime` |
| Speech to text, for cascades | `aws.nova-stt`, `aws.voxtral`, `deepgram.stt`, `cartesia.stt`, `sarvam.stt`, `fishaudio.stt`, `xai.stt`, `meta.stt`, `microsoft.stt`, `openrouter.stt`, `selfhosted.stt` |
| LLM, for cascades | `openai.llm`, `aws.bedrock-llm`, `litellm.llm`, `selfhosted.llm` |
| Text to speech, for cascades | `elevenlabs.tts`, `aws.nova-tts`, `deepgram.tts`, `cartesia.tts`, `sarvam.tts`, `fishaudio.tts`, `xai.tts`, `openrouter.tts`, `selfhosted.tts` |
| Offline, for tests | `offline.scripted-stt`, `offline.echo-llm`, `offline.scripted-llm`, `offline.tool-echo-llm`, `offline.tone-tts` |

- `google.live` signs in with an AI Studio key (`GEMINI_API_KEY`), or with
  Vertex AI when `GOOGLE_GENAI_USE_VERTEXAI=true` (Application Default
  Credentials, `GOOGLE_CLOUD_PROJECT`, the `google` extra).
- `litellm.llm/<litellm model>` routes through litellm, which harbor already
  depends on: `litellm.llm/gemini/...`, `litellm.llm/vertex_ai/...`,
  `litellm.llm/xai/...`, `litellm.llm/bedrock/...`,
  `litellm.llm/openrouter/<model>` (`OPENROUTER_API_KEY`), and any vLLM, SGLang or
  Ollama server as `litellm.llm/hosted_vllm/<model>` with `api_base`.
- `microsoft.stt` is MAI-Transcribe (default `MAI-Transcribe-2`) through an
  Azure Speech resource: `AZURE_SPEECH_KEY` and `AZURE_SPEECH_ENDPOINT`
  (`https://<resource>.cognitiveservices.azure.com`, in a region that offers it).
- `selfhosted.stt` and `selfhosted.tts` call an OpenAI-compatible speech
  server (vLLM, vLLM-Omni, SGLang): `/v1/audio/transcriptions` and
  `/v1/audio/speech`, at `SELFHOSTED_STT_BASE_URL` / `SELFHOSTED_TTS_BASE_URL`.
- `selfhosted.llm/<model>` calls a server's `/v1/chat/completions` directly
  (vLLM, SGLang, Ollama), at `SELFHOSTED_LLM_BASE_URL`, with no litellm in
  between.
- `openrouter.stt/<model>` and `openrouter.tts/<model>` are OpenRouter's
  speech models (`OPENROUTER_API_KEY`), e.g. `openrouter.stt/deepgram/nova-3`
  or `openrouter.tts/elevenlabs/eleven-flash-v2.5` with `voice = "sarah"`.
  Most speech models need a voice; each model's voices are in
  `GET https://openrouter.ai/api/v1/models?output_modalities=speech`.
  Models OpenRouter serves only as MP3 (MiniMax) are not supported.
- `selfhosted.realtime/<model>` is a speech-to-speech model behind vLLM-Omni's
  `/v1/realtime` (Qwen3-Omni), at `SELFHOSTED_REALTIME_BASE_URL`. All four
  send `SELFHOSTED_API_KEY` as a bearer token when it is set. Inside a
  trial's sandbox `localhost` is the sandbox itself, so a base URL pointing
  there is refused before the call starts: use `host.docker.internal` for
  local Docker, or the server's public URL.
- Serving Qwen3-Omni-30B-A3B with vLLM-Omni takes two 80 GB GPUs: it puts the
  thinker on GPU 0 and the talker and code2wav on GPU 1. Install
  `vllm-omni[server-vad]`, or the server refuses server VAD and the model
  never hears the caller finish. On an image without the CUDA toolkit, set
  `VLLM_USE_FLASHINFER_SAMPLER=0`, or vLLM tries to compile FlashInfer's
  sampler at startup and fails.

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

The call itself is `out/call.wav`, both parties on one timeline; the
trajectory names it by hash, so `harbor-voice listen out/trajectory.json` finds it.

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
| `google` | Gemini Live on Vertex AI |
| `daytona`, `ec2`, `gke`, `langsmith`, `modal`, `prime`, `sbx`, `vercel` | harbor's cloud backend of that name |
