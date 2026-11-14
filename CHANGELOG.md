# Changelog

harbor-voice and its two adapters (`harbor-voice-tau-voice-adapter`,
`harbor-voice-eva-adapter`) are released together, at one version. Dataset
versions are listed in
[harbor-voice-datasets](https://github.com/amoh-ai/harbor-voice-datasets#images).

## 0.1.2 (2026-11-14)

### New
- **More providers.** Speech-to-speech: Google Gemini Live (`google.live`), xAI Grok (`xai.realtime`), Deepgram Voice Agent (`deepgram.agent`), and a self-hosted realtime server (`selfhosted.realtime`). STT and TTS: Deepgram, Cartesia, Fish Audio, Sarvam, Microsoft Azure STT, Meta STT, xAI and OpenRouter. LLMs: LiteLLM, OpenRouter and self-hosted.
- **Cascade agents on tau-voice.** An STT, an LLM and a TTS can be the agent under test (`tau-voice job-config --cascade`), as well as a speech-to-speech model.
- **OpenAI Responses API.** `api = "responses"` on `openai.llm`, with reasoning carried across tool rounds.
- **x86 and ARM images, datasets 0.1.1.** Every image is built for `linux/amd64` and `linux/arm64`. Datasets 0.1.1 pins them.
- **Omni + TTS cascade.** An omni model hears the caller and writes the reply, and a TTS speaks it: `selfhosted.omni` (vLLM) and `openrouter.omni`. Runs on the built-in call loop.
- **Bedrock chat.** `aws.bedrock-llm` holds a whole conversation (Converse), so it can drive the tau-voice cascade.
- **OpenAI instructions.** `system` and `system_role` (`system`, `developer`, or the Responses API's `instructions`) on `openai.llm`.
- **Self-hosted providers.** Speech and LLM servers behind your own URL, Modal included.
- **Persona caller.** It gets its brief, hears the agent (through an optional `[caller] stt` for a speech-to-speech agent), and can end the call with `[END CALL]`.
- **EVA judge routes.** `eva-adapter judges --gemini api-key|vertex|vertex-json`: Gemini on the API, on Vertex with ADC, or on Vertex with a service account (`VERTEXAI_CREDENTIALS`).

### Fixed
- Full duplex waits 1 s of quiet before ending a turn, so replies are no longer cut off. The silence after a reply is no longer counted as part of it, or as bleed.
- A failing cascade slot ends the call at once, instead of waiting out every turn's timeout or going unreported after the last turn.
- Provider keys in vendor headers can't follow a redirect to another host. Redirects are followed only by self-hosted servers.
- Bedrock chat no longer sends a blank system text. A non-JSON 200 reply now raises a clear MALFORMED error.
- The persona caller's lines stay paired with the right agent replies after probes. A second run starts clean. Clients are closed when a call ends.
- The keys file in the sandbox is removed even if loading it fails.
- Deepgram TTS retries a dropped stream before any audio was sent.

### Datasets 0.1.2
All eight datasets are registered at [harbor-voice-datasets v0.1.2](https://github.com/amoh-ai/harbor-voice-datasets/releases/tag/v0.1.2). Only the EVA tasks change: they pass `VERTEXAI_CREDENTIALS` to the sidecar. The tau-voice tasks are the same as 0.1.1.

```bash
eva-adapter job-config --hub-dataset eva@0.1.2 --model openai.realtime/gpt-realtime-1.5 -o eva.yaml
```

The images are unchanged from 0.1.1, built for amd64 and arm64:

| Image | Tag |
|---|---|
| tau-voice sidecar | `ghcr.io/amoh-ai/tau-voice-runtime:2174a60-ce6c029644` |
| tau-voice `main` | `ghcr.io/amoh-ai/tau-voice-main:413a7f6ab7` |
| EVA sidecar | `ghcr.io/amoh-ai/eva-sidecar:039efa0a-077f8e4f23` |
| EVA `main` | `ghcr.io/amoh-ai/eva-main:413a7f6ab7` |

### Housekeeping
- The package ships `py.typed`. CI also tests Python 3.12.
- The lock file is refreshed: filelock 4.2.10, dnspython 2.9.2, opentelemetry 1.45.2. This clears the open Dependabot alerts.
- Added `SECURITY.md` (report privately through the Security tab) and a weekly Dependabot config.

## 0.1.1 (2026-11-12)

### Fixed
- The caller's L2 speaking rate and its dynamic muffling. These are the persona speech perturbations.

## 0.1.0 (2026-11-04)

The first release on PyPI: the library, the tau-voice and EVA adapters, the
release checks and the release workflow.
