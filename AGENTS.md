# harbor-voice — notes for coding agents

harbor-voice evaluates voice agents on [harbor](https://github.com/harbor-framework/harbor).
harbor is a pinned dependency (`harbor==0.25.0` in `pyproject.toml`), never a
copy: import `harbor.*` directly, and bump the pin deliberately.

## Layout

- `src/harbor_voice/` — the package: transports and providers, the caller
  simulator and its layer stack, ATIF-voice trajectories, metrics, the
  `harbor-voice` CLI, and `VoiceCallAgent` (a voice call as a harbor agent).
- `adapters/tau-voice/` — the tau-voice benchmark adapter (its own package).
- `adapters/eva/` — the EVA benchmark adapter (its own package).
- `tests/unit/voice/` — the package's tests; `tests/adapters/eva/`,
  `tests/adapters/tau-voice/` — each adapter's tests and fixtures.

## Commands

```bash
uv sync                                   # dev environment (adapter included)
uv run pytest                             # all tests
uv run ruff check src adapters tests && uv run ruff format src adapters tests
uv run ty check
```

Running voice tasks goes through harbor's own runner, mounted into our CLI:
`harbor-voice trials start -p <task> -m <s2s spec>`, `harbor-voice jobs start -c job.yaml`.

## Rules

- Never commit secrets; keys come from the environment (`.env` is ignored).
- Keep audio out of JSON: trajectories reference audio by content hash.
