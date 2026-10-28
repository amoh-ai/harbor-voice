# tau-voice runtime: tau2 at a pinned commit, its own Python.
# One image for every task; the task is chosen at POST /open.
# Docker Hub's official image through Google's mirror: dozens of trials
# building at once from shared cloud IPs hit Docker Hub's pull limit (429).
FROM mirror.gcr.io/library/python:3.12-slim

ARG TAU2_BENCH_REPO="https://github.com/sierra-research/tau2-bench.git"
# Pinned, unlike upstream's tau3 runtime (which clones HEAD): parity needs the
# original side and this image on the same tau2.
ARG TAU2_BENCH_COMMIT="2174a603f6d014ef94473ffa95957f6ce27100db"

ENV TAU2_BENCH_ROOT=/opt/tau2-bench \
    TAU2_DATA_DIR=/opt/tau2-bench/data \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# portaudio + a compiler: tau2[voice] pulls pyaudio, which has no Linux wheel.
# tau2[live] is OpenAI Live's WebRTC (aiortc), for the original side's runs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential portaudio19-dev \
    && git clone "${TAU2_BENCH_REPO}" "${TAU2_BENCH_ROOT}" \
    && git -C "${TAU2_BENCH_ROOT}" checkout "${TAU2_BENCH_COMMIT}" \
    && pip install --no-cache-dir "${TAU2_BENCH_ROOT}[voice,live]" \
    && apt-get purge -y --auto-remove git build-essential \
    && rm -rf /var/lib/apt/lists/*

# tau2's Live provider sends "OpenAI-Alpha: quicksilver=v3" when it creates a
# session. The documented API takes no such header, and with it a project
# without that preview is refused (live_api_access_denied); harbor-voice's
# transport never sends it. Removed here so both parity sides call the same
# API; the build fails if the line is not exactly where it was.
RUN f="$(python -c "import importlib.util as u; print(u.find_spec('tau2').submodule_search_locations[0])")/voice/audio_native/openai/live_provider.py"     && grep -q '"OpenAI-Alpha": "quicksilver=v3",' "$f"     && sed -i '/"OpenAI-Alpha": "quicksilver=v3",/d' "$f"     && ! grep -q 'OpenAI-Alpha' "$f"

# tau2 seeds a call's effects with seed + hash(task.id), and Python's string
# hash is randomised per process unless pinned: pinned on both parity
# sides, so effect timing is reproducible.
ENV PYTHONHASHSEED=0

# tau2 installs numpy/scipy unpinned; the vendored effects in harbor-voice
# run on these exact versions (adapters/tau-voice/pyproject.toml).
RUN pip install --no-cache-dir numpy==2.5.4 scipy==1.18.1

WORKDIR /app
COPY server.py evaluate.py ./

EXPOSE 8000
HEALTHCHECK --interval=2s --timeout=5s --retries=30 --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["python", "server.py"]
