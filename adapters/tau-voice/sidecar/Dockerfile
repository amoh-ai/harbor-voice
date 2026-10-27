# tau-voice runtime: tau2 at a pinned commit, its own Python.
# One image for every task; the task is chosen at POST /open.
FROM python:3.12-slim

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
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential portaudio19-dev \
    && git clone "${TAU2_BENCH_REPO}" "${TAU2_BENCH_ROOT}" \
    && git -C "${TAU2_BENCH_ROOT}" checkout "${TAU2_BENCH_COMMIT}" \
    && pip install --no-cache-dir "${TAU2_BENCH_ROOT}[voice]" \
    && apt-get purge -y --auto-remove git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.py evaluate.py ./
COPY test.sh /tests/test.sh

EXPOSE 8000
HEALTHCHECK --interval=2s --timeout=5s --retries=30 --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["python", "server.py"]
