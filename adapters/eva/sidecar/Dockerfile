# EVA sidecar: EVA's own image at the pinned commit, plus server.py.
# One image for every task; the record is chosen at POST /open, and each
# trial gets its own container (one session per container, as tau-voice).
#
# EVA_BASE is EVA's image at the commit the adapter reproduces
# (eva_adapter.EVA_COMMIT), from either of two builds of ServiceNow's public
# repo at that commit (eva_adapter.images):
#
#   eva-base:039efa0a  (the default) EVA's own Dockerfile on EVA's repo:
#                          docker build -t eva-base:039efa0a \
#                            https://github.com/ServiceNow/eva.git#039efa0a80916a2e59ca83b330f6ed6fd63ab4db
#                      or, from a dev checkout of the commit,
#                          docker build -t eva-base:039efa0a eva
#                      `eva-adapter build-image` builds it if it is missing.
#   eva-from-source    the stages below: EVA's Dockerfile stage for stage, its
#                      build context a checkout of the same commit. Needs
#                      nothing local, so a task can build it wherever it runs
#                      (the converter passes it unless --prebuilt).
#
# Either holds EVA's locked venv (/opt/venv, from uv.lock with --frozen) and
# its repo layout under /app (src/ configs/ data/ assets/ scripts/), which
# EVA reads by relative path. With BuildKit (Docker's default builder) the
# stages of the base not chosen are never built.
ARG EVA_BASE=eva-base:039efa0a
ARG EVA_REPO=https://github.com/ServiceNow/eva.git
ARG EVA_COMMIT=039efa0a80916a2e59ca83b330f6ed6fd63ab4db

# ---------------------------------------------------------------- eva-from-source
# EVA's source at the commit, fetched by its full id and checked.
FROM mirror.gcr.io/library/python:3.11-slim AS eva-source
ARG EVA_REPO
ARG EVA_COMMIT
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && git init -q /eva \
    && git -C /eva remote add origin "${EVA_REPO}" \
    && git -C /eva fetch -q --depth 1 origin "${EVA_COMMIT}" \
    && git -C /eva checkout -q FETCH_HEAD \
    && test "$(git -C /eva rev-parse HEAD)" = "${EVA_COMMIT}" \
    && rm -rf /eva/.git /var/lib/apt/lists/*

# EVA's Dockerfile copies uv from ghcr.io/astral-sh/uv:latest. Only the
# installer floats there (the lock is frozen); here it is pinned to the uv that
# built eva-base:039efa0a, so the two builds install with the same uv.
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

# EVA's Dockerfile @039efa0a from here on, stage for stage; `COPY x` from its
# build context reads `COPY --from=eva-source /eva/x`. python:3.11-slim is
# Docker Hub's official image, pulled through Google's mirror (concurrent
# builds from shared cloud IPs hit Docker Hub's pull limit; tau-voice's note).

# Stage 1: Deps
FROM mirror.gcr.io/library/python:3.11-slim AS eva-deps
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=eva-source /eva/pyproject.toml /eva/uv.lock /eva/README.md ./
RUN mkdir -p src/eva && echo '__version__ = "0.0.0"' > src/eva/__init__.py
RUN uv venv /opt/venv && \
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-cache

# Stage 2: Builder
FROM mirror.gcr.io/library/python:3.11-slim AS eva-builder
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=eva-deps /opt/venv /opt/venv
COPY --from=eva-source /eva/pyproject.toml /eva/README.md ./
COPY --from=eva-source /eva/src/ ./src/
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps .

# Stage 3: Runtime
FROM mirror.gcr.io/library/python:3.11-slim AS eva-from-source
ARG EVA_COMMIT
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=eva-deps /opt/venv /opt/venv
COPY --from=eva-builder /opt/venv/lib/python3.11/site-packages/eva /opt/venv/lib/python3.11/site-packages/eva
COPY --from=eva-builder /opt/venv/lib/python3.11/site-packages/eva-*.dist-info /opt/venv/lib/python3.11/site-packages/
COPY --from=eva-builder /opt/venv/bin/eva /opt/venv/bin/eva
COPY --from=eva-source /eva/scripts/docker_eva_wrapper.sh /usr/local/bin/eva
RUN chmod +x /usr/local/bin/eva
ENV PATH="/usr/local/bin:/opt/venv/bin:$PATH"
COPY --from=eva-source /eva/src/ ./src/
COPY --from=eva-source /eva/scripts/ ./scripts/
COPY --from=eva-source /eva/configs/ ./configs/
COPY --from=eva-source /eva/data/ ./data/
COPY --from=eva-source /eva/assets/ ./assets/
RUN groupadd --gid 1000 eva && \
    useradd --uid 1000 --gid eva --create-home eva
RUN chmod -R a+rwX /opt/venv
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
# EVA's provenance args, as its scripts/docker_build.sh sets them for a clean
# checkout of the commit; /health reports GIT_COMMIT_SHA.
ENV GIT_COMMIT_SHA=${EVA_COMMIT} \
    GIT_BRANCH=main \
    GIT_DIRTY=false \
    GIT_DIFF_HASH=
USER eva
ENTRYPOINT ["eva"]

# ---------------------------------------------------------------- the sidecar
FROM ${EVA_BASE}

# EVA's runtime layout: cwd at its repo root, its own venv. Run directories go
# to a container-local path the base image's non-root user can write.
WORKDIR /app
ENV EVA_SIDECAR_OUTPUT_DIR=/tmp/eva-sidecar \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

COPY server.py aggregate.py /opt/eva-sidecar/

EXPOSE 8000
HEALTHCHECK --interval=2s --timeout=5s --retries=30 --start-period=10s \
    CMD ["/opt/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

# The base's entrypoint is EVA's CLI wrapper (it tries to install Krisp);
# the sidecar runs its own server instead.
ENTRYPOINT ["/opt/venv/bin/python", "/opt/eva-sidecar/server.py"]
