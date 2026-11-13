# Running on cloud

The orchestrator stays on your machine. The backend creates one environment
per trial and deletes it afterwards. Choose the backend with `job-config --env
<type>` (both adapters), or `--env` on `trials start`. Settings that belong to
your account go in your own copy of the job config, never in the tasks.

The benchmark runs in a sidecar container next to the agent's, so a backend
needs docker-compose support. tau-voice has run on local Docker, Daytona,
EC2, Modal, Docker Sandboxes (`sbx`), GKE, Prime Intellect and Vercel.
LangSmith's sandbox could not start the sidecar, so it is not supported yet.

Each backend is harbor's; its extra installs its SDK (`uv sync` installs them
all). Account settings go under `environment.kwargs` in the job config.

## Daytona

Set `DAYTONA_API_KEY`. Each trial is a Docker-in-Docker sandbox sized from
the task.

```bash
tau-voice job-config --hub-dataset tau-voice-control@0.1.0 \
    --model openai.realtime/gpt-realtime-1.5 --env daytona -o job.yaml
harbor-voice jobs start -c job.yaml
```

## EC2

Sign in with any standard AWS credential (a profile, SSO, environment
variables). harbor launches a VM per trial, connects over SSH and terminates
the VM afterwards. You need a key pair and a security group that allows SSH
from your machine. Add your account's settings to the job config:

```yaml
environment:
  type: ec2
  kwargs:
    region: us-east-1
    aws_profile: <your profile>
    ami_id: <an Ubuntu 24.04 AMI>
    instance_type: m7i-flex.large
    key_name: <your key pair>
    ssh_key_path: ~/.ssh/<your key>.pem
    security_group_ids: [<sg allowing SSH from you>]
```

## Modal, Docker Sandboxes, GKE, Prime, Vercel

| Backend | `--env` | Sign-in | Job config `kwargs` |
|---|---|---|---|
| Modal | `modal` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | none |
| Docker Sandboxes | `sbx` | install the `sbx` CLI and run `sbx login` once | `target: local` |
| GKE | `gke` | `gcloud` application default credentials | `cluster_name`, `region`, `namespace`, `registry_location`, `registry_name`, `project_id` |
| Prime Intellect | `prime` | `PRIME_API_KEY` | `team_id` |
| Vercel | `vercel` | `VERCEL_TOKEN`, `VERCEL_TEAM_ID`, `VERCEL_PROJECT_ID` | `sandbox_lifetime_seconds` longer than a call (2400 works) |

## Faster startup

The published datasets name prebuilt images on `ghcr.io/amoh-ai`, so a fresh
environment pulls them. Tasks you convert yourself build their images in each
environment by default, which takes minutes on a fresh cloud sandbox. To
avoid that, build the images once, push them to a registry the backend can
pull from, and convert against it, from a checkout:

```bash
uv run tau-voice build-image --registry <prefix> --push
uv run tau-voice --tau2-root ../tau2-bench --output-dir datasets/tau-voice --registry <prefix>

uv run eva-adapter build-image --registry <prefix> --push
uv run eva-adapter convert --out datasets/eva --registry <prefix>
```

## Concurrency

Calls run in real time, a few minutes each, so concurrency (`-n`) sets a
job's wall time. Keep it within your providers' limits on concurrent
sessions (ElevenLabs' especially), or calls are refused.
