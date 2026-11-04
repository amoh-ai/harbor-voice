# Running on cloud

The orchestrator stays on your machine. The backend creates one environment
per trial and deletes it afterwards. Choose the backend with `job-config --env
<type>` (both adapters), or `--env` on `trials start`. Settings that belong to
your account go in your own copy of the job config, never in the tasks.

The benchmark runs in a sidecar container next to the agent's, so a backend
needs docker-compose support. Local Docker, Daytona and EC2 do; more backends
are on the way.

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
