"""Publishing harbor-voice datasets: images, registry entries, release checks.

The hub follows harbor's own layout. Task folders live in the datasets repo
(``amoh-ai/harbor-voice-datasets``, as ``datasets/<name>/<task>/``);
``registry.json`` at this repo's root lists each published dataset version,
every task pinned to the datasets-repo commit it was released at; the images
the tasks name are on ``ghcr.io/amoh-ai``, by digest. CONTRIBUTING's
"Publishing a dataset version" has the whole sequence::

    uv run python scripts/hub.py images [--force] [--dry-run]
    uv run python scripts/hub.py register --datasets-repo ../harbor-voice-datasets \\
        --commit <sha> --version 0.1.0 --dataset eva "<description>" ...
    uv run python scripts/hub.py check [--against <git ref>]
                                       [--datasets-repo <dir> [--anonymous]]

``images`` builds and pushes what isn't published yet and records each
image's digest in ``adapters/<name>/images.json``; a tag already in the
registry is never pushed again. ``register`` appends one entry per dataset
to ``registry.json``. ``check`` validates ``registry.json`` and the digest
records offline (and, with ``--against``, that published entries were only
ever added); with ``--datasets-repo`` it runs the release checks too.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry.json"
HUB_REGISTRY = "ghcr.io/amoh-ai"
DATASETS_GIT_URL = "https://github.com/amoh-ai/harbor-voice-datasets.git"
MAIN = "main"
"""The datasets repo's branch every pinned commit must stay on."""

SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)


class HubError(Exception):
    """A step that cannot go on; the message says why."""


def adapters() -> dict[str, ModuleType]:
    """Each adapter's ``images`` module: ``refs``, ``build``, ``published``, ``RECORD``."""
    from eva_adapter import images as eva
    from tau_voice import images as tau_voice

    return {"tau-voice": tau_voice, "eva": eva}


# ------------------------------------------------------------------- git


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if check and result.returncode:
        raise HubError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def git_ok(repo: Path, *args: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_commit(repo: Path, commit: str) -> str:
    """``commit`` as a full SHA of a commit in ``repo``; empty if it isn't one."""
    out = git(
        repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}", check=False
    )
    return out.strip()


def remote_refs(repo: Path, git_url: str) -> dict[str, str]:
    """The datasets repo's branches and tags as the remote has them, each
    annotated tag peeled to its commit."""
    refs: dict[str, str] = {}
    for line in git(repo, "ls-remote", "--heads", "--tags", git_url).splitlines():
        sha, _, ref = line.partition("\t")
        refs[ref.removesuffix("^{}")] = sha
    return refs


def release_problems(
    repo: Path, sha: str, git_url: str, refs: dict[str, str] | None = None
) -> list[str]:
    """Why ``sha`` can't be pinned yet: it must be on the remote's ``main``
    and tagged there, so it stays reachable for good."""
    refs = remote_refs(repo, git_url) if refs is None else refs
    problems = []
    tip = refs.get(f"refs/heads/{MAIN}")
    if tip is None:
        problems.append(f"{git_url} has no {MAIN} branch")
    elif not git_ok(repo, "cat-file", "-e", f"{tip}^{{commit}}"):
        problems.append(f"{MAIN}'s tip {tip[:12]} is not in {repo}: fetch first")
    elif not git_ok(repo, "merge-base", "--is-ancestor", sha, tip):
        problems.append(f"{sha[:12]} is not on {git_url} {MAIN}")
    tags = sorted(
        ref.removeprefix("refs/tags/")
        for ref, target in refs.items()
        if ref.startswith("refs/tags/") and target == sha
    )
    if not tags:
        problems.append(
            f"no tag on {git_url} points at {sha[:12]}: tag it (git tag v<version> "
            f"{sha[:12]} && git push origin v<version>)"
        )
    return problems


def read_files(repo: Path, sha: str, paths: Iterable[str]) -> dict[str, bytes | None]:
    """Each path's bytes at ``sha`` (``None`` where it doesn't exist), in one
    ``git cat-file --batch``."""
    paths = list(paths)
    if not paths:
        return {}
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{sha}:{path}\n" for path in paths).encode(),
        capture_output=True,
        check=True,
    )
    out, at, files = result.stdout, 0, {}
    for path in paths:
        end = out.index(b"\n", at)
        header = out[at:end].decode()
        at = end + 1
        if header.endswith(" missing"):
            files[path] = None
            continue
        size = int(header.split()[2])
        files[path] = out[at : at + size]
        at += size + 1
    return files


# ---------------------------------------------------------------- images


def remote_digest(ref: str) -> str | None:
    """The digest ``ref``'s registry serves for it, ``None`` if it has none."""
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--format"]
        + ["{{.Manifest.Digest}}", ref],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    if "not found" in result.stderr.lower():
        return None
    raise HubError(f"cannot read {ref} from its registry: {result.stderr.strip()}")


def local_labels(ref: str) -> dict[str, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", ref],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=True,
    )
    return json.loads(result.stdout) or {}


def push(ref: str) -> None:
    subprocess.run(["docker", "push", ref], check=True, stdin=subprocess.DEVNULL)


def write_record(path: Path, record: dict[str, dict[str, str]]) -> None:
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def publish_images(
    images: ModuleType, registry: str, *, force: bool = False, dry_run: bool = False
) -> list[str]:
    """Push what ``registry`` lacks of one adapter's images and record every
    digest; returns what was (or, dry, would be) pushed.

    A tag the registry already serves is recorded, never pushed again: a
    rebuild need not give the same bytes, and published tasks name the
    first push by digest.
    """
    record = images.published()
    pinned = record.setdefault(registry, {})
    missing = []
    for ref in images.refs(registry):
        key = ref.removeprefix(registry + "/")
        served, known = remote_digest(ref), pinned.get(key)
        if served is None and known is not None:
            raise HubError(f"{ref} is recorded as {known} but {registry} lacks it")
        if served is not None and known is not None and served != known:
            raise HubError(
                f"{ref} was pushed again: {registry} serves {served}, "
                f"{images.RECORD.name} records {known}"
            )
        if served is None:
            missing.append(ref)
        elif known is None:
            print(f"recorded {ref} (already in {registry})")
            pinned[key] = served
    if dry_run:
        return missing
    if missing:
        images.build(registry=registry, force=force)
        for ref in missing:
            if images.LABELS.items() - local_labels(ref).items():
                raise HubError(
                    f"{ref} was built without the OCI labels; rebuild with --force"
                )
        for ref in missing:
            push(ref)
            digest = remote_digest(ref)
            if digest is None:
                raise HubError(f"pushed {ref}, but {registry} doesn't serve it")
            pinned[ref.removeprefix(registry + "/")] = digest
            write_record(images.RECORD, record)  # what's pushed stays recorded
            print(f"pushed {ref}@{digest}")
    write_record(images.RECORD, record)
    return missing


def cmd_images(args: argparse.Namespace) -> int:
    registry = args.registry.rstrip("/")
    for name, images in adapters().items():
        pushed = publish_images(
            images, registry, force=args.force, dry_run=args.dry_run
        )
        verb = "would push" if args.dry_run else "pushed"
        print(f"{name}: {verb} {len(pushed)}: {', '.join(pushed) or 'nothing'}")
    return 0


def recorded_digests(registry: str = HUB_REGISTRY) -> dict[str, str]:
    """Every adapter's published images under ``registry``: ``ref -> digest``."""
    return {
        f"{registry}/{key}": digest
        for images in adapters().values()
        for key, digest in images.published().get(registry, {}).items()
    }


def image_problems(image: str, digests: dict[str, str]) -> list[str]:
    ref, _, digest = image.partition("@")
    if not ref.startswith(HUB_REGISTRY + "/"):
        return [f"{image} is not under {HUB_REGISTRY}"]
    if not DIGEST.match(digest):
        return [f"{image} is not pinned by digest"]
    if digests.get(ref) != digest:
        return [f"{image} is not the digest the adapters recorded for {ref}"]
    return []


def anonymous_pull(image: str) -> bool:
    """Whether a registry serves ``image`` (``host/repo:tag@digest``) to
    anyone, without credentials: the registry v2 API with an anonymous token."""
    ref, _, digest = image.partition("@")
    host, _, path = ref.partition("/")
    repository = path.rsplit(":", 1)[0]
    url = f"https://{host}/v2/{repository}/manifests/{digest}"
    headers: dict[str, str] = {"Accept": ", ".join(MANIFEST_TYPES)}
    try:
        urllib.request.urlopen(
            urllib.request.Request(url, headers=headers, method="HEAD")
        )
        return True
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            return False
        challenge = exc.headers.get("WWW-Authenticate", "")
    params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    if "realm" not in params:
        return False
    query = urllib.parse.urlencode(
        {key: params[key] for key in ("service", "scope") if key in params}
    )
    try:
        with urllib.request.urlopen(f"{params['realm']}?{query}") as response:
            token = json.load(response).get("token", "")
        headers["Authorization"] = f"Bearer {token}"
        urllib.request.urlopen(
            urllib.request.Request(url, headers=headers, method="HEAD")
        )
        return True
    except urllib.error.HTTPError:
        return False


# -------------------------------------------------------------- registry


def load_registry(path: Path = REGISTRY) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise HubError(f"{path.name} must be a list of datasets, as harbor reads it")
    return entries


def write_registry(entries: list[dict[str, Any]], path: Path = REGISTRY) -> None:
    # ASCII: harbor reads it with the platform's default encoding.
    text = json.dumps(entries, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def tasks_at(files: Iterable[str], name: str) -> tuple[list[str], list[str]]:
    """The task folders of dataset ``name`` among the paths of a commit, and
    the paths in it that aren't inside a task folder with a ``task.toml``."""
    prefix = f"datasets/{name}/"
    tasks, other, by_task = set(), [], {}
    for path in files:
        if not path.startswith(prefix):
            continue
        parts = path.removeprefix(prefix).split("/")
        if len(parts) == 1:
            other.append(path)
            continue
        by_task.setdefault(parts[0], []).append(path)
        if parts[1:] == ["task.toml"]:
            tasks.add(parts[0])
    for task, paths in by_task.items():
        if task not in tasks:
            other.extend(paths)
    return sorted(tasks), sorted(other)


def task_images(
    files: dict[str, bytes | None], task: str
) -> tuple[list[str], list[str]]:
    """Every image a task folder names (``docker_image``, compose ``image:``,
    the Dockerfile's ``FROM``), and the compose services it would build
    instead of pulling, which a published task must not."""
    named, builds = [], []
    config = files.get(f"{task}/task.toml")
    if config is not None:
        environment = tomllib.loads(config.decode()).get("environment", {})
        if environment.get("docker_image"):
            named.append(environment["docker_image"])
    compose = files.get(f"{task}/environment/docker-compose.yaml")
    if compose is not None:
        services = (yaml.safe_load(compose) or {}).get("services") or {}
        for name, service in services.items():
            if "image" in service:
                named.append(service["image"])
            if "build" in service:
                builds.append(f"{task}: service {name} is built, not pulled")
    dockerfile = files.get(f"{task}/environment/Dockerfile")
    if dockerfile is not None:
        named += re.findall(r"(?im)^FROM\s+(\S+)", dockerfile.decode())
    return named, builds


TASK_FILES = ("task.toml", "environment/docker-compose.yaml", "environment/Dockerfile")


def dataset_problems(
    repo: Path, sha: str, name: str, digests: dict[str, str]
) -> tuple[list[str], list[str], set[str]]:
    """Dataset ``name``'s task folders at ``sha``, what's wrong with them
    (stray files, images not published by digest), and the images they name."""
    files = git(repo, "ls-tree", "-r", "--name-only", sha, "--", f"datasets/{name}")
    tasks, other = tasks_at(files.splitlines(), name)
    problems = (
        [f"datasets/{name} at {sha[:12]} has no task folders"] if not tasks else []
    )
    problems += [f"{path} is not part of a task folder" for path in other]
    contents = read_files(
        repo,
        sha,
        [f"datasets/{name}/{task}/{file}" for task in tasks for file in TASK_FILES],
    )
    images: set[str] = set()
    for task in tasks:
        named, builds = task_images(contents, f"datasets/{name}/{task}")
        images.update(named)
        problems += builds
    for image in sorted(images):
        problems += image_problems(image, digests)
    return tasks, problems, images


def registry_problems(
    entries: list[dict[str, Any]], git_url: str = DATASETS_GIT_URL
) -> list[str]:
    """What harbor would refuse, or what breaks the hub's pinning rules."""
    from harbor.models.registry import DatasetSpec
    from pydantic import ValidationError

    problems, seen = [], set()
    for at, entry in enumerate(entries):
        try:
            spec = DatasetSpec.model_validate(entry)
        except ValidationError as exc:
            problems.append(f"entry {at}: not a harbor DatasetSpec: {exc}")
            continue
        label = f"{spec.name}@{spec.version}"
        if (spec.name, spec.version) in seen:
            problems.append(f"{label} is listed twice")
        seen.add((spec.name, spec.version))
        if not spec.tasks:
            problems.append(f"{label} has no tasks")
        if spec.metrics:
            problems.append(f"{label} names metrics; they belong in job-config")
        commits = {task.git_commit_id for task in spec.tasks}
        if len(commits) > 1:
            problems.append(f"{label} pins {len(commits)} commits, not one")
        names = [task.name for task in spec.tasks]
        if len(set(names)) != len(names):
            problems.append(f"{label} lists a task twice")
        for task in spec.tasks:
            where = f"{label} task {task.name}"
            if task.git_url != git_url:
                problems.append(f"{where}: git_url is not {git_url}")
            if not SHA.match(task.git_commit_id or ""):
                problems.append(f"{where}: git_commit_id is not a full commit SHA")
            if task.path.as_posix() != f"datasets/{spec.name}/{task.name}":
                problems.append(
                    f"{where}: path is not datasets/{spec.name}/{task.name}"
                )
    return problems


def append_only_problems(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[str]:
    """Published entries never change: a fix is a new version."""
    now = {(entry.get("name"), entry.get("version")): entry for entry in after}
    problems = []
    for entry in before:
        key = (entry.get("name"), entry.get("version"))
        if key not in now:
            problems.append(f"{key[0]}@{key[1]} was removed")
        elif now[key] != entry:
            problems.append(f"{key[0]}@{key[1]} was changed; publish a new version")
    return problems


def registry_at(ref: str) -> list[dict[str, Any]]:
    """``registry.json`` at ``ref`` of this repo; empty where it didn't exist."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:registry.json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        if (
            "exists on disk, but not in" in result.stderr
            or "does not exist" in result.stderr
        ):
            return []
        raise HubError(f"git show {ref}:registry.json: {result.stderr.strip()}")
    return json.loads(result.stdout)


def record_problems() -> list[str]:
    problems = []
    for name, images in adapters().items():
        try:
            record = images.published()
        except json.JSONDecodeError as exc:
            problems.append(f"adapters/{name}/images.json: {exc}")
            continue
        for registry, pinned in record.items():
            for key, digest in pinned.items():
                if not isinstance(digest, str) or not DIGEST.match(digest):
                    problems.append(f"adapters/{name}/images.json: {registry}/{key}")
    return problems


def register(
    repo: Path,
    commit: str,
    version: str,
    datasets: Sequence[tuple[str, str]],
    *,
    registry: Path = REGISTRY,
    git_url: str = DATASETS_GIT_URL,
) -> list[dict[str, Any]]:
    """Append one entry per ``(name, description)``: every task folder of
    ``datasets/<name>/`` at ``commit``, pinned to it. Nothing is written
    unless every dataset passes."""
    from harbor.models.registry import DatasetSpec

    sha = resolve_commit(repo, commit)
    if not sha:
        raise HubError(f"{commit} is not a commit in {repo}")
    problems = release_problems(repo, sha, git_url)
    entries = load_registry(registry)
    listed = {(entry["name"], entry["version"]) for entry in entries}
    digests = recorded_digests()
    added = []
    for name, description in datasets:
        if (name, version) in listed:
            problems.append(f"{name}@{version} is already published")
            continue
        tasks, found, _ = dataset_problems(repo, sha, name, digests)
        problems += found
        entry = {
            "name": name,
            "version": version,
            "description": description,
            "tasks": [
                {
                    "name": task,
                    "git_url": git_url,
                    "git_commit_id": sha,
                    "path": f"datasets/{name}/{task}",
                }
                for task in tasks
            ],
        }
        DatasetSpec.model_validate(entry)
        added.append(entry)
    if problems:
        raise HubError("not registered:\n  " + "\n  ".join(problems))
    write_registry(entries + added, registry)
    return added


def cmd_register(args: argparse.Namespace) -> int:
    added = register(
        args.datasets_repo,
        args.commit,
        args.version,
        args.dataset,
        git_url=args.git_url,
    )
    for entry in added:
        print(
            f"registered {entry['name']}@{entry['version']}: {len(entry['tasks'])} tasks"
        )
    return 0


def check(
    *,
    registry: Path = REGISTRY,
    against: str | None = None,
    repo: Path | None = None,
    anonymous: bool = False,
    git_url: str = DATASETS_GIT_URL,
) -> list[str]:
    entries = load_registry(registry)
    problems = registry_problems(entries, git_url) + record_problems()
    if against is not None:
        problems += append_only_problems(registry_at(against), entries)
    if repo is None:
        return problems
    if problems:  # the release checks assume a well-formed registry
        return problems
    digests, refs = recorded_digests(), remote_refs(repo, git_url)
    commits = {task["git_commit_id"] for entry in entries for task in entry["tasks"]}
    present = set()
    for sha in sorted(commits):
        if resolve_commit(repo, sha) != sha:
            problems.append(f"{sha[:12]} is not in {repo}: fetch first")
            continue
        present.add(sha)
        problems += release_problems(repo, sha, git_url, refs)
    images: set[str] = set()
    for entry in entries:
        sha = entry["tasks"][0]["git_commit_id"]
        if sha not in present:
            continue
        tasks, found, named = dataset_problems(repo, sha, entry["name"], digests)
        problems += found
        images |= named
        listed = sorted(task["name"] for task in entry["tasks"])
        if listed != tasks:
            problems.append(
                f"{entry['name']}@{entry['version']} lists {len(listed)} tasks; "
                f"datasets/{entry['name']} at {sha[:12]} has {len(tasks)}"
            )
    if anonymous:
        problems += [
            f"{image} can't be pulled without credentials"
            for image in sorted(images)
            if "@" in image and not anonymous_pull(image)
        ]
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    problems = check(
        against=args.against,
        repo=args.datasets_repo,
        anonymous=args.anonymous,
        git_url=args.git_url,
    )
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)" if problems else "ok")
    return 1 if problems else 0


# ------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hub.py", description=(__doc__ or "").splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    images = commands.add_parser("images", help="push and record the images")
    images.add_argument("--registry", default=HUB_REGISTRY)
    images.add_argument("--force", action="store_true", help="rebuild before pushing")
    images.add_argument(
        "--dry-run",
        action="store_true",
        help="say what would be pushed; change nothing",
    )
    images.set_defaults(run=cmd_images)

    reg = commands.add_parser("register", help="add dataset versions to registry.json")
    reg.add_argument("--datasets-repo", type=Path, required=True)
    reg.add_argument("--commit", required=True, help="the release commit (or its tag)")
    reg.add_argument("--version", required=True, help="e.g. 0.1.0")
    reg.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "DESCRIPTION"),
        help="a folder under datasets/, and one line on what it is; repeatable",
    )
    reg.add_argument("--git-url", default=DATASETS_GIT_URL, help=argparse.SUPPRESS)
    reg.set_defaults(run=cmd_register)

    chk = commands.add_parser("check", help="validate registry.json and the records")
    chk.add_argument("--against", help="a git ref whose entries must be unchanged")
    chk.add_argument(
        "--datasets-repo", type=Path, help="also run the release checks on this clone"
    )
    chk.add_argument(
        "--anonymous",
        action="store_true",
        help="with --datasets-repo: every image pulls without credentials",
    )
    chk.add_argument("--git-url", default=DATASETS_GIT_URL, help=argparse.SUPPRESS)
    chk.set_defaults(run=cmd_check)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except HubError as exc:
        print(f"hub.py {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
