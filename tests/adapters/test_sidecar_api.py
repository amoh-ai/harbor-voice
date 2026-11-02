"""Each sidecar's API version moves with its API.

A driver is installed from the running checkout at every trial, but a task's
sidecar is the image it was converted with, so the two check each other's
API version before a call (``SIDECAR_API`` in the client, ``API_VERSION`` in
the sidecar). That check is only as good as the version: here, a fingerprint
of each sidecar's routes and request models is pinned with the version, so a
change to either fails until the version is bumped (and both sides agree).

The fingerprint sees routes, their arguments and the request models' fields.
It does not see the shape of what a route returns: changing that is also an
API change and needs a bump by hand.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVERS = {
    "tau-voice": ROOT / "adapters" / "tau-voice" / "sidecar" / "server.py",
    "eva": ROOT / "adapters" / "eva" / "sidecar" / "server.py",
}
PINNED = {
    "tau-voice": (1, "1ee7cbc874f95be4"),
    "eva": (1, "ac0b2a7163ede00b"),
}
"""``(API_VERSION, fingerprint)`` per sidecar. When this fails: if the API
changed, bump ``API_VERSION`` in the sidecar and ``SIDECAR_API`` in its client,
then pin the new pair; if it didn't (a rename the wire never sees), pin the
new fingerprint alone."""


def _api(tree: ast.Module) -> dict[str, list]:
    routes, models = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "app"
                ):
                    args = [
                        [
                            arg.arg,
                            ast.unparse(arg.annotation) if arg.annotation else None,
                        ]
                        for arg in node.args.args
                    ]
                    path = ast.literal_eval(decorator.args[0])
                    routes.append([decorator.func.attr, path, args])
        if isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases
        ):
            fields = [
                [
                    ast.unparse(item.target),
                    ast.unparse(item.annotation),
                    ast.unparse(item.value) if item.value else None,
                ]
                for item in node.body
                if isinstance(item, ast.AnnAssign)
            ]
            models.append([node.name, fields])
    return {"routes": sorted(routes), "models": sorted(models)}


def fingerprint(server: Path) -> str:
    tree = ast.parse(server.read_text(encoding="utf-8"))
    text = json.dumps(_api(tree), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def api_version(server: Path) -> int:
    for node in ast.parse(server.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "API_VERSION" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{server} has no API_VERSION")


@pytest.mark.parametrize("adapter", sorted(SERVERS))
def test_the_sidecar_and_its_driver_speak_one_api(adapter: str) -> None:
    from eva_adapter import client as eva
    from tau_voice import client as tau_voice

    client = {"tau-voice": tau_voice, "eva": eva}[adapter]
    assert api_version(SERVERS[adapter]) == client.SIDECAR_API


@pytest.mark.parametrize("adapter", sorted(SERVERS))
def test_the_api_version_moves_with_the_api(adapter: str) -> None:
    server = SERVERS[adapter]
    assert (api_version(server), fingerprint(server)) == PINNED[adapter], (
        f"{adapter}'s sidecar API changed: bump API_VERSION and SIDECAR_API if "
        "the wire changed, then pin the new pair in PINNED"
    )


def test_the_fingerprint_sees_a_new_field(tmp_path: Path) -> None:
    server = tmp_path / "server.py"
    base = (
        "class OpenRequest(BaseModel):\n    domain: str\n\n"
        '@app.post("/open")\ndef open_session(request: OpenRequest) -> dict:\n    ...\n'
    )
    server.write_text(base, encoding="utf-8")
    before = fingerprint(server)
    server.write_text(base.replace("domain: str\n", "domain: str\n    seed: int = 0\n"))
    assert fingerprint(server) != before
    server.write_text("# a comment\n" + base, encoding="utf-8")
    assert fingerprint(server) == before
