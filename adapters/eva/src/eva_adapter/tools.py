"""The agent's tools during an EVA call: run by EVA's executor in the sidecar.

EVA's Realtime server runs each tool call inside its receive loop
(``openai_realtime_server.py:600-640``): ``execute_tool`` on the scenario
DB, then the result as the ``function_call_output``. Here the transport's
``tool_flow="serial"`` reproduces the loop, and this executor is the
``execute_tool``: ``POST /tool`` to the sidecar, whose ``execute_tool`` is
EVA's own on the attempt's scenario DB.

The result goes back to the model as EVA's result object; the transport
serialises it as EVA does (``json.dumps(result, ensure_ascii=False)``, with
``tool_output_ascii=False``). A ``/tool`` that fails raises: the router then
answers the model with an error, and the attempt's ``/record`` is refused,
because the audit log no longer matches what EVA's executor ran. That is an
infrastructure failure, not an EVA result, and the trial fails loudly.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolSpec

from eva_adapter.client import Sidecar


def tool_specs(tools: Sequence[Mapping[str, Any]] | None) -> list[ToolSpec]:
    """EVA's Realtime tools (``{type, name, description, parameters}``) as specs.

    Taken as given: the transport renders each back into the same
    ``session.tools`` entry, and the driver's session check
    (:func:`eva_adapter.trial.check_session`) fails the trial if any differs.
    """
    return [
        ToolSpec(
            name=str(tool["name"]),
            description=str(tool.get("description") or ""),
            input_schema=dict(tool.get("parameters") or {}),
        )
        for tool in tools or ()
    ]


@dataclass(slots=True)
class SidecarToolExecutor:
    """``ToolExecutor`` over the sidecar's ``POST /tool``.

    Attributes:
        log: Every call in order: ``call_id``, ``name``, ``arguments``,
            ``started`` / ``ended`` (epoch seconds), and ``result`` or
            ``error``.
    """

    sidecar: Sidecar
    tools: list[ToolSpec]
    log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return self.tools

    async def execute(self, invocation: ToolInvocation) -> Mapping[str, Any]:
        entry: dict[str, Any] = {
            "call_id": invocation.tool_use_id,
            "name": invocation.name,
            "arguments": dict(invocation.arguments),
            "started": time.time(),
        }
        self.log.append(entry)
        try:
            reply = await self.sidecar.tool(invocation.name, invocation.arguments)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["ended"] = time.time()
        entry["result"] = reply.result
        return reply.result
