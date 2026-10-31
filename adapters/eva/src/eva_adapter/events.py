"""``user_simulator_events.jsonl``: EVA's ``UserSimulatorEventLogger``, ported.

The same class, methods, signatures and JSONL as
``eva/user_simulator/event_logger.py`` @039efa0a, so the call loop and the
caller log exactly what EVA's simulator logs and EVA's metrics code reads it
unchanged. Two things in that file matter to the metrics and are kept as
they are:

- Every event carries ``timestamp``, epoch milliseconds truncated with
  ``int()`` at the moment it is logged; EVA's processor sorts the merged
  history of all its logs by it.
- ``audio_start`` / ``audio_end`` put ``event_type``, ``user`` and
  ``audio_timestamp`` (epoch seconds, a float) at the top level, not in
  ``data``. ``audio_timestamp`` is what every timing metric is built from;
  callers pass it back-dated where EVA does.

``log_audio_start`` treats a timestamp of ``0`` as "now" (``timestamp or
time.time()``) while ``log_audio_end`` only treats ``None`` so; that
asymmetry is EVA's and is kept.

The one deliberate difference: :meth:`EventLog.save` writes UTF-8 with
``\\n`` line ends on every platform, which is what EVA's ``open(path, "w")``
gives on the Linux it runs on.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVENTS_FILE = "user_simulator_events.jsonl"
"""The file name EVA's simulator writes (``user_simulator/base.py:81-84``)."""


class EventLog:
    """Logs provider-neutral simulator events for EVA's metrics processing.

    Port of ``UserSimulatorEventLogger`` (``event_logger.py:13-192``).
    """

    def __init__(self, output_path: Path, *, provider: str) -> None:
        self.output_path = output_path
        self.provider = provider
        self._events: list[dict[str, Any]] = []
        self._sequence = 0

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event (``event_logger.py:31-47``)."""
        self._sequence += 1
        event: dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
            "sequence": self._sequence,
            "type": event_type,
            "data": data,
        }
        event["provider"] = self.provider
        self._events.append(event)
        logger.debug("User simulator event: %s", event_type)

    def log_user_speech(self, text: str, is_final: bool = True) -> None:
        """Log user speech transcription."""
        self.log_event("user_speech", {"text": text, "is_final": is_final})

    def log_assistant_speech(self, text: str) -> None:
        """Log assistant speech."""
        self.log_event("assistant_speech", {"text": text})

    def log_audio_sent(self, size_bytes: int) -> None:
        """Log audio data sent to assistant."""
        self.log_event("audio_sent", {"size_bytes": size_bytes})

    def log_audio_received(self, size_bytes: int) -> None:
        """Log audio data received from assistant."""
        self.log_event("audio_received", {"size_bytes": size_bytes})

    def log_connection_state(
        self, state: str, details: dict[str, Any] | None = None
    ) -> None:
        """Log connection state change."""
        self.log_event("connection_state", {"state": state, "details": details or {}})

    def log_error(self, error: str, details: dict[str, Any] | None = None) -> None:
        """Log an error."""
        self.log_event("error", {"error": error, "details": details or {}})

    def log_audio_start(self, role: str, timestamp: float | None = None) -> None:
        """Log when audio starts for ``role`` (``event_logger.py:106-127``).

        Args:
            role: ``"simulated_user"`` or ``"assistant"``.
            timestamp: Epoch seconds the audio started; falsy means now.
        """
        audio_timestamp = timestamp or time.time()
        self._sequence += 1
        event = {
            "timestamp": int(time.time() * 1000),
            "sequence": self._sequence,
            "event_type": "audio_start",
            "user": role,
            "audio_timestamp": audio_timestamp,
            "provider": self.provider,
        }
        self._events.append(event)
        logger.debug("Audio start logged: %s", role)

    def log_audio_end(self, role: str, timestamp: float | None = None) -> None:
        """Log when audio ends for ``role`` (``event_logger.py:129-153``).

        Args:
            role: ``"simulated_user"`` or ``"assistant"``.
            timestamp: Epoch seconds of the *actual* end. End of audio is
                detected after a silence threshold, so callers pass the real
                end; ``None`` means now.
        """
        audio_timestamp = timestamp if timestamp is not None else time.time()
        self._sequence += 1
        event = {
            "timestamp": int(time.time() * 1000),
            "sequence": self._sequence,
            "event_type": "audio_end",
            "user": role,
            "audio_timestamp": audio_timestamp,
            "provider": self.provider,
        }
        self._events.append(event)
        logger.debug("Audio end logged: %s", role)

    def save(self) -> None:
        """Write every event, one JSON object per line (``event_logger.py:155-162``)."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(
                json.dumps(event, ensure_ascii=False) + "\n" for event in self._events
            )
        logger.info(
            "Saved %d user simulator events to %s", len(self._events), self.output_path
        )

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Logged events, optionally filtered by type."""
        if event_type is None:
            return self._events.copy()
        return [
            e
            for e in self._events
            if (e.get("type") or e.get("event_type")) == event_type
        ]

    def get_summary(self) -> dict[str, Any]:
        """A count of logged events by type."""
        event_counts: dict[str, int] = {}
        for event in self._events:
            event_type = event.get("type") or event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return {"total_events": len(self._events), "event_counts": event_counts}

    def clear(self) -> None:
        """Clear all logged events."""
        self._events.clear()
        self._sequence = 0
