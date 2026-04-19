from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class EventRecord:
    event_id: int
    kind: str
    run_id: str | None
    created_at: str

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "run_id": self.run_id,
            "created_at": self.created_at,
        }


class EventBroker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._event_id = 0
        self._last_event = EventRecord(
            event_id=0,
            kind="snapshot",
            run_id=None,
            created_at=utc_now(),
        )

    def publish(self, kind: str, run_id: str | None = None) -> EventRecord:
        with self._condition:
            self._event_id += 1
            self._last_event = EventRecord(
                event_id=self._event_id,
                kind=kind,
                run_id=run_id,
                created_at=utc_now(),
            )
            self._condition.notify_all()
            return self._last_event

    def wait_for_update(self, since: int, timeout: float) -> dict[str, object]:
        with self._condition:
            if since == 0:
                return {
                    "event_id": self._event_id,
                    "changed": True,
                    "event": self._last_event.to_dict(),
                }

            if self._event_id <= since:
                self._condition.wait_for(lambda: self._event_id > since, timeout=timeout)

            changed = self._event_id > since
            return {
                "event_id": self._event_id,
                "changed": changed,
                "event": self._last_event.to_dict() if changed else None,
            }


event_broker = EventBroker()
