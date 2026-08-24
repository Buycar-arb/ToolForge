"""Glue between the Gradio callbacks and the async pipeline.

Two things every view needs:

* :func:`stream` — run an async job in a background thread and yield its log
  lines as they arrive, so the UI updates live instead of freezing until the end.
* :class:`LogBuffer` — a bounded, timestamped log the job writes into.

Log lines produced by the pipeline are rendered in the interface language on
their way to the screen; what the pipeline writes to disk stays untouched.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import traceback
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime
from typing import Any

from toolforge.webui.i18n import t, translate_log

#: Sentinel pushed onto the queue when the job finishes.
_DONE = object()


class LogBuffer:
    """A bounded log with timestamps, rendered straight into a textbox."""

    def __init__(self, limit: int = 400) -> None:
        self._lines: deque[str] = deque(maxlen=limit)

    def __call__(self, message: str) -> None:
        self._lines.append(f"{datetime.now():%H:%M:%S}  {translate_log(message)}")

    def clear(self) -> None:
        self._lines.clear()

    def text(self) -> str:
        return "\n".join(self._lines)


def stream(
    job: Callable[[Callable[[str], None]], Awaitable[Any]],
    *,
    log: LogBuffer | None = None,
    render: Callable[[Any], str] | None = None,
    poll: float = 0.25,
) -> Iterator[tuple[str, str]]:
    """Run ``job`` in the background, yielding ``(status, log)`` as it goes.

    ``job`` receives an ``emit(message)`` callback.  Whatever it returns is
    passed through ``render`` to produce the final status markdown.
    """
    log = log or LogBuffer()
    log.clear()
    updates: queue.Queue[Any] = queue.Queue()
    outcome: dict[str, Any] = {}

    def emit(message: str) -> None:
        log(message)
        updates.put(message)

    def run() -> None:
        try:
            outcome["value"] = asyncio.run(job(emit))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not swallowed
            outcome["error"] = exc
            outcome["traceback"] = traceback.format_exc()
        finally:
            updates.put(_DONE)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    working = t("run.working")
    yield working, log.text()
    while True:
        try:
            item = updates.get(timeout=poll)
        except queue.Empty:
            yield working, log.text()
            continue
        if item is _DONE:
            break
        yield working, log.text()

    worker.join(timeout=5)

    if "error" in outcome:
        error = outcome["error"]
        log(f"❌ {type(error).__name__}: {error}")
        yield (
            t("run.failed", kind=type(error).__name__, error=error,
              traceback=outcome.get("traceback", "")),
            log.text(),
        )
        return

    result = outcome.get("value")
    yield (render(result) if render else t("run.done")), log.text()


def guard(message: str) -> tuple[str, str]:
    """A validation failure shown in the status pane without running anything."""
    return t("run.guard", message=message), ""
