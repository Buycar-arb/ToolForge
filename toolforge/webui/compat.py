"""Smooth over the differences between Gradio 5 and Gradio 6.

Gradio 6 dropped a few component keyword arguments and moved ``theme`` / ``css``
from the :class:`gradio.Blocks` constructor to :meth:`gradio.Blocks.launch`.
Keeping that knowledge in one place lets the views be written once and run on
both, so the UI does not force a particular Gradio release on the user.
"""

from __future__ import annotations

import inspect
from typing import Any

import gradio as gr

#: Major version of the installed Gradio.
MAJOR = int(gr.__version__.split(".", 1)[0])

#: True when ``theme`` / ``css`` belong on ``launch()`` rather than ``Blocks()``.
THEME_ON_LAUNCH = MAJOR >= 6


def supported(component: type, name: str) -> bool:
    """Whether ``component.__init__`` accepts the keyword ``name``."""
    try:
        return name in inspect.signature(component.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic components
        return False


def textbox(**kwargs: Any) -> gr.Textbox:
    """A :class:`gradio.Textbox` with unsupported keywords dropped."""
    return gr.Textbox(**{key: value for key, value in kwargs.items() if supported(gr.Textbox, key)})


def code(**kwargs: Any) -> gr.Code:
    """A :class:`gradio.Code` with unsupported keywords dropped."""
    return gr.Code(**{key: value for key, value in kwargs.items() if supported(gr.Code, key)})


def blocks(*, theme: Any, css: str, **kwargs: Any) -> gr.Blocks:
    """Build the top-level :class:`gradio.Blocks`, styled the way this version wants."""
    if THEME_ON_LAUNCH:
        return gr.Blocks(**kwargs)
    return gr.Blocks(theme=theme, css=css, **kwargs)


def launch_kwargs(theme: Any, css: str) -> dict[str, Any]:
    """Extra keyword arguments for ``Blocks.launch`` on this version."""
    return {"theme": theme, "css": css} if THEME_ON_LAUNCH else {}
