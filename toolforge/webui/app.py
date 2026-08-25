"""The ToolForge Web UI.

Five tabs, one per thing you actually do:

============  =========================================================
tab           what it is for
============  =========================================================
Overview      what the pipeline builds, and whether this install can run it
Tool bank     inspect the 22 tool libraries, edit ``TOOL_LIST``, run stage 1
Label         stage 2 — tag questions with a tool and a routing class
Generate      stages 3 + 4 — author dialogues and keep the ones that validate
Data          browse any output file, or re-run the rule checks over it
============  =========================================================

The interface is Chinese by default; ``toolforge webui --lang en`` (or
``UI_LANG=en``) switches it. All strings live in :mod:`toolforge.webui.i18n`.

Launch it with ``toolforge webui`` (or ``python -m toolforge webui``).
"""

from __future__ import annotations

import gradio as gr

from toolforge import __version__
from toolforge.config import settings
from toolforge.webui import (
    compat,
    theme,
    view_generate,
    view_inspect,
    view_labeling,
    view_overview,
    view_toolbank,
)
from toolforge.webui.components import config_chips
from toolforge.webui.i18n import set_language, t
from toolforge.webui.security import launch_security


def build(language: str | None = None) -> gr.Blocks:
    """Assemble the Gradio app in the requested interface language."""
    set_language(language)
    with compat.blocks(
        theme=theme.build_theme(),
        css=theme.CSS,
        title=f"ToolForge {__version__}",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(theme.masthead(t("app.subtitle")))
        chips = gr.HTML(config_chips())

        with gr.Tabs():
            with gr.Tab(t("tab.overview")):
                view_overview.build()
            with gr.Tab(t("tab.toolbank")):
                view_toolbank.build()
            with gr.Tab(t("tab.label")):
                view_labeling.build()
            with gr.Tab(t("tab.generate")):
                view_generate.build()
            with gr.Tab(t("tab.data")):
                view_inspect.build()

        demo.load(config_chips, outputs=[chips])
    return demo


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    language: str | None = None,
) -> None:
    """Start the server (used by ``toolforge webui``)."""
    auth, show_error = launch_security(host, share)
    print(f"ToolForge {__version__}")
    print(settings.describe())
    print(f"\nOpening http://{host}:{port}\n")
    build(language).launch(
        server_name=host,
        server_port=port,
        share=share,
        auth=auth,
        show_error=show_error,
        **compat.launch_kwargs(theme.build_theme(), theme.CSS),
    )


if __name__ == "__main__":
    launch()
