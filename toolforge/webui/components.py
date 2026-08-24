"""Reusable widgets shared by the Web UI views.

Every user-visible string goes through :func:`toolforge.webui.i18n.t`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr

from toolforge import jsonl
from toolforge.config import settings
from toolforge.llm import MODEL_IDS, model_choices
from toolforge.webui import compat, theme
from toolforge.webui.i18n import t


def model_picker(label: str, value: str | None = None, info: str = "") -> gr.Dropdown:
    """A model dropdown that also accepts any id you type into it."""
    return gr.Dropdown(
        label=label,
        choices=model_choices(),
        value=value or (settings.generation_model if settings.generation_model in MODEL_IDS else MODEL_IDS[0]),
        allow_custom_value=True,
        info=info or t("model.info"),
    )


def sampling_controls(temperature: float = 0.0, max_tokens: int = 8192) -> tuple[gr.Slider, gr.Number]:
    """The temperature / max-tokens pair used by every stage."""
    temp = gr.Slider(
        label=t("temp.label"), minimum=0.0, maximum=2.0, value=temperature, step=0.05,
        info=t("temp.info"),
    )
    tokens = gr.Number(
        label=t("tokens.label"), value=max_tokens, precision=0, minimum=512, maximum=64000,
        info=t("tokens.info"),
    )
    return temp, tokens


def status_and_log(what: str) -> tuple[gr.Markdown, gr.Textbox]:
    """The standard result / live-log pair placed under every action button."""
    with gr.Row():
        with gr.Column(scale=3):
            status = gr.Markdown(t("status.waiting", what=what), elem_classes=["tf-body"])
        with gr.Column(scale=2):
            log = compat.textbox(
                label=t("log.label"), lines=18, max_lines=18, interactive=False,
                elem_classes=["tf-log"], show_copy_button=True,
                placeholder=t("log.placeholder"),
            )
    return status, log


def _describe(path: str | Path) -> str:
    """A localised one-line summary of a JSONL file."""
    file = Path(path)
    if not file.is_file():
        return t("inspect.notfound")
    stat = file.stat()
    return (
        t(
            "inspect.summary",
            name=file.name,
            count=jsonl.count(file),
            size=f"{stat.st_size / 1024:,.1f}",
            modified=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
        + f"\n\n`{file}`"
    )


def _record_at(path: str | Path, index: int) -> str:
    """A localised pretty-printed record, or a localised reason it is unavailable."""
    file = Path(path)
    if not file.is_file():
        return t("inspect.missing", path=file)
    total = jsonl.count(file)
    if total == 0:
        return t("inspect.empty")
    if index < 1 or index > total:
        return t("inspect.oob", total=total)
    return jsonl.record_at(file, index)


def file_inspector(default_path: str = "", *, label: str = "") -> tuple[gr.Textbox, gr.Slider]:
    """A record-by-record browser for any JSONL file the pipeline produces."""
    with gr.Row():
        with gr.Column(scale=3):
            path = gr.Textbox(
                label=label or t("inspect.path"), value=default_path,
                placeholder="/path/to/file.jsonl", info=t("inspect.path.info"),
            )
        with gr.Column(scale=1, min_width=140):
            reload_button = gr.Button(t("inspect.load"), variant="secondary")

    summary = gr.Markdown(t("inspect.none"), elem_classes=["tf-body"])
    # Gradio requires minimum < maximum, so the ceiling never drops below 2.
    index = gr.Slider(label=t("inspect.record"), minimum=1, maximum=2, value=1, step=1)
    content = compat.code(label=t("inspect.content"), language="json", lines=24)

    def load(file_path: str):
        if not file_path or not Path(file_path).is_file():
            return t("inspect.notfound"), gr.update(maximum=2, value=1), t("inspect.nothing")
        total = jsonl.count(file_path)
        if total == 0:
            return _describe(file_path), gr.update(maximum=2, value=1), t("inspect.empty")
        return _describe(file_path), gr.update(maximum=max(2, total), value=1), _record_at(file_path, 1)

    reload_button.click(load, inputs=[path], outputs=[summary, index, content])
    path.submit(load, inputs=[path], outputs=[summary, index, content])
    index.change(
        lambda file_path, position: _record_at(file_path, int(position)),
        inputs=[path, index], outputs=[content],
    )
    return path, index


def config_chips() -> str:
    """Status chips summarising whether the app is ready to make API calls."""
    from toolforge.llm import resolve_provider
    from toolforge.toolbank import domain_names

    items: list[tuple[str, str]] = []
    for key, model in (("chip.gen", settings.generation_model), ("chip.judge", settings.judge_model)):
        provider, model_id = resolve_provider(model)
        ready = bool(settings.keys_for(provider))
        items.append((f"{t(key)}: {model_id}", "ok" if ready else "bad"))

    keys = len(set(settings.openai_api_keys) | set(settings.anthropic_api_keys))
    items.append((t("chip.keys", count=keys), "ok" if keys else "bad"))

    tools = len(domain_names(settings.tool_bank_dir))
    items.append((t("chip.libraries", count=tools), "ok" if tools else "bad"))
    return theme.chips(items)


def stage_rail(active: int) -> str:
    """The four-stage rail shown at the top of each tab."""
    cards = []
    for index in (1, 2, 3, 4):
        highlight = ' style="border-color:var(--tf-brand)"' if index == active else ""
        cards.append(
            f'<div class="tf-step"{highlight}>'
            f'<div class="n">{t(f"rail.{index}.stage")}</div>'
            f'<div class="t">{t(f"rail.{index}.title")}</div>'
            f'<div class="d">{t(f"rail.{index}.desc")}</div></div>'
        )
    return f'<div class="tf-rail">{"".join(cards)}</div>'
