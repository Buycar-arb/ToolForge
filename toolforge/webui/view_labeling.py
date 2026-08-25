"""Stage 2 tab — label multi-hop questions with a tool and a routing class."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from toolforge.config import ROOT_DIR, settings
from toolforge.llm import LLMClient
from toolforge.stages.labeling import LabelStats, ToolLabeler, run_labeling
from toolforge.webui import theme
from toolforge.webui.components import (
    file_inspector,
    model_picker,
    sampling_controls,
    stage_rail,
    status_and_log,
)
from toolforge.webui.i18n import t
from toolforge.webui.runtime import LogBuffer, guard, stream

DEFAULT_INPUT = str(ROOT_DIR / "data" / "source_qa" / "HotpotQA" / "bridge_hp.jsonl")
DEFAULT_OUTPUT = str(settings.output_dir / "labelled" / "output.jsonl")


def _render(stats: LabelStats) -> str:
    attempted = stats.labelled + stats.failed
    rate = (stats.labelled / attempted * 100) if attempted else 0.0
    return t(
        "label.result",
        total=stats.total, labelled=stats.labelled, failed=stats.failed,
        deferred=stats.deferred, rate=f"{rate:.1f}",
    )


def build() -> None:
    gr.HTML(stage_rail(2))
    gr.HTML(theme.note(t("label.note")))

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                input_file = gr.Textbox(
                    label=t("label.input"), value=DEFAULT_INPUT, info=t("label.input.info")
                )
                output_file = gr.Textbox(label=t("label.output"), value=DEFAULT_OUTPUT)
                residue_file = gr.Textbox(
                    label=t("label.residue"),
                    value=str(settings.output_dir / "labelled" / "residue.jsonl"),
                    info=t("label.residue.info"),
                )
            preview = gr.Markdown(t("label.preview"), elem_classes=["tf-body"])

        with gr.Column(scale=2):
            with gr.Group():
                model = model_picker(t("label.model"))
                temperature, max_tokens = sampling_controls(temperature=0.0)
                limit = gr.Number(
                    label=t("label.limit"), value=200, precision=0, minimum=1,
                    info=t("label.limit.info"),
                )
                concurrency = gr.Slider(
                    label=t("label.concurrency"), minimum=1, maximum=32,
                    value=settings.concurrency, step=1, info=t("label.concurrency.info"),
                )
                single_call = gr.Checkbox(
                    label=t("label.single"), value=False, info=t("label.single.info")
                )

    run_button = gr.Button(t("label.run"), variant="primary", size="lg")
    status, log = status_and_log(t("label.status"))

    with gr.Accordion(t("label.inspect"), open=False):
        file_inspector(DEFAULT_OUTPUT, label=t("label.inspect.label"))

    def describe(path: str) -> str:
        from toolforge.webui.components import _describe

        return _describe(path) if path else t("label.preview")

    def launch(inp, out, residue, model_id, temp, tokens, count, workers, force_single):
        # A callback invocation belongs to one browser request/session.  Do not
        # share its mutable log deque with concurrent users.
        buffer = LogBuffer()
        if not inp or not Path(inp).is_file():
            yield guard(t("label.no_input", path=inp))
            return
        if not out:
            yield guard(t("label.no_output"))
            return
        try:
            client = LLMClient(model_id, temperature=float(temp), max_tokens=int(tokens))
        except Exception as exc:  # noqa: BLE001 - missing keys land here
            yield guard(str(exc))
            return

        labeler = ToolLabeler(client, force_single_call=bool(force_single))

        async def job(emit):
            return await run_labeling(
                inp, out,
                residue_file=residue or None,
                max_records=int(count),
                concurrency=int(workers),
                labeler=labeler,
                on_event=emit,
            )

        yield from stream(job, log=buffer, render=_render)

    input_file.change(describe, inputs=[input_file], outputs=[preview])
    run_button.click(
        launch,
        inputs=[input_file, output_file, residue_file, model, temperature, max_tokens,
                limit, concurrency, single_call],
        outputs=[status, log],
    )
