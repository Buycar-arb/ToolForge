"""Tool bank tab — inspect the bank, edit ``TOOL_LIST``, and run stage 1."""

from __future__ import annotations

import json

import gradio as gr

from toolforge.config import settings
from toolforge.llm import LLMClient
from toolforge.stages.variants import VariantGenerator
from toolforge.toolbank import domain_names
from toolforge.webui import compat, theme
from toolforge.webui.components import model_picker, sampling_controls, stage_rail, status_and_log
from toolforge.webui.i18n import t
from toolforge.webui.runtime import LogBuffer, guard, stream
from toolforge.webui.toollist import bank_report, read_tool_list, write_tool_list

EXAMPLE_TOOL = {
    "name": "person_information_search",
    "description": "Search tool for people. Retrieves identity details, life events, "
                   "family relations, education and career history.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up"},
            "person_name": {"type": "string", "description": "Name or alias of the person"},
        },
        "required": ["query"],
    },
}


def build() -> None:
    gr.HTML(stage_rail(1))

    with gr.Tab(t("bank.tab.overview")):
        gr.HTML(theme.note(t("bank.note")))
        report = gr.Markdown(bank_report(), elem_classes=["tf-body"])
        gr.Button(t("bank.rescan")).click(bank_report, outputs=[report])

    with gr.Tab(t("bank.tab.toollist")):
        gr.HTML(theme.note(t("toollist.note")))
        with gr.Row():
            with gr.Column():
                available = gr.CheckboxGroup(
                    label=t("toollist.label"), choices=[], value=[], info=t("toollist.info")
                )
            with gr.Column(scale=0, min_width=190):
                refresh_button = gr.Button(t("toollist.reload"), variant="secondary")
                select_all = gr.Button(t("toollist.selectall"))
                clear_all = gr.Button(t("toollist.clear"))
                save_button = gr.Button(t("toollist.save"), variant="primary")
        save_status = gr.Markdown(t("toollist.unsaved"), elem_classes=["tf-body"])

        def load_choices():
            return gr.update(choices=sorted(domain_names(settings.tool_bank_dir)), value=read_tool_list())

        def save(selection):
            ok, message = write_tool_list(list(selection))
            return f"{'### ✅ ' if ok else '### ⚠️ '}{message}"

        refresh_button.click(load_choices, outputs=[available])
        select_all.click(
            lambda: gr.update(value=sorted(domain_names(settings.tool_bank_dir))), outputs=[available]
        )
        clear_all.click(lambda: gr.update(value=[]), outputs=[available])
        save_button.click(save, inputs=[available], outputs=[save_status])

    with gr.Tab(t("bank.tab.variants")):
        gr.HTML(theme.note(t("variants.note")))
        with gr.Row():
            with gr.Column(scale=3):
                tool_json = compat.code(
                    label=t("variants.tool"), language="json",
                    value=json.dumps(EXAMPLE_TOOL, indent=2, ensure_ascii=False), lines=18,
                )
                output_file = gr.Textbox(
                    label=t("variants.output"),
                    value=str(settings.tool_bank_dir / "person_information_search.jsonl"),
                    info=t("variants.output.info"),
                )
            with gr.Column(scale=2):
                with gr.Group():
                    model = model_picker(t("variants.model"))
                    temperature, max_tokens = sampling_controls(temperature=1.0)
                    target = gr.Number(label=t("variants.target"), value=20, precision=0, minimum=1)
                with gr.Group():
                    cosine = gr.Slider(
                        label=t("variants.cosine"), minimum=0.0, maximum=1.0, value=0.7, step=0.01,
                        info=t("variants.cosine.info"),
                    )
                    bm25 = gr.Slider(
                        label=t("variants.bm25"), minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                        info=t("variants.bm25.info"),
                    )

        run_button = gr.Button(t("variants.run"), variant="primary", size="lg")
        status, log = status_and_log(t("variants.status"))
        def launch(raw_tool, out, model_id, temp, tokens, count, cos, bm):
            # Keep streamed logs private to this callback invocation.
            buffer = LogBuffer()
            try:
                original = json.loads(raw_tool)
            except json.JSONDecodeError as exc:
                yield guard(t("variants.bad_json", error=exc))
                return
            if "name" not in original or "description" not in original:
                yield guard(t("variants.need_fields"))
                return
            if not out:
                yield guard(t("variants.need_output"))
                return
            try:
                client = LLMClient(model_id, temperature=float(temp), max_tokens=int(tokens))
            except Exception as exc:  # noqa: BLE001
                yield guard(str(exc))
                return

            generator = VariantGenerator(client)

            async def job(emit):
                variants = await generator.run(
                    original, out,
                    target=int(count), cosine_threshold=float(cos), bm25_threshold=float(bm),
                    on_event=emit,
                )
                return len(variants), int(count), out

            def render(result):
                produced, wanted, path = result
                text = t(
                    "variants.result",
                    mark="✅" if produced >= wanted else "⚠️",
                    produced=produced, wanted=wanted, path=path,
                )
                return text if produced >= wanted else f"{text}\n\n{t('variants.shortfall')}"

            yield from stream(job, log=buffer, render=render)

        run_button.click(
            launch,
            inputs=[tool_json, output_file, model, temperature, max_tokens, target, cosine, bm25],
            outputs=[status, log],
        )
