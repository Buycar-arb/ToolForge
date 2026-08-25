"""Stages 3 + 4 tab — generate dialogues and keep only the ones that validate."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import gradio as gr

from toolforge.config import settings
from toolforge.llm import LLMClient
from toolforge.stages.cases import CASE_IDS, CASE_SPECS, summary_table
from toolforge.stages.dialogue import DialogueGenerator
from toolforge.stages.judge import DialogueJudge
from toolforge.stages.pipeline import CaseJob, Pipeline, format_report, load_records
from toolforge.stages.validation import ValidationOptions
from toolforge.webui import compat, theme
from toolforge.webui.components import (
    file_inspector,
    model_picker,
    sampling_controls,
    stage_rail,
    status_and_log,
)
from toolforge.webui.i18n import t, translate_report
from toolforge.webui.runtime import LogBuffer, guard, stream

DEFAULT_INPUT = str(settings.output_dir / "labelled" / "output.jsonl")


def _case_label(case_id: str) -> str:
    return f"{case_id}  ·  {CASE_SPECS[case_id].description}"


def build() -> None:
    gr.HTML(stage_rail(3))
    gr.HTML(theme.note(t("gen.note")))

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                input_file = gr.Textbox(
                    label=t("gen.input"), value=DEFAULT_INPUT, info=t("gen.input.info")
                )
                output_dir = gr.Textbox(label=t("gen.outdir"), value=str(settings.output_dir))
                selected_cases = gr.Dropdown(
                    label=t("gen.cases"),
                    choices=[(_case_label(case_id), case_id) for case_id in CASE_IDS],
                    value=["case_C1"], multiselect=True, info=t("gen.cases.info"),
                )
                target = gr.Number(
                    label=t("gen.target"), value=10, precision=0, minimum=1,
                    info=t("gen.target.info"),
                )
            advanced_config = compat.code(label=t("gen.advanced"), language="json", lines=8, value="")
            gr.HTML(theme.note(t("gen.advanced.note")))

        with gr.Column(scale=2):
            with gr.Group():
                model = model_picker(t("gen.model"), info=t("gen.model.info"))
                judge_model = model_picker(
                    t("gen.judge"), value=settings.judge_model, info=t("gen.judge.info")
                )
                temperature, max_tokens = sampling_controls(temperature=0.0)
            with gr.Group():
                concurrency = gr.Slider(
                    label=t("gen.concurrency"), minimum=1, maximum=16,
                    value=min(4, settings.concurrency), step=1, info=t("gen.concurrency.info"),
                )
                delay = gr.Slider(
                    label=t("gen.delay"), minimum=0.0, maximum=10.0, value=1.0, step=0.5,
                    info=t("gen.delay.info"),
                )
                virtual_min = gr.Slider(
                    label=t("gen.vmin"), minimum=0, maximum=20,
                    value=settings.virtual_tool_min, step=1,
                )
                virtual_max = gr.Slider(
                    label=t("gen.vmax"), minimum=1, maximum=25,
                    value=settings.virtual_tool_max, step=1, info=t("gen.vmax.info"),
                )
            with gr.Accordion(t("gen.strict"), open=False):
                strict_references = gr.Checkbox(
                    label=t("gen.strict.refs"), value=False, info=t("gen.strict.refs.info")
                )
                strict_answer = gr.Checkbox(
                    label=t("gen.strict.answer"), value=False, info=t("gen.strict.answer.info")
                )

    run_button = gr.Button(t("gen.run"), variant="primary", size="lg")
    status, log = status_and_log(t("gen.status"))

    with gr.Accordion(t("gen.inspect"), open=False):
        file_inspector(str(settings.output_dir / "data" / "case_C1.jsonl"), label=t("gen.inspect.label"))

    with gr.Accordion(t("gen.cases_accordion"), open=False):
        gr.Markdown(summary_table(), elem_classes=["tf-body"])

    def launch(inp, out_dir, case_ids, per_case, raw_config, gen_model, jdg_model,
               temp, tokens, workers, pause, vmin, vmax, strict_refs, strict_ans):
        # Keep streamed logs private to this callback invocation.
        buffer = LogBuffer()
        if not inp or not Path(inp).is_file():
            yield guard(t("gen.no_input", path=inp))
            return

        base = Path(out_dir or settings.output_dir)
        jobs: list[CaseJob] = []
        if raw_config and raw_config.strip():
            try:
                blocks = json.loads(raw_config)
            except json.JSONDecodeError as exc:
                yield guard(t("gen.bad_json", error=exc))
                return
            unknown = [case_id for case_id in blocks if case_id not in CASE_SPECS]
            if unknown:
                yield guard(t("gen.unknown_case", names=", ".join(unknown)))
                return
            jobs = [CaseJob.from_config(case_id, block, base) for case_id, block in blocks.items()]
        else:
            if not case_ids:
                yield guard(t("gen.pick_case"))
                return
            jobs = [
                CaseJob(
                    case_id=case_id,
                    target=int(per_case),
                    data_output=base / "data" / f"{case_id}.jsonl",
                    score_output=base / "scores" / f"{case_id}.jsonl",
                )
                for case_id in case_ids
            ]

        if int(vmin) > int(vmax):
            yield guard(t("gen.vrange"))
            return

        # The distractor sliders live on the settings object the tool bank reads.
        run_config = replace(settings, virtual_tool_min=int(vmin), virtual_tool_max=int(vmax))

        try:
            generator = DialogueGenerator(
                LLMClient(gen_model, temperature=float(temp), max_tokens=int(tokens), config=run_config),
                config=run_config,
            )
            judge = DialogueJudge(
                LLMClient(jdg_model, temperature=0.0, max_tokens=int(tokens), config=run_config),
                config=run_config,
            )
        except Exception as exc:  # noqa: BLE001 - missing keys land here
            yield guard(str(exc))
            return

        pipeline = Pipeline(
            config=run_config,
            generator=generator,
            judge=judge,
            validation_options=ValidationOptions(
                strict_reference_check=bool(strict_refs),
                strict_final_answer_format=bool(strict_ans),
            ),
        )

        async def job(emit):
            records = load_records(inp)
            if not records:
                raise ValueError(t("gen.no_records", path=inp))
            emit(t("gen.loaded", count=len(records)))
            return await pipeline.run(
                records, jobs, on_event=emit, delay=float(pause), concurrency=int(workers)
            )

        yield from stream(job, log=buffer, render=lambda r: translate_report(format_report(r)))

    run_button.click(
        launch,
        inputs=[input_file, output_dir, selected_cases, target, advanced_config,
                model, judge_model, temperature, max_tokens, concurrency, delay,
                virtual_min, virtual_max, strict_references, strict_answer],
        outputs=[status, log],
    )
