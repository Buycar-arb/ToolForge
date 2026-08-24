"""Data tab — read any JSONL the pipeline produced, and re-run the checks on it."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import gradio as gr

from toolforge import jsonl
from toolforge.config import settings
from toolforge.stages.validation import ValidationOptions, validate
from toolforge.webui import theme
from toolforge.webui.components import file_inspector
from toolforge.webui.i18n import CHECK_LABELS_ZH, current, t


def _localise(reason: str) -> str:
    return CHECK_LABELS_ZH.get(reason, reason) if current() == "zh" else reason


def _revalidate(path: str, strict_refs: bool, strict_answer: bool) -> str:
    """Re-run the nine rule checks over an existing generated-data file."""
    if not path or not Path(path).is_file():
        return t("data.revalidate.notfound", path=path)

    options = ValidationOptions(
        strict_reference_check=bool(strict_refs),
        strict_final_answer_format=bool(strict_answer),
    )
    total = passed = 0
    failures: Counter[str] = Counter()
    by_case: Counter[str] = Counter()

    for record in jsonl.read(path):
        total += 1
        if not isinstance(record, list) or not record or "case" not in record[0]:
            failures[t("data.revalidate.notrecord")] += 1
            continue
        case_id = record[0]["case"]
        by_case[case_id] += 1
        outcome = validate(record, case_id, options)
        if outcome.passed:
            passed += 1
        else:
            failures.update(_localise(reason) for reason in outcome.failures)

    if total == 0:
        return t("data.revalidate.empty")

    lines = [
        t("data.revalidate.head", mark="✅" if passed == total else "⚠️", passed=passed, total=total),
        "",
        t("data.revalidate.cases"),
    ]
    lines += [f"| `{case_id}` | {count} |" for case_id, count in sorted(by_case.items())]
    if failures:
        lines += ["", t("data.revalidate.fails")]
        lines += [f"| {count} | {reason} |" for reason, count in failures.most_common()]
    return "\n".join(lines)


def build() -> None:
    gr.HTML(theme.note(t("data.note")))

    with gr.Tab(t("data.tab.browse")):
        file_inspector(
            str(settings.output_dir / "data" / "case_C1.jsonl"), label=t("data.browse.label")
        )

    with gr.Tab(t("data.tab.revalidate")):
        gr.HTML(theme.note(t("data.revalidate.note")))
        path = gr.Textbox(
            label=t("data.revalidate.path"),
            value=str(settings.output_dir / "data" / "case_C1.jsonl"),
        )
        with gr.Row():
            strict_refs = gr.Checkbox(label=t("gen.strict.refs"), value=False)
            strict_answer = gr.Checkbox(label=t("gen.strict.answer"), value=False)
        run_button = gr.Button(t("data.revalidate.run"), variant="primary")
        report = gr.Markdown(t("data.revalidate.idle"), elem_classes=["tf-body"])
        run_button.click(_revalidate, inputs=[path, strict_refs, strict_answer], outputs=[report])
