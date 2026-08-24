"""The interface must be complete in every language it claims to support."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("gradio", reason="the webui extra is not installed")

from toolforge.webui import i18n  # noqa: E402

WEBUI = Path(__file__).resolve().parent.parent / "toolforge" / "webui"


def test_every_language_has_every_key() -> None:
    reference = set(i18n.ZH)
    for language, table in i18n.STRINGS.items():
        missing = reference - set(table)
        extra = set(table) - reference
        assert not missing, f"{language} is missing: {sorted(missing)}"
        assert not extra, f"{language} has keys Chinese lacks: {sorted(extra)}"


def test_every_key_used_in_the_views_exists() -> None:
    """A typo in a `t("…")` call must fail the build, not ship a raw key."""
    used: set[str] = set()
    for source in WEBUI.glob("*.py"):
        if source.name == "i18n.py":
            continue
        used |= set(re.findall(r'\bt\(\s*"([a-z0-9_.]+)"', source.read_text(encoding="utf-8")))
    # `rail.N.*` keys are built dynamically from an f-string.
    used |= {f"rail.{n}.{part}" for n in (1, 2, 3, 4) for part in ("stage", "title", "desc")}
    unknown = sorted(key for key in used if key not in i18n.ZH)
    assert not unknown, f"used in a view but not defined: {unknown}"


def test_placeholders_match_across_languages() -> None:
    """A `{name}` in one language must exist in the other, or formatting breaks."""
    for key, chinese in i18n.ZH.items():
        english = i18n.EN[key]
        assert set(re.findall(r"\{(\w+)", chinese)) == set(re.findall(r"\{(\w+)", english)), (
            f"placeholder mismatch for {key!r}"
        )


def test_language_selection_falls_back_to_chinese() -> None:
    assert i18n.set_language("en") == "en"
    assert i18n.set_language("zh") == "zh"
    assert i18n.set_language("klingon") == "zh"
    assert i18n.set_language(None) == "zh"


def test_check_labels_cover_all_nine_checks() -> None:
    """Every rule-check label must have a Chinese rendering for display."""
    from toolforge.stages.validation import CHECK_LABELS

    for label in CHECK_LABELS.values():
        assert label in i18n.CHECK_LABELS_ZH, f"no Chinese rendering for: {label}"


@pytest.mark.parametrize(
    ("line", "expected_fragment"),
    [
        ("▶ case_C1: target 3, 6 labelled records, 2 workers", "目标 3 条"),
        ("  ✓ case_C1 1/3", "已保留 1/3"),
        ("  ✗ attempt 4: 5. Tool-RAG consistency check failed", "第 4 次尝试"),
        ("✅ case_C1: 3/3 kept from 4 attempts (75.0%)", "产出率 75.0%"),
        ("read 6 records — labelling 3, deferring 3", "读取 6 条记录"),
    ],
)
def test_pipeline_progress_is_translated(line: str, expected_fragment: str) -> None:
    i18n.set_language("zh")
    assert expected_fragment in i18n.translate_log(line)


def test_english_leaves_progress_lines_alone() -> None:
    i18n.set_language("en")
    line = "  ✓ case_C1 1/3"
    assert i18n.translate_log(line) == line
    i18n.set_language("zh")


def test_rejection_reasons_are_translated_in_reports() -> None:
    i18n.set_language("zh")
    report = "### Run complete\n\n- `12×` 5. Tool-RAG consistency check failed"
    translated = i18n.translate_report(report)
    assert "运行完成" in translated
    assert "tool 消息与检索段落不一致" in translated


def test_the_ui_builds_in_every_language() -> None:
    from toolforge.webui.app import build

    for language in i18n.SUPPORTED:
        assert build(language).blocks
    i18n.set_language("zh")
