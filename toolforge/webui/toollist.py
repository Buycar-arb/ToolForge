"""Read and rewrite the ``TOOL_LIST`` block that stage 2 shows the model.

``TOOL_LIST`` lives in :mod:`toolforge.prompts.tool_selection` as one
``name：description`` line per tool.  It must stay in step with the tool bank on
disk: a tool stage 2 can pick but the bank cannot supply will break stage 3.
This module is what the Web UI's *Tool bank* tab edits.
"""

from __future__ import annotations

import re
from pathlib import Path

from toolforge.config import settings
from toolforge.toolbank import describe_bank
from toolforge.webui.i18n import t

#: The module whose ``TOOL_LIST`` literal is rewritten in place.
TOOL_LIST_FILE = Path(__file__).resolve().parent.parent / "prompts" / "tool_selection.py"

_BLOCK = re.compile(r'TOOL_LIST\s*=\s*"""(.*?)"""', re.DOTALL)

#: Stage 2's prompts use a full-width colon between name and description.
SEPARATOR = "："


def read_tool_list(path: Path | None = None) -> list[str]:
    """The tool names currently offered to stage 2, in file order."""
    source = (path or TOOL_LIST_FILE).read_text(encoding="utf-8")
    match = _BLOCK.search(source)
    if not match:
        return []
    return [
        line.split(SEPARATOR, 1)[0].strip()
        for line in match.group(1).strip().splitlines()
        if SEPARATOR in line
    ]


def write_tool_list(names: list[str], path: Path | None = None) -> tuple[bool, str]:
    """Rewrite the ``TOOL_LIST`` literal with ``names``, keeping bank descriptions."""
    target = path or TOOL_LIST_FILE
    if not names:
        return False, t("toollist.refuse_empty")

    descriptions = describe_bank(settings.tool_bank_dir)
    missing = [name for name in names if name not in descriptions]
    if missing:
        return False, t("toollist.not_in_bank", names=", ".join(missing))

    body = "\n".join(f"{name}{SEPARATOR}{descriptions[name]}" for name in names)
    source = target.read_text(encoding="utf-8")
    if not _BLOCK.search(source):
        return False, t("toollist.no_definition", file=target)

    target.write_text(_BLOCK.sub(f'TOOL_LIST = """\n{body}\n"""', source, count=1), encoding="utf-8")
    return True, t("toollist.saved", count=len(names), file=target.name)


def bank_report() -> str:
    """A markdown table of the bank: tool, variant count, description."""
    from toolforge.toolbank import domain_files, load_variants

    files = domain_files(settings.tool_bank_dir)
    if not files:
        return t("bank.report.none", path=settings.tool_bank_dir)

    active = set(read_tool_list())
    rows = [
        t("bank.report.head", path=settings.tool_bank_dir, total=len(files), active=len(active)),
        "",
        t("bank.report.cols"),
    ]
    for path in files:
        variants = load_variants(path)
        description = variants[0].get("description", "") if variants else t("bank.empty_file")
        mark = "✅" if path.stem in active else "—"
        rows.append(f"| {mark} | `{path.stem}` | {len(variants)} | {description[:110]} |")
    return "\n".join(rows)
