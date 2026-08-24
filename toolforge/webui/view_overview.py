"""Overview tab — what the pipeline does, and whether this install can run it."""

from __future__ import annotations

import gradio as gr

from toolforge import __version__
from toolforge.config import ROOT_DIR, reload_settings, settings
from toolforge.stages.cases import CASE_IDS, CASE_SPECS
from toolforge.webui import compat
from toolforge.webui.components import stage_rail
from toolforge.webui.i18n import t


def _readiness() -> str:
    """A checklist of everything that must be in place before a run."""
    from toolforge.llm import resolve_provider
    from toolforge.toolbank import domain_names

    config = reload_settings()
    rows: list[tuple[str, bool, str]] = []

    uses_anthropic = False
    for key, model in (("overview.check.gen", config.generation_model),
                       ("overview.check.judge", config.judge_model)):
        provider, model_id = resolve_provider(model)
        uses_anthropic = uses_anthropic or provider == "anthropic"
        has_key = bool(config.keys_for(provider))
        variable = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        detail = (
            t("overview.check.model", model=model_id, provider=provider) if has_key
            else t("overview.check.needkey", model=model_id, provider=provider, variable=variable)
        )
        rows.append((t(key), has_key, detail))

    if uses_anthropic:
        try:
            import anthropic  # noqa: F401
            rows.append((t("overview.check.sdk"), True, t("overview.check.installed")))
        except ImportError:
            rows.append((t("overview.check.sdk"), False, t("overview.check.sdk.missing")))

    tools = domain_names(config.tool_bank_dir)
    rows.append((
        t("overview.check.bank"), bool(tools),
        t("overview.check.bank.detail", count=len(tools), path=config.tool_bank_dir),
    ))

    try:
        import bm25s  # noqa: F401
        rows.append((t("overview.check.bm25"), True, t("overview.check.installed")))
    except ImportError:
        rows.append((t("overview.check.bm25"), False, t("overview.check.bm25.missing")))

    blocked = [name for name, ok, _ in rows if not ok]

    # A missing .env only matters when something is actually unset — supplying
    # the variables in the environment directly is a perfectly good setup.
    env_file = ROOT_DIR / ".env"
    if env_file.is_file():
        rows.append((t("overview.check.env"), True, t("overview.check.env.ok", name=env_file.name)))
    elif blocked:
        rows.append((t("overview.check.env"), False, t("overview.check.env.missing")))
        blocked.append(t("overview.check.env"))
    else:
        rows.append((t("overview.check.env"), True, t("overview.check.env.unused")))

    lines = [t("overview.header")]
    lines += [f"| {'✅' if ok else '❌'} | {name} | {detail} |" for name, ok, detail in rows]
    verdict = t("overview.ready") if not blocked else t("overview.notready", blocked=", ".join(blocked))
    return "\n".join(lines) + verdict


def build() -> None:
    gr.HTML(stage_rail(0))

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                t(
                    "overview.body",
                    count=len(CASE_SPECS),
                    a_cases=", ".join(c for c in CASE_IDS if c[5] == "A"),
                    b_cases=", ".join(c for c in CASE_IDS if c[5] == "B"),
                    c_count=sum(1 for c in CASE_IDS if c[5] == "C"),
                    d_count=sum(1 for c in CASE_IDS if c[5] == "D"),
                ),
                elem_classes=["tf-body"],
            )
        with gr.Column(scale=2):
            gr.Markdown(t("overview.readiness"), elem_classes=["tf-body"])
            readiness = gr.Markdown(_readiness(), elem_classes=["tf-body"])
            recheck = gr.Button(t("overview.recheck"), variant="secondary")
            gr.Markdown(t("overview.panel", version=__version__), elem_classes=["tf-body"])

    with gr.Accordion(t("overview.config.accordion"), open=False):
        config_view = compat.code(
            value=settings.describe(), lines=12, label=t("overview.config.label")
        )

    recheck.click(
        lambda: (_readiness(), reload_settings().describe()),
        outputs=[readiness, config_view],
    )
