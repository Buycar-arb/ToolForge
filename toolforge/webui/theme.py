"""Visual identity for the ToolForge Web UI.

A Gradio theme plus a stylesheet, kept apart from the views so the look can be
changed in one place.  Both light and dark are first-class: every colour is a
CSS custom property redefined under ``.dark``.
"""

from __future__ import annotations

import gradio as gr

#: Brand colour — the amber of hot metal, for a tool *forge*.
PRIMARY = gr.themes.colors.Color(
    name="forge",
    c50="#fff8ed", c100="#ffefd4", c200="#ffdba8", c300="#ffc071",
    c400="#ff9e38", c500="#fb8312", c600="#ec6708", c700="#c44c09",
    c800="#9c3c10", c900="#7d3310", c950="#441806",
)


def build_theme() -> gr.themes.Base:
    """The Gradio theme object used by :func:`toolforge.webui.app.build`."""
    return gr.themes.Base(
        primary_hue=PRIMARY,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        block_background_fill="white",
        block_background_fill_dark="*neutral_900",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.06)",
        block_title_text_weight="600",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_primary_text_color="white",
        input_background_fill="white",
        input_background_fill_dark="*neutral_800",
    )


CSS = """
/* ---------------------------------------------------------------- tokens */
:root {
  --tf-ink: #0f172a;
  --tf-muted: #64748b;
  --tf-line: #e2e8f0;
  --tf-surface: #ffffff;
  --tf-raised: #f8fafc;
  --tf-brand: #ec6708;
  --tf-ok: #15803d;
  --tf-warn: #b45309;
  --tf-bad: #b91c1c;
}
.dark {
  --tf-ink: #e2e8f0;
  --tf-muted: #94a3b8;
  --tf-line: #334155;
  --tf-surface: #0f172a;
  --tf-raised: #1e293b;
  --tf-brand: #fb8312;
  --tf-ok: #4ade80;
  --tf-warn: #fbbf24;
  --tf-bad: #f87171;
}

.gradio-container { max-width: 1440px !important; }

/* ---------------------------------------------------------------- masthead */
.tf-masthead {
  display: flex; align-items: center; gap: 18px;
  padding: 22px 26px; margin-bottom: 6px;
  border: 1px solid var(--tf-line); border-radius: 14px;
  background: linear-gradient(115deg, rgba(236,103,8,.10), rgba(236,103,8,0) 55%), var(--tf-surface);
}
.tf-mark {
  flex: 0 0 auto; width: 46px; height: 46px; border-radius: 12px;
  display: grid; place-items: center; font-size: 24px;
  background: linear-gradient(140deg, #fb8312, #c44c09);
  box-shadow: 0 6px 16px rgba(236,103,8,.28);
}
.tf-masthead h1 { margin: 0; font-size: 21px; font-weight: 650; color: var(--tf-ink); letter-spacing: -.01em; }
.tf-masthead p  { margin: 3px 0 0; font-size: 13.5px; color: var(--tf-muted); }
.tf-masthead .tf-spacer { flex: 1 1 auto; }

/* ---------------------------------------------------------------- chips */
.tf-chips { display: flex; flex-wrap: wrap; gap: 7px; }
.tf-chip {
  font-size: 11.5px; font-weight: 550; letter-spacing: .02em;
  padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--tf-line); background: var(--tf-raised); color: var(--tf-muted);
  white-space: nowrap;
}
.tf-chip.ok   { color: var(--tf-ok);   border-color: color-mix(in srgb, var(--tf-ok) 40%, transparent); }
.tf-chip.warn { color: var(--tf-warn); border-color: color-mix(in srgb, var(--tf-warn) 40%, transparent); }
.tf-chip.bad  { color: var(--tf-bad);  border-color: color-mix(in srgb, var(--tf-bad) 40%, transparent); }

/* ---------------------------------------------------------------- panels */
.tf-panel {
  border: 1px solid var(--tf-line); border-radius: 12px;
  padding: 16px 18px; background: var(--tf-surface);
}
.tf-note {
  border-left: 3px solid var(--tf-brand);
  padding: 10px 14px; margin: 6px 0 2px;
  background: var(--tf-raised); border-radius: 0 8px 8px 0;
  font-size: 13px; color: var(--tf-muted); line-height: 1.6;
}
.tf-note strong { color: var(--tf-ink); font-weight: 600; }

/* ---------------------------------------------------------------- stage rail */
.tf-rail { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 14px; }
.tf-step {
  flex: 1 1 160px; min-width: 160px;
  border: 1px solid var(--tf-line); border-radius: 11px;
  padding: 12px 14px; background: var(--tf-surface);
}
.tf-step .n {
  font-size: 10.5px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--tf-brand);
}
.tf-step .t { font-size: 14px; font-weight: 600; color: var(--tf-ink); margin-top: 3px; }
.tf-step .d { font-size: 12.5px; color: var(--tf-muted); margin-top: 4px; line-height: 1.5; }

/* ---------------------------------------------------------------- logs */
.tf-log textarea {
  font-family: var(--font-mono) !important;
  font-size: 12.5px !important; line-height: 1.65 !important;
  background: var(--tf-raised) !important;
}

/* ---------------------------------------------------------------- tables */
.tf-body table { width: 100%; border-collapse: collapse; font-size: 13px; }
.tf-body th {
  text-align: left; font-weight: 600; color: var(--tf-muted);
  border-bottom: 1px solid var(--tf-line); padding: 7px 10px;
  font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em;
}
.tf-body td { border-bottom: 1px solid var(--tf-line); padding: 7px 10px; color: var(--tf-ink); }
.tf-body tr:last-child td { border-bottom: none; }
.tf-body code {
  font-size: 12px; padding: 1px 5px; border-radius: 4px;
  background: var(--tf-raised); border: 1px solid var(--tf-line);
}

/* ---------------------------------------------------------------- tabs */
button.svelte-1ipelgc, .tab-nav button { font-weight: 550 !important; }
footer { display: none !important; }
"""


def masthead(subtitle: str) -> str:
    """The header block shown at the top of the app."""
    return f"""
<div class="tf-masthead">
  <div class="tf-mark">🔨</div>
  <div>
    <h1>ToolForge</h1>
    <p>{subtitle}</p>
  </div>
</div>
"""


def chips(items: list[tuple[str, str]]) -> str:
    """Render status chips: ``[(text, "ok" | "warn" | "bad" | "")]``."""
    rendered = "".join(f'<span class="tf-chip {tone}">{text}</span>' for text, tone in items)
    return f'<div class="tf-chips">{rendered}</div>'


def note(html: str) -> str:
    """An inline explanatory callout."""
    return f'<div class="tf-note">{html}</div>'
