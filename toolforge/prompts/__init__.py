"""Every prompt in the pipeline, grouped by the job it does.

============================  =====================================================
module                        used by
============================  =====================================================
:mod:`~toolforge.prompts.tool_selection`  stage 2 — pick the tool and the route
:mod:`~toolforge.prompts.variants`        stage 1 — paraphrase a tool definition
:mod:`~toolforge.prompts.planning`        stage 3 — plan the tool-calling trajectory
:mod:`~toolforge.prompts.dialogue`        stage 3 — author the conversation
:mod:`~toolforge.prompts.cases`           stage 3 — one user prompt per case
:mod:`~toolforge.prompts.flows`           stage 3 — the reasoning flow of each case
:mod:`~toolforge.prompts.agent`           embedded in the training data itself
:mod:`~toolforge.prompts.judge`           stage 4 — LLM quality scoring
============================  =====================================================
"""

from toolforge.prompts import agent, cases, dialogue, flows, judge, planning  # noqa: F401

__all__ = ["agent", "cases", "dialogue", "flows", "judge", "planning"]
