"""The four pipeline stages.

* :mod:`~toolforge.stages.variants` — stage 1, grow the tool bank
* :mod:`~toolforge.stages.labeling` — stage 2, label questions with tools + routes
* :mod:`~toolforge.stages.cases` — the 29 dialogue cases, declared as data
* :mod:`~toolforge.stages.dialogue` — stage 3, the generation engine
* :mod:`~toolforge.stages.validation` — stage 4, the nine rule checks
* :mod:`~toolforge.stages.judge` — stage 4, the LLM judge
* :mod:`~toolforge.stages.pipeline` — stages 3 + 4 wired together
"""
