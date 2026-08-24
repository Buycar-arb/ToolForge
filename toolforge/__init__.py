"""ToolForge — an automated SFT data factory for LLM tool-calling.

Four stages turn raw multi-hop QA into validated training dialogues:

======= ============================================================ =============================
stage   what it does                                                 module
======= ============================================================ =============================
1       grow the tool bank with semantically identical variants      :mod:`toolforge.stages.variants`
2       label each question with a tool and a routing class          :mod:`toolforge.stages.labeling`
3       author a multi-turn dialogue for one of 29 case shapes       :mod:`toolforge.stages.dialogue`
4       validate it (9 rule checks) and score it (LLM judge)         :mod:`toolforge.stages.validation`
======= ============================================================ =============================

Stages 3 and 4 run together — see :mod:`toolforge.stages.pipeline`.

Quick start::

    from toolforge import Pipeline, CaseJob, load_records

    records = load_records("Stage_2/label_data/output.jsonl")
    jobs = [CaseJob("case_C1", target=100,
                    data_output="output/data/case_C1.jsonl",
                    score_output="output/scores/case_C1.jsonl")]
    await Pipeline().run(records, jobs)

Or from the shell: ``toolforge --help``.
"""

from toolforge.config import Settings, settings
from toolforge.llm import MODEL_REGISTRY, LLMClient
from toolforge.stages.cases import CASE_IDS, CASE_SPECS, CaseSpec
from toolforge.stages.dialogue import DialogueGenerator, GeneratedSample, SourceRecord
from toolforge.stages.judge import DialogueJudge, Score
from toolforge.stages.pipeline import CaseJob, CaseProgress, Pipeline, load_records
from toolforge.stages.validation import ValidationOptions, validate

__version__ = "2.0.0"

__all__ = [
    "CASE_IDS",
    "CASE_SPECS",
    "CaseJob",
    "CaseProgress",
    "CaseSpec",
    "DialogueGenerator",
    "DialogueJudge",
    "GeneratedSample",
    "LLMClient",
    "MODEL_REGISTRY",
    "Pipeline",
    "Score",
    "Settings",
    "SourceRecord",
    "ValidationOptions",
    "__version__",
    "load_records",
    "settings",
    "validate",
]
