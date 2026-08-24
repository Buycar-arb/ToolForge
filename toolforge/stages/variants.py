"""Stage 1 — grow a tool library by paraphrasing a tool definition.

Each domain file in the tool bank holds many *variants* of the same tool: same
capability, different name, wording and optional parameters.  Stage 3 draws a
random variant per sample, so a model trained on this data has to read tool
descriptions rather than memorise tool names.

A candidate is kept only when it is **semantically close** to the ones already
in the file (cosine similarity above a threshold — it must mean the same thing)
but **lexically far** from them (BM25 similarity below a threshold — it must not
be a near-copy).  That double test is what keeps the library diverse without
letting it drift.

The embedding model is optional: without ``sentence-transformers`` installed the
generator still runs and simply skips the similarity gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolforge import jsonl
from toolforge.config import Settings
from toolforge.config import settings as default_settings
from toolforge.llm import LLMClient
from toolforge.prompts.variants import VARIANT_SYSTEM_PROMPT, VARIANT_USER_PROMPT

log = logging.getLogger(__name__)

EventHook = Callable[[str], None]
Tool = dict[str, Any]


# --------------------------------------------------------------------------- #
# Parsing model output
# --------------------------------------------------------------------------- #


def normalise_tool(payload: Any) -> Tool | None:
    """Coerce whatever the model returned into a ``{"name", "description", ...}`` dict.

    Handles the usual shapes: the tool itself, a ``{"tool": {...}}`` wrapper, a
    list containing one, or a ``title``-instead-of-``name`` variation.
    """
    def looks_like_tool(candidate: Any) -> bool:
        return isinstance(candidate, dict) and "name" in candidate and "description" in candidate

    if isinstance(payload, dict):
        if looks_like_tool(payload.get("tool")):
            return payload["tool"]
        if looks_like_tool(payload):
            return payload
        if "title" in payload and "description" in payload:
            tool = dict(payload)
            tool["name"] = tool.get("name") or tool["title"]
            return tool
        return None

    if isinstance(payload, list):
        for item in payload:
            tool = normalise_tool(item)
            if tool:
                return tool
    return None


def parse_tool(content: str) -> Tool | None:
    """Parse a tool definition out of the model's answer, fenced or bare."""
    text = (content or "").strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return normalise_tool(json.loads(text))
    except json.JSONDecodeError as exc:
        log.warning("could not parse a tool from the model answer: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# The similarity gate
# --------------------------------------------------------------------------- #


@dataclass
class SimilarityVerdict:
    """Why a candidate variant was accepted or rejected."""

    accepted: bool
    cosine: float
    bm25: float
    reason: str


class SimilarityGate:
    """Keeps variants that mean the same thing but do not read the same.

    Parameters
    ----------
    model_path:
        A local sentence-transformers model (``bge-m3`` in the paper).  When it
        cannot be loaded the gate becomes a no-op and every candidate is kept.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model = None
        self.model_path = str(model_path) if model_path else None
        if not self.model_path:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_path)
            try:
                import torch

                if torch.cuda.is_available():
                    self.model = self.model.cuda()
            except ImportError:
                pass
            log.info("similarity gate using %s", self.model_path)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.warning("similarity gate disabled (%s); every candidate will be kept", exc)
            self.model = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    @staticmethod
    def _text(tool: Tool) -> str:
        return f"{tool.get('name', '')}: {tool.get('description', '')}"

    def check(
        self,
        candidate: Tool,
        existing: Sequence[Tool],
        cosine_threshold: float = 0.7,
        bm25_threshold: float = 0.6,
    ) -> SimilarityVerdict:
        """Accept ``candidate`` when it is close in meaning and far in wording."""
        if not existing:
            return SimilarityVerdict(True, 0.0, 0.0, "first variant — nothing to compare against")
        if not self.enabled:
            return SimilarityVerdict(True, 0.0, 0.0, "similarity gate disabled")

        import bm25s
        import numpy as np

        new_text = self._text(candidate)
        old_texts = [self._text(tool) for tool in existing]

        vectors = self.model.encode([new_text, *old_texts], convert_to_numpy=True)
        new_vector, old_vectors = vectors[0], vectors[1:]
        norms = np.linalg.norm(new_vector) * np.linalg.norm(old_vectors, axis=1)
        cosine = float(np.mean(np.where(norms > 0, old_vectors @ new_vector / np.where(norms > 0, norms, 1), 0.0)))

        retriever = bm25s.BM25(corpus=old_texts)
        retriever.index(bm25s.tokenize(old_texts, show_progress=False), show_progress=False)
        _docs, scores = retriever.retrieve(
            bm25s.tokenize([new_text], show_progress=False), k=len(old_texts), show_progress=False
        )
        bm25_score = float(np.mean(scores[0]))

        problems = []
        if cosine <= cosine_threshold:
            problems.append(f"meaning drifted too far (cosine {cosine:.3f} ≤ {cosine_threshold})")
        if bm25_score >= bm25_threshold:
            problems.append(f"wording too close to an existing variant (BM25 {bm25_score:.3f} ≥ {bm25_threshold})")

        if problems:
            return SimilarityVerdict(False, cosine, bm25_score, "; ".join(problems))
        return SimilarityVerdict(True, cosine, bm25_score, "accepted")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def format_existing(tools: Sequence[Tool]) -> str:
    """The "already generated" list shown to the model so it does not repeat itself."""
    if not tools:
        return "No variants available"
    return "\n".join(
        f"{index}. {tool.get('name', '')} - {tool.get('description', '')[:80]}"
        for index, tool in enumerate(tools, 1)
    )


class VariantGenerator:
    """Generates tool variants until a target count is reached."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        gate: SimilarityGate | None = None,
        config: Settings | None = None,
    ) -> None:
        self.config = config or default_settings
        self.client = client or LLMClient(config=self.config)
        self.gate = gate if gate is not None else SimilarityGate(_embedding_model_path())

    async def propose(self, original: Tool, existing: Sequence[Tool]) -> Tool | None:
        """Ask for one new variant."""
        prompt = VARIANT_USER_PROMPT.format(
            tool=json.dumps(original, ensure_ascii=False), variants=format_existing(existing)
        )
        content = await self.client.complete(
            [{"role": "user", "content": prompt}], system=VARIANT_SYSTEM_PROMPT
        )
        return parse_tool(content) if content else None

    async def run(
        self,
        original: Tool,
        output_file: Path | str,
        *,
        target: int = 20,
        cosine_threshold: float = 0.7,
        bm25_threshold: float = 0.6,
        max_attempts: int | None = None,
        delay: float = 1.0,
        on_event: EventHook | None = None,
    ) -> list[Tool]:
        """Top ``output_file`` up to ``target`` variants and return them all.

        The file is appended to, so re-running continues where the last run
        stopped.  ``max_attempts`` defaults to ``target * 5`` and stops runaway
        loops when the similarity gate rejects everything.
        """
        emit = on_event or print
        output = Path(output_file)
        jsonl.touch(output)

        existing = [tool for tool in (normalise_tool(row) for row in jsonl.read(output)) if tool]
        emit(f"{len(existing)} variant(s) already in {output.name}; target is {target}")
        if not self.gate.enabled:
            emit("⚠️  no embedding model — the similarity gate is off, every candidate will be kept")

        budget = max_attempts if max_attempts is not None else max(target * 5, 20)
        attempt = 0

        while len(existing) < target and attempt < budget:
            attempt += 1
            candidate = await self.propose(original, existing)
            if candidate is None:
                emit(f"  ✗ attempt {attempt}: the model returned no usable tool")
                continue

            verdict = self.gate.check(candidate, existing, cosine_threshold, bm25_threshold)
            if not verdict.accepted:
                emit(f"  ✗ attempt {attempt}: {verdict.reason}")
            else:
                existing.append(candidate)
                jsonl.append(output, candidate)
                emit(
                    f"  ✓ {len(existing)}/{target} — {candidate.get('name')} "
                    f"(cosine {verdict.cosine:.3f}, BM25 {verdict.bm25:.3f})"
                )
            if delay:
                await asyncio.sleep(delay)

        if len(existing) < target:
            emit(f"⚠️  stopped after {attempt} attempts with {len(existing)}/{target} variants — "
                 "try loosening the thresholds")
        else:
            emit(f"✅ {output} now holds {len(existing)} variants")
        return existing


def _embedding_model_path() -> str | None:
    import os

    path = os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH")
    return path or None
