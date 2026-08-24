"""BM25 retrieval used to build realistic (noisy) tool responses.

Stage 3 needs a tool call to return something that *looks* like search output:
a handful of passages, mostly distractors, with the gold passages mixed in.
:func:`retrieve` provides the distractor half by running BM25 over the
non-supporting passages of the source record.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

Passage = dict[str, Any]


@lru_cache(maxsize=1)
def _chinese_tokenizer():
    """Build a jieba-backed tokenizer compatible with ``bm25s``."""
    import bm25s
    import jieba
    from bm25s.tokenization import Tokenized

    def tokenize(texts, return_ids: bool = True, show_progress: bool = False, leave: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        vocab: dict[str, int] = {}
        corpus_ids: list[list[int]] = []
        for text in texts:
            corpus_ids.append([vocab.setdefault(token, len(vocab)) for token in jieba.lcut(text)])
        if return_ids:
            return Tokenized(ids=corpus_ids, vocab=vocab)
        reverse = list(vocab)
        return [[reverse[i] for i in ids] for ids in corpus_ids]

    return bm25s, tokenize


def retrieve(
    corpus: Sequence[Passage],
    query: str,
    top_k_min: int,
    top_k_max: int,
    language: str = "english",
) -> list[Passage]:
    """Return between ``top_k_min`` and ``top_k_max`` passages ranked by BM25.

    ``corpus`` items are ``{"title": ..., "content": ...}`` dicts; the ``content``
    field is what gets indexed.  The exact count is drawn at random inside the
    range so that generated tool responses vary in length like real ones.
    Returns ``[]`` for an empty corpus.
    """
    if not corpus:
        return []

    import bm25s

    if language == "chinese":
        module, tokenizer = _chinese_tokenizer()
        module.tokenizer = tokenizer

    contents = [passage["content"] for passage in corpus]
    by_content = {passage["content"]: passage for passage in corpus}

    retriever = bm25s.BM25(corpus=contents)
    retriever.index(bm25s.tokenize(contents, show_progress=False), show_progress=False)

    top_k = min(random.randint(top_k_min, top_k_max), len(contents))
    documents, _scores = retriever.retrieve(
        bm25s.tokenize(query, show_progress=False), k=top_k, show_progress=False
    )

    hits: list[Passage] = []
    for batch in documents:
        for content in batch:
            passage = by_content.get(content)
            if passage is not None:
                hits.append(passage)
    return hits


def deduplicate(passages: Sequence[Passage] | Sequence[Sequence[Passage]]) -> list[Passage]:
    """Flatten, drop passages with duplicate ``content``, and shuffle.

    Accepts either a flat list or a list of per-tool-call lists.  Shuffling
    matters: it stops the gold passages from always landing at the end of a
    tool response, which the model would otherwise learn to exploit.
    """
    if not passages:
        return []

    flat: list[Passage] = []
    for item in passages:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)

    unique: dict[str, Passage] = {}
    for passage in flat:
        if isinstance(passage, dict) and "content" in passage:
            unique.setdefault(passage["content"], passage)

    result = list(unique.values())
    random.shuffle(result)
    return result
