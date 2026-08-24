#!/usr/bin/env python3
"""Run a tool-calling model over ToolForge-generated data and record its trajectory.

This replays each generated sample as a *live* agent loop: the model is given the
same system prompt and question, and whenever it emits a ``<tool_call>`` the
tool is answered by running BM25 over that record's own passages.  What the
model does with those results is entirely its own.

The same script evaluates both sides of a comparison — the only difference is
where the model is served:

    # your fine-tuned checkpoint, served locally by Swift + vLLM
    python run_benchmark.py data.jsonl ours.jsonl \\
        --model history-8B --base-url http://0.0.0.0:8000/v1 --api-key EMPTY

    # a baseline, over an API
    python run_benchmark.py data.jsonl baseline.jsonl --model gpt-5.1

Drop the two output files into ``viewer/compare.html`` to see them side by side.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toolforge import bm25, jsonl  # noqa: E402
from toolforge.llm import LLMClient  # noqa: E402

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """Every ``<tool_call>`` block the model emitted, as parsed JSON."""
    calls: list[dict[str, Any]] = []
    for raw in TOOL_CALL.findall(content or ""):
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            print(f"  ! unparseable tool call ({exc}): {raw.strip()[:120]}")
            continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": payload.get("name", ""),
            "arguments": payload.get("arguments", {}),
        })
    return calls


def arguments_to_query(arguments: dict[str, Any]) -> str:
    """Flatten a tool call's arguments into one retrieval query."""
    parts = []
    for name, value in arguments.items():
        if value is None:
            continue
        rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        parts.append(f"{name}: {rendered}")
    return "\n".join(parts)


def format_passages(passages: list[dict[str, str]], start: int = 1) -> str:
    """Render retrieved passages in the numbered format the training data uses."""
    return "\n\n".join(
        f"**{index}**\ntitle: {p.get('title', '')}\ncontent: {p.get('content', '')}"
        for index, p in enumerate(passages, start)
    )


def corpus_from(record: list) -> list[dict[str, str]]:
    """Every passage of the source record — the search space for this question."""
    return [
        {"title": title, "content": sentence}
        for title, sentences in record[6]["context"]
        for sentence in sentences
    ]


def answer_tool_calls(calls: list[dict[str, Any]], corpus: list[dict[str, str]], top_k: int) -> str:
    """Run retrieval for each call and merge the results into one tool message."""
    blocks: list[str] = []
    cursor = 1
    for call in calls:
        hits = bm25.retrieve(corpus, arguments_to_query(call["arguments"]), top_k, top_k)
        blocks.append(format_passages(hits, start=cursor))
        cursor += len(hits)
    return "\n\n".join(blocks)


async def run_sample(
    client: LLMClient, record: list, *, top_k: int, max_rounds: int, verbose: bool
) -> dict[str, Any]:
    """Replay one sample as a live agent loop and return its trajectory."""
    question = record[6]["question"]
    gold = record[6]["answer"]
    corpus = corpus_from(record)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": record[1]["messages"][0]["content"]},
        {"role": "user", "content": record[1]["messages"][1]["content"]},
    ]

    reply = await client.complete(messages[1:], system=messages[0]["content"])
    messages.append({"role": "assistant", "content": reply})
    if verbose:
        print(f"  assistant 1: {reply[:160]}")

    calls = parse_tool_calls(reply)
    for round_number in range(max_rounds):
        if not calls:
            break
        messages.append({"role": "tool", "content": answer_tool_calls(calls, corpus, top_k)})

        follow_up = await client.complete(messages[1:], system=messages[0]["content"])
        messages.append({"role": "assistant", "content": follow_up})
        if verbose:
            print(f"  assistant {round_number + 2}: {follow_up[:160]}")

        found = ANSWER.search(follow_up)
        if found:
            correct = found.group(1).strip() == str(gold).strip()
            print(f"  → {'✓' if correct else '✗'} {found.group(1).strip()[:70]}  (gold: {gold})")
            break
        calls = parse_tool_calls(follow_up)
    else:
        print(f"  → stopped after {max_rounds} tool rounds without an answer")

    return {
        "sample_id": record[0].get("uuid", ""),
        "original_query": question,
        "golden_answer": gold,
        "messages": messages,
    }


async def run(args: argparse.Namespace) -> int:
    client = LLMClient(
        args.model,
        api_keys=[args.api_key] if args.api_key else None,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"model: {client.model} via {client.provider} at {client.base_url}\n")

    records = [r for r in jsonl.read(args.input) if isinstance(r, list) and len(r) >= 7]
    if not records:
        print(f"No ToolForge generated-data records in {args.input}", file=sys.stderr)
        return 1
    if args.limit:
        records = records[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    correct = 0
    for index, record in enumerate(records, 1):
        print(f"[{index}/{len(records)}] {record[6]['question'][:80]}")
        try:
            result = await run_sample(
                client, record, top_k=args.top_k, max_rounds=args.max_rounds, verbose=args.verbose
            )
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the sweep
            print(f"  ! failed: {exc}")
            continue

        result["sample_id"] = result["sample_id"] or index
        jsonl.append(output, result)

        final = next(
            (ANSWER.search(m["content"]) for m in reversed(result["messages"])
             if m["role"] == "assistant" and ANSWER.search(m["content"])), None
        )
        if final and final.group(1).strip() == str(result["golden_answer"]).strip():
            correct += 1

    print(f"\n{correct}/{len(records)} exact match ({correct / len(records) * 100:.1f}%)")
    print(f"written to {output}")
    print("\nCompare two runs by opening viewer/compare.html and dropping both files in.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="ToolForge generated-data JSONL (output/data/case_XX.jsonl)")
    parser.add_argument("output", help="where to write the trajectories")
    parser.add_argument("--model", required=True, help="model id, or the vLLM --served-model-name")
    parser.add_argument("--base-url", help="override the endpoint (e.g. http://0.0.0.0:8000/v1)")
    parser.add_argument("--api-key", help="single key; omit to use the keys from .env")
    parser.add_argument("--limit", type=int, help="only evaluate the first N samples")
    parser.add_argument("--top-k", type=int, default=10, help="passages per tool call (default 10)")
    parser.add_argument("--max-rounds", type=int, default=5, help="tool rounds before giving up")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("-v", "--verbose", action="store_true", help="print every model turn")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
