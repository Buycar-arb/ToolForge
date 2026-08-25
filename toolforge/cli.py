"""``toolforge`` — one command for the whole pipeline.

::

    toolforge doctor                       check the environment and credentials
    toolforge cases                        list the 29 dialogue cases
    toolforge label   IN OUT               stage 2: label questions with tools
    toolforge variants TOOL.json OUT       stage 1: grow a tool library
    toolforge generate LABELS --case ...   stages 3+4: generate and validate
    toolforge validate DATA                re-run the rule checks on existing data
    toolforge convert to-jsonl DIR         Parquet -> JSONL (and to-parquet back)
    toolforge webui [--lang en]            launch the visual interface (Chinese by default)

Every command reads defaults from ``.env`` — run ``toolforge doctor`` first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from toolforge import jsonl
from toolforge.config import ROOT_DIR, reload_settings, settings


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Print the resolved configuration and flag anything that will not work."""
    from toolforge.llm import resolve_provider
    from toolforge.toolbank import domain_names

    config = reload_settings()
    print("ToolForge configuration\n" + "-" * 60)
    print(config.describe())

    problems: list[str] = []

    for role, model in (("generation", config.generation_model), ("judge", config.judge_model)):
        provider, model_id = resolve_provider(model)
        keys = config.keys_for(provider)
        marker = "ok " if keys else "!! "
        print(f"\n{marker}{role} model '{model_id}' → provider '{provider}'")
        if not keys:
            variable = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            problems.append(f"no key for the {role} model — set {variable} in .env")
        if provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError:
                problems.append(
                    f"the {role} model needs the Anthropic SDK — run: pip install 'toolforge[anthropic]'"
                )

    tools = domain_names(config.tool_bank_dir)
    print(f"\n{'ok ' if tools else '!! '}tool bank: {len(tools)} domain libraries at {config.tool_bank_dir}")
    if not tools:
        problems.append(f"the tool bank at {config.tool_bank_dir} is empty or missing")

    # A missing .env only matters when something is actually unset — passing the
    # variables in the environment directly is a perfectly good setup.
    if not (ROOT_DIR / ".env").is_file():
        if problems:
            problems.append("no .env file — copy .env.example to .env and fill it in")
        else:
            print("\n·  no .env file, but the environment supplies everything needed")

    print("\n" + "-" * 60)
    if problems:
        print("Problems found:")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("Everything checks out.")
    return 0


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def cmd_cases(args: argparse.Namespace) -> int:
    """List the dialogue cases, or explain one in detail."""
    from toolforge.prompts.flows import CASE_FLOWS
    from toolforge.stages.cases import CASE_SPECS, get, summary_table

    if args.case:
        spec = get(args.case)
        print(f"{spec.case_id} — {spec.description}\n")
        print(f"  family            : {spec.family} ({spec.turns} turn(s))")
        print(f"  source route      : {spec.source_route}  (what stage 2 must have labelled)")
        print(f"  tool messages     : {len(spec.tool_messages)}  {list(spec.tool_messages)}")
        print(f"  fallback tool set : {spec.use_fallback_tools}")
        print(f"  tool policy       : {spec.tool_policy.value}")
        print(f"  argument check    : {spec.argument_check_range or 'disabled'}")
        print(f"\n  reasoning flow:\n{CASE_FLOWS[spec.case_id]}")
        return 0

    from toolforge.stages.cases import ROUTE_TO_FAMILY

    print(summary_table())
    print("\nA case only fits records stage 2 routed to its family:\n")
    for route, family in ROUTE_TO_FAMILY.items():
        cases = ", ".join(c for c in CASE_SPECS if c[5] == family)
        print(f"  {route}  →  family {family}   {cases}")
    print(f"\n{len(CASE_SPECS)} cases.  Use `toolforge cases --case case_C1` for the details of one.")
    return 0


# --------------------------------------------------------------------------- #
# label (stage 2)
# --------------------------------------------------------------------------- #


def cmd_label(args: argparse.Namespace) -> int:
    from toolforge.llm import LLMClient
    from toolforge.stages.labeling import ToolLabeler, run_labeling

    labeler = ToolLabeler(
        LLMClient(args.model) if args.model else None,
        force_single_call=args.single_call,
    )
    stats = asyncio.run(
        run_labeling(
            args.input,
            args.output,
            residue_file=args.residue,
            max_records=args.limit,
            concurrency=args.concurrency,
            labeler=labeler,
        )
    )
    print(stats.summary())
    return 0 if stats.labelled else 1


# --------------------------------------------------------------------------- #
# variants (stage 1)
# --------------------------------------------------------------------------- #


def cmd_variants(args: argparse.Namespace) -> int:
    from toolforge.llm import LLMClient
    from toolforge.stages.variants import VariantGenerator

    original = json.loads(Path(args.tool).read_text(encoding="utf-8"))
    generator = VariantGenerator(LLMClient(args.model) if args.model else None)
    variants = asyncio.run(
        generator.run(
            original,
            args.output,
            target=args.target,
            cosine_threshold=args.cosine_threshold,
            bm25_threshold=args.bm25_threshold,
        )
    )
    return 0 if len(variants) >= args.target else 1


# --------------------------------------------------------------------------- #
# generate (stages 3 + 4)
# --------------------------------------------------------------------------- #


def _build_jobs(args: argparse.Namespace) -> list:
    from toolforge.stages.pipeline import CaseJob

    output_dir = Path(args.output_dir)
    if args.config:
        blocks = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return [CaseJob.from_config(case_id, block, output_dir) for case_id, block in blocks.items()]
    return [
        CaseJob(
            case_id=case_id,
            target=args.target,
            data_output=output_dir / "data" / f"{case_id}.jsonl",
            score_output=output_dir / "scores" / f"{case_id}.jsonl",
        )
        for case_id in args.case
    ]


def cmd_generate(args: argparse.Namespace) -> int:
    from toolforge.llm import LLMClient
    from toolforge.stages.dialogue import DialogueGenerator
    from toolforge.stages.judge import DialogueJudge
    from toolforge.stages.pipeline import Pipeline, format_report, load_records, route_summary
    from toolforge.stages.validation import ValidationOptions

    jobs = _build_jobs(args)
    if not jobs:
        print("Nothing to do — pass --case or --config.", file=sys.stderr)
        return 2

    records = load_records(args.input)
    if not records:
        print(f"No labelled records in {args.input}. Run `toolforge label` first.", file=sys.stderr)
        return 1
    print(f"{len(records)} labelled records loaded from {args.input}")
    print(route_summary(records))

    pipeline = Pipeline(
        generator=DialogueGenerator(LLMClient(args.model) if args.model else None),
        judge=DialogueJudge(LLMClient(args.judge_model) if args.judge_model else None),
        validation_options=ValidationOptions(
            strict_reference_check=args.strict_references,
            strict_final_answer_format=args.strict_answer_format,
            require_argument_change=args.strict_argument_change,
        ),
    )
    results = asyncio.run(
        pipeline.run(
            records,
            jobs,
            on_event=print,
            delay=args.delay,
            concurrency=args.concurrency,
        )
    )
    print("\n" + format_report(results))
    return 0 if all(progress.complete for progress in results.values()) else 1


# --------------------------------------------------------------------------- #
# validate (stage 4 only)
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    """Re-run the rule checks over an existing generated-data file."""
    from collections import Counter

    from toolforge.stages.validation import ValidationOptions, validate

    options = ValidationOptions(
        strict_reference_check=args.strict_references,
        strict_final_answer_format=args.strict_answer_format,
        require_argument_change=args.strict_argument_change,
    )
    passed = 0
    total = 0
    failures: Counter[str] = Counter()

    for record in jsonl.read(args.input):
        total += 1
        case_id = record[0]["case"] if isinstance(record, list) and record else None
        if not case_id:
            failures["record has no case header"] += 1
            continue
        outcome = validate(record, case_id, options)
        if outcome.passed:
            passed += 1
        else:
            failures.update(outcome.failures)

    print(f"\n{passed}/{total} records pass all nine checks")
    for reason, count in failures.most_common():
        print(f"  {count:>5}×  {reason}")
    if total == 0:
        print(f"No valid records found in {args.input}.", file=sys.stderr)
        return 1
    return 0 if passed == total else 1


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert a directory of Parquet files to JSONL, or the other way round."""
    from toolforge.convert import to_jsonl, to_parquet

    print(f"{args.direction}: {args.directory}")
    convert = to_jsonl if args.direction == "to-jsonl" else to_parquet
    written = convert(args.directory, args.output_dir)
    print(f"\n{len(written)} file(s) written.")
    return 0 if written else 1


# --------------------------------------------------------------------------- #
# webui
# --------------------------------------------------------------------------- #


def cmd_webui(args: argparse.Namespace) -> int:
    from toolforge.webui.app import launch

    launch(host=args.host, port=args.port, share=args.share, language=args.lang)
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolforge",
        description="An automated SFT data factory for LLM tool-calling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check configuration and credentials")
    doctor.set_defaults(func=cmd_doctor)

    cases = subparsers.add_parser("cases", help="list the dialogue cases")
    cases.add_argument("--case", help="explain one case in detail, e.g. case_C1")
    cases.set_defaults(func=cmd_cases)

    label = subparsers.add_parser("label", help="stage 2 — label questions with tools and routes")
    label.add_argument("input", help="raw multi-hop QA JSONL")
    label.add_argument("output", help="where to write the labelled JSONL")
    label.add_argument("--residue", help="where to park records beyond --limit")
    label.add_argument("--limit", type=int, help="label only the first N records")
    label.add_argument("--concurrency", type=int, default=settings.concurrency)
    label.add_argument("--model", help=f"override the model (default {settings.generation_model})")
    label.add_argument("--single-call", action="store_true",
                       help="force route case1 — used to top up the single-call class")
    label.set_defaults(func=cmd_label)

    variants = subparsers.add_parser("variants", help="stage 1 — grow a tool library")
    variants.add_argument("tool", help="JSON file holding the tool definition to paraphrase")
    variants.add_argument("output", help="tool library JSONL to append to")
    variants.add_argument("--target", type=int, default=20, help="how many variants in total")
    variants.add_argument("--cosine-threshold", type=float, default=0.7,
                          help="minimum semantic similarity to accept (default 0.7)")
    variants.add_argument("--bm25-threshold", type=float, default=0.6,
                          help="maximum lexical similarity to accept (default 0.6)")
    variants.add_argument("--model", help=f"override the model (default {settings.generation_model})")
    variants.set_defaults(func=cmd_variants)

    generate = subparsers.add_parser("generate", help="stages 3+4 — generate and validate dialogues")
    generate.add_argument("input", help="labelled JSONL from stage 2")
    generate.add_argument("--case", action="append", default=[],
                          help="case to generate (repeatable), e.g. --case case_C1")
    generate.add_argument("--target", type=int, default=10, help="samples per case (default 10)")
    generate.add_argument("--config", help="JSON file with per-case targets and output paths")
    generate.add_argument("--output-dir", default=str(settings.output_dir))
    generate.add_argument("--concurrency", type=int, default=settings.concurrency)
    generate.add_argument("--delay", type=float, default=1.0, help="seconds between attempts")
    generate.add_argument("--model", help=f"generation model (default {settings.generation_model})")
    generate.add_argument("--judge-model", help=f"judge model (default {settings.judge_model})")
    generate.add_argument("--strict-references", action="store_true",
                          help="enable check 7, which was inert in the published release")
    generate.add_argument("--strict-answer-format", action="store_true",
                          help="reject a malformed final <answer> block instead of warning")
    generate.add_argument("--strict-argument-change", action="store_true",
                          help="reject a retry whose arguments are identical to the failed call")
    generate.set_defaults(func=cmd_generate)

    validate_cmd = subparsers.add_parser("validate", help="re-run the rule checks on generated data")
    validate_cmd.add_argument("input", help="generated-data JSONL")
    validate_cmd.add_argument("--strict-references", action="store_true")
    validate_cmd.add_argument("--strict-answer-format", action="store_true")
    validate_cmd.add_argument("--strict-argument-change", action="store_true")
    validate_cmd.set_defaults(func=cmd_validate)

    convert = subparsers.add_parser("convert", help="convert between Parquet and JSONL")
    convert.add_argument("direction", choices=["to-jsonl", "to-parquet"])
    convert.add_argument("directory", help="directory to convert (all matching files in it)")
    convert.add_argument("--output-dir", help="write elsewhere (default: alongside the source)")
    convert.set_defaults(func=cmd_convert)

    webui = subparsers.add_parser("webui", help="launch the visual interface")
    webui.add_argument(
        "--host",
        default="127.0.0.1",
        help="listen address; non-loopback hosts require Web UI credentials",
    )
    webui.add_argument("--port", type=int, default=7860)
    webui.add_argument(
        "--share",
        action="store_true",
        help="create an authenticated public Gradio link",
    )
    webui.add_argument("--lang", choices=["zh", "en"], default=None,
                       help="interface language (default: zh, or $UI_LANG)")
    webui.set_defaults(func=cmd_webui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
