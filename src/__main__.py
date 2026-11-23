"""Entry point: uv run python -m src [options]."""

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .decoding import GenerationError
from .io_utils import FileError, load_functions, load_prompts, save_calls
from .llm import ModelLoadError, load_model_and_vocab
from .model_api import LanguageModel
from .models import FunctionCall
from .pipeline import generate_call
from .vocabulary import Vocabulary

ModelLoader = Callable[[], tuple[LanguageModel, Vocabulary]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse. When None, `sys.argv` is used.

    Returns:
        The parsed options `functions_definition`, `input` and `output`, all of
        them `Path` objects.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="JSON file describing the available functions",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="JSON file containing the prompts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
        help="JSON file where the function calls are written",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, load_model: ModelLoader) -> int:
    """Process every prompt and write the output file.

    The input files are read first, so a missing file is reported before the
    slow model loading. A prompt that fails is reported and skipped; the other
    prompts are still processed and written.

    Args:
        args: Parsed command-line arguments.
        load_model: Function that returns the model and its vocabulary. It is
            called only after the input files were read successfully.

    Returns:
        The process exit code: 0 if every prompt was processed, 1 if there was
        an error.
    """
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except FileError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    try:
        model, vocab = load_model()
    except ModelLoadError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    calls: list[FunctionCall] = []
    failures = 0
    started = time.time()
    for position, item in enumerate(prompts, start=1):
        try:
            call = generate_call(model, vocab, functions, item.prompt)
        except GenerationError as err:
            failures += 1
            print(
                f"[{position}/{len(prompts)}] FAILED {item.prompt!r}: {err}",
                file=sys.stderr,
            )
            continue
        calls.append(call)
        print(
            f"[{position}/{len(prompts)}] {item.prompt!r} -> "
            f"{call.name}({call.parameters})"
        )

    try:
        save_calls(args.output, calls)
    except FileError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    elapsed = time.time() - started
    print(
        f"Done: {len(calls)} calls written to {args.output} "
        f"({failures} failed) in {elapsed:.1f}s."
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Run the program and return the process exit code.

    Args:
        argv: Command-line arguments. When None, `sys.argv` is used.

    Returns:
        The exit code: 0 on success, 1 on error, 130 when interrupted.
    """
    args = parse_args(argv)
    try:
        return run(args, load_model_and_vocab)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
