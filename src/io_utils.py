"""Reading the input files and writing the output file."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .models import FunctionCall, FunctionDefinition, PromptItem


class FileError(Exception):
    """Raised when a file cannot be read, parsed, validated or written."""


def read_json(path: Path) -> Any:
    """Read a JSON file, converting any failure into a `FileError`.

    Args:
        path: File to read.

    Returns:
        The parsed JSON content.

    Raises:
        FileError: If the file is missing, cannot be read, is not valid UTF-8
            or is not valid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileError(f"File not found: {path}") from None
    except json.JSONDecodeError as err:
        raise FileError(f"Invalid JSON in {path}: {err}") from err
    except UnicodeDecodeError as err:
        raise FileError(f"Cannot decode {path} as UTF-8: {err}") from err
    except OSError as err:
        raise FileError(f"Cannot read {path}: {err}") from err


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions.

    Args:
        path: JSON file with the list of function definitions.

    Returns:
        The validated function definitions.

    Raises:
        FileError: If the file cannot be read, has the wrong structure or
            defines no function.
    """
    data = read_json(path)
    try:
        functions = TypeAdapter(list[FunctionDefinition]).validate_python(
            data
        )
    except ValidationError as err:
        raise FileError(f"Invalid content in {path}: {err}") from err
    if not functions:
        raise FileError(f"No function defined in {path}")
    names = [function.name for function in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise FileError(
            f"Duplicate function names in {path}: {', '.join(duplicates)}"
        )
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the list of prompts.

    Args:
        path: JSON file with a list of objects that have a `prompt`.

    Returns:
        The validated prompts.

    Raises:
        FileError: If the file cannot be read or has the wrong structure.
    """
    data = read_json(path)
    try:
        return TypeAdapter(list[PromptItem]).validate_python(data)
    except ValidationError as err:
        raise FileError(f"Invalid content in {path}: {err}") from err


def check_output_path(path: Path) -> None:
    """Make sure the output file can be written, before any slow work.

    Missing folders are created. The file itself is not created.

    Args:
        path: The output file.

    Returns:
        None.

    Raises:
        FileError: If the path is a directory or its folder cannot be
            created or written to.
    """
    if path.is_dir():
        raise FileError(f"Output path is a directory: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise FileError(f"Cannot create the folder of {path}: {err}") from err
    if path.exists():
        writable = os.access(path, os.W_OK)
    else:
        writable = os.access(path.parent, os.W_OK)
    if not writable:
        raise FileError(f"Cannot write {path}: permission denied")


def save_calls(path: Path, calls: list[FunctionCall]) -> None:
    """Write the function calls to a JSON file.

    NaN and infinity are rejected, so the file is always strict JSON.

    Args:
        path: File to write. Missing folders are created.
        calls: The function calls to write.

    Returns:
        None.

    Raises:
        FileError: If the file cannot be written or a value is not strict JSON
            (NaN or infinity).
    """
    payload = [call.model_dump() for call in calls]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload, file, indent=2, ensure_ascii=False, allow_nan=False
            )
    except (OSError, ValueError) as err:
        raise FileError(f"Cannot write {path}: {err}") from err
