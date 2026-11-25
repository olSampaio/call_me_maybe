"""Constrained generation of JSON numbers."""

import math
import re
from collections.abc import Sequence

from .decoding import GenerationError, pick_allowed
from .model_api import LanguageModel
from .vocabulary import Vocabulary

MAX_NUMBER_CHARS = 30

# A "prefix" regex accepts every text that can still become a valid
# number; a "full" regex accepts only finished numbers. Both accept one
# leading space: the context ends right after the colon, and the tokenizer
# merges that space with what follows (' -', ' 4').
_NUMBER_PREFIX = re.compile(
    r" ?(?:-|-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?)?"
)
_NUMBER_FULL = re.compile(r" ?-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_INTEGER_PREFIX = re.compile(r" ?(?:-|-?(?:0|[1-9][0-9]*))?")
_INTEGER_FULL = re.compile(r" ?-?(?:0|[1-9][0-9]*)")

_NUMBER_START_CHARS = "-0123456789."
_INTEGER_START_CHARS = "-0123456789"
# Tokens starting with these characters mean "the number is finished":
# in the JSON we are writing, a value is followed by "," or "}".
_STOP_CHARS = ",}"


def number_continuations(
    vocab: Vocabulary, written: str, integer: bool
) -> set[int]:
    """Return the tokens that keep `written` a valid number prefix.

    Args:
        vocab: Token id -> text table.
        written: Text of the number written so far.
        integer: True to accept only integers (no decimal point).

    Returns:
        The ids of the tokens that can be appended to `written`.
    """
    prefix = _INTEGER_PREFIX if integer else _NUMBER_PREFIX
    start = _INTEGER_START_CHARS if integer else _NUMBER_START_CHARS
    if not written:
        # Only the first token may bring the leading space.
        start = " " + start
    return {
        token_id
        for token_id in vocab.starting_with(start)
        if prefix.fullmatch(written + vocab.texts[token_id])
    }


def is_complete_number(text: str, integer: bool) -> bool:
    """Tell whether `text` is a finished number of the requested kind.

    Args:
        text: Text to check.
        integer: True to require an integer (no decimal point).

    Returns:
        True if `text` is a complete number, False otherwise.
    """
    full = _INTEGER_FULL if integer else _NUMBER_FULL
    return full.fullmatch(text) is not None


def generate_number(
    model: LanguageModel,
    vocab: Vocabulary,
    context_ids: Sequence[int],
    integer: bool = False,
) -> tuple[float | int, list[int]]:
    """Let the model write a number, digit by digit.

    Only tokens that keep the text a valid number are allowed. Once the number
    is complete, the tokens that start with a comma or a closing brace are also
    allowed: choosing one of them means that the number is over. That token is
    not kept; the caller writes the real JSON punctuation itself.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context.
        integer: Generate an integer instead of a decimal number.

    Returns:
        The value (a float, or an int when `integer` is True) and the generated
        ids.

    Raises:
        GenerationError: If no token can continue the number or the value is
            out of range.
    """
    stop_ids = vocab.starting_with(_STOP_CHARS)
    ids = list(context_ids)
    new_ids: list[int] = []
    written = ""
    while True:
        complete = is_complete_number(written, integer)
        if complete and len(written) >= MAX_NUMBER_CHARS:
            break
        allowed = number_continuations(vocab, written, integer)
        if complete:
            allowed |= stop_ids
        if not allowed:
            raise GenerationError(
                f"No token can continue the number {written!r}"
            )
        token = pick_allowed(model.get_logits(ids), allowed)
        if complete and token in stop_ids:
            break
        ids.append(token)
        new_ids.append(token)
        written += vocab.texts[token]

    written = written.strip()
    if integer:
        return int(written), new_ids
    value = float(written)
    if not math.isfinite(value):
        raise GenerationError(f"Number out of range: {written!r}")
    return value, new_ids
