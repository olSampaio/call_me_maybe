"""Constrained generation of JSON strings."""

import json
from collections.abc import Sequence

from .decoding import GenerationError, pick_allowed
from .model_api import LanguageModel
from .vocabulary import Vocabulary

MAX_STRING_TOKENS = 100


def generate_string(
    model: LanguageModel,
    vocab: Vocabulary,
    context_ids: Sequence[int],
) -> tuple[str, list[int]]:
    """Let the model write a JSON string, from its opening quote.

    The context ends right after the colon of the key. The first token must
    start with a space and the opening quote (' "', ' "/'...), which is how
    the tokenizer naturally splits JSON. After it, allowed tokens are those
    that are safe inside a JSON string (no raw control characters, no broken
    escapes) plus the tokens that contain the closing quote. Choosing one of
    the latter ends the string: only the text before the quote is kept, and
    the caller writes the closing quote itself.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context, ending right after the
            colon that follows the key.

    Returns:
        The decoded string value and the generated ids (the opening quote
        included, the closing quote not).

    Raises:
        GenerationError: If no token can open the string or the generated
            text is not a valid string.
    """
    openers, closed_openers = vocab.string_openers()
    if not openers and not closed_openers:
        raise GenerationError("No token can open a string")
    ids = list(context_ids)
    first = pick_allowed(
        model.get_logits(ids), set(openers) | set(closed_openers)
    )
    if first in closed_openers:
        # The string was opened and closed by the same token (' ""').
        raw = closed_openers[first]
        return _decode(raw), model.encode(' "' + raw)

    plain, ends = vocab.string_tokens()
    allowed = plain | frozenset(ends)
    ids.append(first)
    new_ids: list[int] = [first]
    raw = openers[first]
    for _ in range(MAX_STRING_TOKENS):
        token = pick_allowed(model.get_logits(ids), allowed)
        if token in ends:
            before_quote = ends[token]
            if before_quote:
                raw += before_quote
                new_ids += model.encode(before_quote)
            break
        ids.append(token)
        new_ids.append(token)
        raw += vocab.texts[token]
    return _decode(raw), new_ids


def _decode(raw: str) -> str:
    """Decode the raw content of a JSON string (escapes included).

    Args:
        raw: The text between the quotes, as generated.

    Returns:
        The string value.

    Raises:
        GenerationError: If `raw` is not valid JSON string content.
    """
    try:
        value = json.loads('"' + raw + '"')
        value.encode("utf-8")
    except (ValueError, UnicodeEncodeError) as err:
        raise GenerationError(f"Invalid string generated: {raw!r}") from err
    return str(value)
