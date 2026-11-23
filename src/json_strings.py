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
    """Let the model write the inside of a JSON string.

    Allowed tokens are those that are safe inside a JSON string (no raw control
    characters, no broken escapes) plus the tokens that contain the closing
    quote. Choosing one of the latter ends the string: only the text before the
    quote is kept, and the caller writes the closing quote itself.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context. The opening quote must be the
            last one.

    Returns:
        The decoded string value and the generated ids.

    Raises:
        GenerationError: If the generated text is not a valid string.
    """
    plain, ends = vocab.string_tokens()
    allowed = plain | frozenset(ends)
    ids = list(context_ids)
    new_ids: list[int] = []
    raw = ""
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

    try:
        value = json.loads('"' + raw + '"')
        value.encode("utf-8")
    except (ValueError, UnicodeEncodeError) as err:
        raise GenerationError(f"Invalid string generated: {raw!r}") from err
    return str(value), new_ids
