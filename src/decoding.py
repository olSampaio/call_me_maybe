"""Constrained decoding: only let the model pick valid tokens."""

from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt

from .model_api import LanguageModel
from .vocabulary import Vocabulary


class GenerationError(Exception):
    """Raised when constrained generation cannot continue."""


def mask_logits(
    logits: Sequence[float], allowed: Collection[int]
) -> npt.NDArray[np.float64]:
    """Set the logit of every token that is not allowed to -infinity.

    Args:
        logits: One score per token, as returned by the model.
        allowed: Ids of the tokens that may be chosen. Ids outside the logits
            vector are ignored.

    Returns:
        A new array with the same length as `logits`, where only the allowed
        tokens keep their score.
    """
    values = np.asarray(logits, dtype=np.float64)
    ids = np.fromiter(
        (i for i in allowed if 0 <= i < values.size), dtype=np.int64
    )
    masked = np.full(values.size, -np.inf, dtype=np.float64)
    masked[ids] = values[ids]
    return masked


def pick_allowed(logits: Sequence[float], allowed: Collection[int]) -> int:
    """Return the allowed token with the highest logit (greedy choice).

    Args:
        logits: One score per token, as returned by the model.
        allowed: Ids of the tokens that may be chosen.

    Returns:
        The id of the allowed token with the highest logit. If several tokens
        tie, the lowest id wins.

    Raises:
        GenerationError: If no allowed token has a usable (finite) score.
    """
    masked = mask_logits(logits, allowed)
    best = int(np.argmax(masked))
    if not np.isfinite(masked[best]):
        raise GenerationError("No allowed token has a usable score")
    return best


def _choose_literal(
    model: LanguageModel,
    vocab: Vocabulary,
    context_ids: Sequence[int],
    options: Sequence[str],
    terminator_id: int | None,
) -> tuple[str, list[int]]:
    """Let the model pick one of `options`, token by token.

    At each step only the tokens that keep the text written so far on a path to
    at least one option are allowed. When a single option is left, the rest of
    it is written without asking the model again, because the choice was
    already made.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context.
        options: The valid texts.
        terminator_id: Token that ends the literal (the closing quote), or
            None. It becomes an extra choice when an option is complete but is
            also the beginning of a longer option.

    Returns:
        The chosen option and the ids generated for it. The terminator itself
        is not included.

    Raises:
        GenerationError: If `options` is empty or has an empty text, or if no
            token can continue the text.
    """
    candidates = list(dict.fromkeys(options))
    if not candidates or "" in candidates:
        raise GenerationError("Options must be a non-empty list of texts")

    ids = list(context_ids)
    new_ids: list[int] = []
    written = ""
    while True:
        alive = [text for text in candidates if text.startswith(written)]
        longer = [text for text in alive if len(text) > len(written)]
        complete = written in alive
        if complete and not longer:
            return written, new_ids
        if not complete and len(alive) == 1:
            # Only one way forward: the choice was already made.
            new_ids += model.encode(alive[0][len(written):])
            return alive[0], new_ids

        allowed = vocab.candidates(text[len(written):] for text in longer)
        if complete and terminator_id is not None:
            allowed.add(terminator_id)
        if not allowed:
            raise GenerationError(
                f"No token can continue the text {written!r}"
            )

        token = pick_allowed(model.get_logits(ids), allowed)
        if complete and token == terminator_id:
            return written, new_ids
        ids.append(token)
        new_ids.append(token)
        written += vocab.texts[token]


def choose_name(
    model: LanguageModel,
    vocab: Vocabulary,
    context_ids: Sequence[int],
    names: Sequence[str],
) -> tuple[str, list[int]]:
    """Let the model pick one function name; the result is always valid.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context.
        names: The valid function names.

    Returns:
        The chosen name and the ids generated for it, without the closing quote
        of the JSON string.

    Raises:
        GenerationError: If `names` is empty or has an empty name, or if no
            token can continue a name.
    """
    return _choose_literal(model, vocab, context_ids, names, vocab.id_of('"'))


def choose_boolean(
    model: LanguageModel, vocab: Vocabulary, context_ids: Sequence[int]
) -> tuple[bool, list[int]]:
    """Let the model choose between the JSON literals true and false.

    The literals are written with their leading space (' true', ' false'),
    because the context ends right after the colon of the key.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context.

    Returns:
        The chosen boolean and the ids generated for it.

    Raises:
        GenerationError: If no token can continue the literal.
    """
    text, ids = _choose_literal(
        model, vocab, context_ids, [" true", " false"], None
    )
    return text.strip() == "true", ids
