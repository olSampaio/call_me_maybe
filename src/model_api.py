"""The minimal interface the generation code needs from a language model."""

from typing import Protocol


class LogitsProvider(Protocol):
    """Anything that can score the next token for a sequence of ids."""

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return one score (logit) per token of the model vocabulary.

        Args:
            input_ids: Token ids already in the context.

        Returns:
            The logits of the next token.
        """
        ...


class LanguageModel(LogitsProvider, Protocol):
    """A model that can also turn text into token ids."""

    def encode(self, text: str) -> list[int]:
        """Convert a text into token ids.

        Args:
            text: Text to tokenise.

        Returns:
            The token ids of `text`.
        """
        ...
