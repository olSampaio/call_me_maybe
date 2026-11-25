"""The language model as seen by the generation code."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class LanguageModel(BaseModel):
    """Wrap a `Small_LLM_Model`, using only its public methods.

    Attributes:
        raw: The `Small_LLM_Model` instance that is wrapped (any object with
            the same public methods works, which makes testing easy).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw: Any

    def encode(self, text: str) -> list[int]:
        """Convert a text into token ids.

        Args:
            text: Text to tokenise.

        Returns:
            The token ids of `text` as a plain list of ints.
        """
        tensor = self.raw.encode(text)
        return [int(token_id) for token_id in tensor[0].tolist()]

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return the next-token logits for a sequence of ids.

        Args:
            input_ids: Token ids already in the context.

        Returns:
            One logit per token of the model vocabulary.
        """
        logits = self.raw.get_logits_from_input_ids(input_ids)
        if hasattr(logits, "tolist"):
            logits = logits.tolist()
        return [float(value) for value in logits]

    def vocab_path(self) -> Path:
        """Return the path of the vocabulary file of the model.

        Returns:
            The path of the vocabulary JSON file.
        """
        return Path(str(self.raw.get_path_to_vocab_file()))
