"""Adapter between the provided llm_sdk and the interface we use."""

import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .io_utils import FileError
from .model_api import LanguageModel
from .vocabulary import Vocabulary


class ModelLoadError(Exception):
    """Raised when the model or its vocabulary cannot be loaded."""


class SdkModel(BaseModel):
    """Wrap a `Small_LLM_Model`, using only its public methods.

    Attributes:
        raw: The `Small_LLM_Model` instance that is wrapped.
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


def load_model_and_vocab() -> tuple[LanguageModel, Vocabulary]:
    """Load the default Qwen model and its vocabulary.

    The SDK is imported here, not at the top of the module, so that the rest of
    the project works without torch installed.

    Returns:
        The model and its vocabulary.

    Raises:
        ModelLoadError: If the SDK, the model or the vocabulary cannot be
            loaded.
    """
    try:
        sdk = importlib.import_module("llm_sdk")
        if not hasattr(sdk, "Small_LLM_Model"):
            raise ModelLoadError(
                "The llm_sdk package does not provide Small_LLM_Model. "
                "Run 'uv sync' and check that llm_sdk/ is the folder that "
                "was provided with the subject."
            )
        model = SdkModel(raw=sdk.Small_LLM_Model())
        vocab = Vocabulary.from_file(model.vocab_path())
    except ModelLoadError:
        raise
    except FileError as err:
        raise ModelLoadError(f"Cannot load the vocabulary: {err}") from err
    except Exception as err:
        # Any failure (missing package, no memory, no network...) must be
        # reported with a clear message instead of crashing.
        raise ModelLoadError(
            f"Cannot load the model ({type(err).__name__}): {err}"
        ) from err
    return model, vocab
