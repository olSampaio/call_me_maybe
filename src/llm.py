"""Loading the provided llm_sdk model and its vocabulary."""

import importlib

from .io_utils import FileError
from .model_api import LanguageModel
from .vocabulary import Vocabulary


class ModelLoadError(Exception):
    """Raised when the model or its vocabulary cannot be loaded."""


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
        model = LanguageModel(raw=sdk.Small_LLM_Model())
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
