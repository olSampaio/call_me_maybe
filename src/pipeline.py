"""From one natural-language prompt to one structured function call."""

import json
import math
from collections.abc import Sequence
from typing import Any

from .decoding import GenerationError, choose_boolean, choose_name
from .json_numbers import generate_number
from .json_strings import generate_string
from .model_api import LanguageModel
from .models import FunctionCall, FunctionDefinition
from .prompt import NAME_PREFIX, build_prompt
from .vocabulary import Vocabulary


class UnsupportedTypeError(GenerationError):
    """Raised when a parameter has a type we cannot generate."""


def generate_value(
    model: LanguageModel,
    vocab: Vocabulary,
    context_ids: Sequence[int],
    type_name: str,
) -> tuple[Any, list[int]]:
    """Generate one argument value of the given JSON-schema type.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        context_ids: Ids already in the context. They must end right after
            the colon that follows the key: the first generated token brings
            the space (and, for a string, the opening quote) with it, which is
            how the tokenizer naturally splits JSON.
        type_name: Type of the parameter: number, integer, string or boolean.

    Returns:
        The generated value and the generated ids.

    Raises:
        UnsupportedTypeError: If the type is not supported.
    """
    if type_name == "number":
        return generate_number(model, vocab, context_ids, integer=False)
    if type_name == "integer":
        return generate_number(model, vocab, context_ids, integer=True)
    if type_name == "string":
        return generate_string(model, vocab, context_ids)
    if type_name == "boolean":
        return choose_boolean(model, vocab, context_ids)
    raise UnsupportedTypeError(f"Unsupported parameter type: {type_name!r}")


def check_call(function: FunctionDefinition, call: FunctionCall) -> None:
    """Verify that a call matches the definition it was generated for.

    Args:
        function: The definition the call must follow.
        call: The generated call.

    Returns:
        None.

    Raises:
        GenerationError: If the name, the keys, their order or the value types
            do not match.
    """
    if call.name != function.name:
        raise GenerationError(f"Wrong function name: {call.name!r}")
    if list(call.parameters) != list(function.parameters):
        raise GenerationError(f"Wrong parameters for {function.name}")
    for key, spec in function.parameters.items():
        value = call.parameters[key]
        if spec.type == "number":
            ok = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        elif spec.type == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif spec.type == "string":
            ok = isinstance(value, str)
        elif spec.type == "boolean":
            ok = isinstance(value, bool)
        else:
            ok = False
        if not ok:
            raise GenerationError(
                f"Parameter {key!r} of {function.name} has a wrong type"
            )


def generate_call(
    model: LanguageModel,
    vocab: Vocabulary,
    functions: Sequence[FunctionDefinition],
    prompt: str,
) -> FunctionCall:
    """Turn a prompt into a function call using constrained decoding.

    The JSON is written in pieces. The fixed pieces (keys and punctuation) are
    put in the context directly. The decisions (which function, and the value
    of each argument) are made by the model, restricted to the tokens that keep
    the output valid for the definitions.

    Args:
        model: Language model that provides the logits and `encode`.
        vocab: Token id -> text table of that model.
        functions: The available functions.
        prompt: The natural-language request.

    Returns:
        The validated function call.

    Raises:
        GenerationError: If generation fails for this prompt, for example
            because a parameter type is not supported.
    """
    by_name = {function.name: function for function in functions}
    context = model.encode(build_prompt(functions, prompt))
    context += model.encode(NAME_PREFIX)

    name, name_ids = choose_name(model, vocab, context, list(by_name))
    context += name_ids
    function = by_name[name]

    values: dict[str, Any] = {}
    fixed = '", "parameters": {'
    for index, (key, spec) in enumerate(function.parameters.items()):
        if index:
            fixed += ", "
        # Stop right after the colon: the tokenizer merges the space with
        # what follows (' -', ' true', ' "/'), so the model writes it.
        fixed += json.dumps(key, ensure_ascii=False) + ":"
        context += model.encode(fixed)
        value, value_ids = generate_value(model, vocab, context, spec.type)
        context += value_ids
        values[key] = value
        fixed = '"' if spec.type == "string" else ""

    call = FunctionCall(prompt=prompt, name=name, parameters=values)
    check_call(function, call)
    return call
