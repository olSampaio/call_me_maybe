"""Pydantic models used across the project."""

from typing import Any

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """Type description of a parameter or of a return value.

    Attributes:
        type: JSON type name, such as number or string.
    """

    type: str


class FunctionDefinition(BaseModel):
    """One callable function described in functions_definition.json.

    Attributes:
        name: Name of the function.
        description: What the function does.
        parameters: Parameter names mapped to their type, in the order they are
            written.
        returns: Type of the value the function returns.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptItem(BaseModel):
    """One natural-language request from the tests file.

    Attributes:
        prompt: The request, as written by the user.
    """

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file.

    Attributes:
        prompt: The original request.
        name: Name of the function to call.
        parameters: Argument names mapped to their values.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
