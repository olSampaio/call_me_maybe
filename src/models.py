"""Pydantic models used across the project."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

# Python-style spellings accepted in definition files, mapped to the
# JSON-schema type they stand for.
TYPE_ALIASES = {
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "str": "string",
}


class ParameterSpec(BaseModel):
    """Type description of a parameter or of a return value.

    Attributes:
        type: JSON type name, such as number or string.
    """

    type: str

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        """Lower-case the type and replace Python-style aliases.

        Args:
            value: The type as written in the file.

        Returns:
            The JSON-schema type name (for example "float" -> "number").
        """
        value = value.strip().lower()
        return TYPE_ALIASES.get(value, value)


class FunctionDefinition(BaseModel):
    """One callable function described in functions_definition.json.

    Attributes:
        name: Name of the function.
        description: What the function does.
        parameters: Parameter names mapped to their type, in the order they are
            written.
        returns: Type of the value the function returns.
    """

    name: str = Field(min_length=1)
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
