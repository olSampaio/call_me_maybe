"""Building the text the model reads before it writes the function call."""

from collections.abc import Sequence

from .models import FunctionDefinition

# The model continues from here: the name is chosen by the model, the
# rest of the JSON is written under constraints.
NAME_PREFIX = '{"name": "'


def describe_function(function: FunctionDefinition) -> str:
    """Return a one-line description of a function and its parameters.

    Args:
        function: The function to describe.

    Returns:
        A line with the name, the typed parameters and the description.
    """
    params = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    return f"- {function.name}({params}): {function.description}"


def build_prompt(functions: Sequence[FunctionDefinition], request: str) -> str:
    """Build the instruction, the function list and the user request.

    Args:
        functions: The functions the model may choose from.
        request: The natural-language request.

    Returns:
        Text that ends right before the JSON answer starts.
    """
    listing = "\n".join(describe_function(f) for f in functions)
    return (
        "You are a function-calling engine. Read the request, choose the "
        "single best function and copy its argument values from the "
        "request exactly as written.\n\n"
        f"Functions:\n{listing}\n\n"
        f"Request: {request}\n"
        "Function call as JSON:\n"
    )
