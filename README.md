*This project has been created as part of the 42 curriculum by <login>.*

# call me maybe

Function calling with constrained decoding on a small language model.

## Description

Given a natural-language request such as `What is the sum of 40 and 2?`,
the program does **not** answer it. It returns the function that should be
called and its arguments, as strict JSON:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The model is `Qwen/Qwen3-0.6B`, which is too small to write valid JSON
reliably when it is only asked to. The output is therefore produced with
**constrained decoding**: at every step the logits of all tokens that would
break the JSON or the schema are set to minus infinity, so the model can
only choose among valid tokens. The result is always valid and always
matches `functions_definition.json`, whatever the model prefers.

The function to call and the value of every argument are chosen by the
model. No heuristic, regular expression or keyword rule picks them.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
make run            # uv run python -m src
make debug          # run under pdb
make lint           # flake8 + mypy (flags required by the subject)
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches
```

The model (about 1.5 GB) is downloaded from Hugging Face the first time the
program runs. PyTorch is installed in its CPU-only build (see
`pyproject.toml`), which is enough for a 0.6B model.

Command line:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

All three options are optional and default to the paths above.

### Example usage

```bash
$ uv run python -m src
[1/5] 'What is the sum of 2 and 3?' -> fn_add_numbers({'a': 2.0, 'b': 3.0})
[2/5] 'Greet shrek' -> fn_greet({'name': 'shrek'})
...
Done: 5 calls written to data/output/function_calls.json (0 failed) in ...s.
```

The exit code is 0 when every prompt was processed and 1 otherwise. Missing
or invalid input files, a model that cannot be loaded and an unwritable
output file are reported with a one-line message on stderr, never with a
traceback. If a single prompt fails, the others are still processed and
written.

## Algorithm explanation

### Pieces of the JSON

The answer is written in pieces. The fixed pieces (`{"name": "`, the keys,
the punctuation) are encoded and appended to the context directly. The
model is only asked to decide:

1. which function to call,
2. the value of each parameter, in the order of the definition.

The context given to the model is a short instruction, the list of
functions with their parameter types and descriptions, the request, and
then the beginning of the JSON answer. The `prompt` field of the output is
copied from the input, never generated.

### The masking step

For every decision the code builds the set of allowed token ids, asks the
model for the logits of the next token, sets every other logit to `-inf`
and takes the best of what is left (`decoding.mask_logits`,
`decoding.pick_allowed`). Greedy selection keeps the output deterministic.

### Token texts

The vocabulary file stores byte-level BPE tokens, where each byte is written
as a printable character (a space is `Ġ`, a newline is `Ċ`).
`vocabulary.token_to_text` undoes that mapping so each token id has the real
text it writes. Tokens that are only a fragment of a multi-byte character
have no text of their own and are never used.

### Function name

At each step the allowed tokens are those whose text is a prefix of what is
still missing from at least one valid name. This accepts every tokenization
of a name (`fn` + `_add`, or `f` + `n` + `_`...). If a name is complete but
is also the beginning of a longer one, the closing quote becomes one more
option. When only one name remains, the rest of it is written without
calling the model again, because the choice has already been made.

### Numbers

A number is generated token by token. A token is allowed only if the text
written so far plus the token is still a prefix of a valid number
(`-?(0|[1-9][0-9]*)(\.[0-9]+)?`, so no leading zeros and one decimal point at
most). Once the number is complete, any token that starts with `,` or `}` is
also allowed, and choosing one of them means "the number is over". That
token is not kept: the code writes the real punctuation itself. Numbers are
limited to 30 characters. `integer` parameters use the same mechanism
without the decimal point.

### Strings

Inside a string, every token is allowed except those with a raw control
character or an incomplete escape sequence (a lone backslash, for example).
Tokens that contain the closing quote end the string: only the text before
the quote is kept. The text is then decoded with `json.loads`, so `\"`,
`\\` and `\n` produce the right characters. A string is cut after 100 tokens
if the model never closes it.

### Booleans

The model chooses between `true` and `false` with the same prefix mechanism
used for function names.

### Final check

Before a call is accepted, `pipeline.check_call` verifies that the name, the
keys, their order and the value types match the definition. The output file
is written with `allow_nan=False`, so it is always strict JSON.

## Design decisions

- **Model behind a small interface.** The generation code depends only on a
  protocol with `encode` and `get_logits` (`model_api.py`). The adapter for
  the provided SDK (`llm.py`) uses only its public methods.
- **`llm_sdk` imported lazily.** `importlib` is used inside the loader, so
  the rest of the project works without torch installed.
- **Pydantic for the data.** Definitions, prompts, outputs, the vocabulary
  and the SDK adapter are pydantic models. Input files are validated with
  `TypeAdapter`.
- **One exception per concern.** `FileError`, `ModelLoadError` and
  `GenerationError` are caught in `__main__` and turned into messages.
- **Fixed text is not generated.** Keys and punctuation are never sampled,
  which removes whole classes of errors and saves model calls.
- **Greedy selection.** Reproducible output, which is what we want here.
- **CPU-only PyTorch.** The evaluation machines may not have a GPU, so the
  project is developed for the worst case. `Small_LLM_Model` picks the CPU
  by itself when there is no GPU.
- **Limits.** 30 characters for numbers and 100 tokens for strings
  guarantee that generation always ends.

Known limitations: only the types `number`, `integer`, `string` and
`boolean` are generated (anything else is reported as an error for that
prompt); `\uXXXX` escapes are not produced (the model can write the
character itself); characters that exist in the vocabulary only as pieces of
a multi-byte sequence cannot appear in strings.

## Performance analysis

What follows from the design:

- A valid output does not depend on the model: tokens that would break the
  JSON or the schema are masked before every choice, so whatever the logits
  are, the result is valid.
- Cost is the number of model calls per prompt: 0 to a few for the name,
  one per digit plus one for each number, one per token plus one for each
  string, one or two per boolean. The SDK recomputes the whole sequence on
  every call, so the length of the prompt matters.

Measured on the evaluation hardware (fill in after running `make run`):

| Item | Result |
|---|---|
| Machine (CPU, RAM) | TODO |
| Model load time | TODO |
| Time for all prompts of `function_calling_tests.json` | TODO |
| Time per prompt (average) | TODO |
| Correct function chosen | TODO / TODO |
| Correct arguments extracted | TODO / TODO |
| Valid JSON | TODO / TODO |

## Challenges faced

- **Memory while loading the model.** On a development machine with little
  RAM the model load froze the computer: on CPU the SDK loads the weights
  in `float32`, which needs several GB. The decoder was first exercised
  with a fake model and then run with the real one on a machine with enough
  memory.
- **Tokens are not text.** The vocabulary writes bytes as printable
  characters, so a token has to be translated before its text can be
  compared with a function name or a number.
- **Names that start other names.** A name can be both complete and the
  beginning of a longer name, so the closing quote has to be an explicit
  choice.
- **Knowing when a number ends.** The model has to be able to stop, but
  tokens such as `,` are merged with others in the vocabulary, so any token
  that starts with the delimiter is accepted as a stop signal.
- **Tokens that mix content and the closing quote** (for example `o",`):
  only the part before the quote is kept.
- **The SDK layout.** The provided folder is a complete uv project, so it is
  declared as a local dependency instead of being flattened.

## Testing strategy

TODO: describe what you actually ran. Suggested content, to be adjusted to
what was done:

- **Provided examples.** Run `make run` on the files in `data/input/` and
  check `data/output/function_calls.json`: valid JSON, exactly the keys
  `prompt`, `name` and `parameters`, the right function and the right types.
- **Edge cases listed in the subject.** Empty strings, large numbers,
  special characters, ambiguous prompts and functions with several
  parameters, using other `--functions_definition` and `--input` files.
- **Error handling.** Missing file, invalid JSON, wrong structure, a
  directory instead of a file, a file that is not UTF-8 and an output path
  that cannot be written must end with a one-line message on stderr and
  exit code 1, never with a traceback.
- **Static checks.** `make lint` and `make lint-strict`.

## Resources

- Willard & Louf, *Efficient Guided Generation for Large Language Models*
  (2023), https://arxiv.org/abs/2307.09702
- Hugging Face, *How to generate text* and the tokenizers summary,
  https://huggingface.co/docs/transformers
- Qwen3-0.6B model card, https://huggingface.co/Qwen/Qwen3-0.6B
- Andrej Karpathy, *Let's build the GPT Tokenizer* (byte-level BPE)
- JSON specification, https://www.json.org and RFC 8259
- Pydantic, https://docs.pydantic.dev; uv, https://docs.astral.sh/uv/

### Use of AI

TODO: adjust this paragraph so that it is exactly true for you. Draft:
Claude (Anthropic) was used to explain the concepts (tokens, logits,
constrained decoding), to plan the architecture, to draft the modules and
to review them. The code was read, run and tested, and every part of it can
be explained by the author.