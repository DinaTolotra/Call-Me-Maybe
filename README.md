*This project has been created as part of the 42 curriculum by todina-r.*

# Call Me Maybe

## Description

Call Me Maybe converts natural-language requests into structured function calls. It uses the Qwen/Qwen3-0.6B language model to select a function and extract its arguments, then writes the results as JSON. It does not execute the selected functions.

The available functions and their parameter types are supplied at runtime in a JSON definition file. Requests are also read from JSON, allowing the same implementation to work with different function sets and prompts.

## Features

- Function selection using model logits and constrained token generation.
- Argument extraction for strings, numbers, and booleans.
- Pydantic validation for input definitions, prompts, templates, and result objects.
- A configurable command-line interface for input, function definitions, and output paths.
- A 128-token generation limit per grammar, with incomplete generations reported as request errors.

## Instructions

### Requirements

- Python 3.13 or later, as declared in `pyproject.toml`.
- `uv` for dependency management.
- Network access on the first model load, unless the model is already cached by Hugging Face.

### Installation

```sh
uv sync
```

### Run

The intended command is:

```sh
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

With no options, the program is intended to use those same input files and output path by default.

**Known CLI issue:** The current `src/parser.py` accepts the double-dash options shown above, but its return statement looks up obsolete single-dash keys. As a result, the command currently fails before processing input. This needs to be corrected before the run examples can be verified end to end. The Makefile's `run` target also has an empty `FN_DEF_FILE` value and is not currently a reliable alternative.

### Make targets

- `make install` installs Python 3.13 and synchronizes the project environment with `uv`.
- `make lint` runs flake8 and the configured mypy checks.
- `make lint-strict` runs flake8 and strict mypy checks.
- `make clean` removes Python caches and build artifacts.

## Input and Output

The request input is a JSON array of objects containing a `prompt` string:

```json
[
	{"prompt": "What is the sum of 40 and 2?"},
	{"prompt": "Greet Ada"}
]
```

The function definition file is a JSON array. Each entry supplies a function `name`, `description`, `parameters`, and `returns`. Supported parameter types are `string`, `number`, and `boolean`.

The intended result format is a JSON array with one object per successfully selected function:

```json
[
	{
		"prompt": "What is the sum of 40 and 2?",
		"name": "fn_add_numbers",
		"parameters": {"a": 40.0, "b": 2.0}
	}
]
```

Results are serialized through Pydantic. Currently, requests with no selected function or a generation error are omitted from the result array; therefore the output is not guaranteed to have one entry per input request.

## Algorithm

1. The input JSON is validated into Pydantic models.
2. The function selector formats the request and a batch of function descriptions into a prompt. It compares functions through batches of up to three, repeating until one candidate remains.
3. For each token, the model produces logits. The implementation normalizes these scores, masks tokens not allowed by the current grammar, and chooses the highest-scoring remaining token.
4. The selected function's parameters are extracted one at a time. Each parameter uses a grammar based on its declared type.
5. The extracted name and typed parameters are validated and serialized as JSON.

The grammar is a finite-state machine: states represent progress through a value, and transitions specify allowed token IDs. Number generation supports digits, an optional leading minus sign, and decimal fractions with at least one digit after the decimal point. Boolean generation follows the tokenized paths for the JSON literals `true;` and `false;`. String generation excludes quote and newline tokens, but does not implement full JSON string escaping.

The output object is serialized as JSON after extraction; the model is not asked to generate the entire result object as JSON. The grammar's token limit bounds each generation, but a limit reached before an accepting state is treated as an error rather than a valid partial value.

## Design Decisions

- Function definitions and prompts are read from files rather than hardcoded into the selection logic.
- Pydantic models reject malformed or unexpected input fields and constrain collection sizes.
- Separate grammars keep value-generation rules local to each supported parameter type.
- The output is written through a temporary file before replacement to avoid leaving a partially written result file.
- `Small_LLM_Model` is wrapped by `src/llm_model.py`; the default model is Qwen/Qwen3-0.6B.

## Performance and Reliability

No accuracy or end-to-end latency benchmark has been recorded, so the assignment's 90% accuracy and five-minute targets are not yet demonstrated. Generation makes a model inference for each token and reprocesses the prompt plus generated prefix; there is no KV-cache or batching optimization in the application code.

Pydantic serialization ensures the output file is valid JSON when writing succeeds. This does not guarantee that function selection or argument extraction is semantically correct. In particular, function-name constraints currently combine candidate token IDs rather than tracking each candidate name's full token path, and string generation does not escape JSON special characters. The CLI issue above also prevents end-to-end verification until fixed.

## Challenges

- Tokenizers can represent a visible value as multiple token IDs, so a grammar must track token-by-token progress rather than accept a set of tokens as a complete value.
- A grammar can constrain structure and type shape, but it cannot ensure that an extracted value matches the user's intent.
- Constrained generation needs an explicit token cap: if no valid accepting path is reached in time, the request must fail instead of returning a partial value.
- The current function-name and string grammars still have limitations; they should be strengthened and tested with the actual Qwen tokenizer.

## Testing Strategy

The numeric grammar was checked with a lightweight tokenizer stub for positive and negative integers and decimals, and for malformed forms such as `-.4`, `4.`, and `1..2`. The boolean token paths were similarly checked for complete lowercase literals, incomplete prefixes, and rejected alternatives. These checks are manual and are not currently included as a checked-in automated test suite.

Before submission, run the lint targets, install dependencies with `uv sync`, verify the CLI after fixing its parser mismatch, and test with the real model and varied function definitions. Include ambiguous prompts, empty and escaped strings, negative and decimal numbers, booleans, malformed inputs, and generation-limit cases.

## Resources

- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Python `json` module](https://docs.python.org/3/library/json.html)
- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- The project subject, `en.subject.pdf`, defines the expected behavior and evaluation requirements.

### AI Usage

GitHub Copilot was used in this development session to review the project against the subject and identify implementation issues.
