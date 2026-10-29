def usage() -> str:
    return " ".join((
        "Usage: uv run python -m src",
        "[-functions_definition <function_definition_file>]",
        "[-input <input_file>] [-output <output_file>]"
    ))


def parse(argv: list[str]) -> tuple[str, str, str]:
    data: dict[str, str] = {
        "-input": "data/input/function_calling_tests.json",
        "-output": "data/output/function_calling_results.json",
        "-functions_definition": "data/input/functions_definition.json"
    }
    argv = argv[1:].copy()

    while len(argv) > 0:
        option = argv.pop(0)
        if option not in data:
            raise ValueError(f"Unknown option: {option}")
        if not argv:
            raise ValueError(f"Missing value for option: {option}")
        data[option] = argv.pop(0)
    return (data["-input"], data["-output"], data["-functions_definition"])
