def usage() -> None:
    print(
        "Usage: uv run python -m src",
        "[-functions_definition <function_definition_file>]",
        "[-input <input_file>] [-output <output_file>]"
    )


def parse(argv: list[str]) -> tuple[str, str, str]:
    data: dict[str, str] = {
        "-input": "data/input/function_calling_tests.json",
        "-output": "data/output/function_calling_results.json",
        "-functions_definition": "data/input/functions_definition.json"
    }
    argv = argv[1:].copy()

    while len(argv) > 0:
        if argv[0] in data.keys():
            if len(argv) > 1:
                data[argv[0]] = argv[1]
                argv.pop(0)
            else:
                data[argv[0]] = ""
            argv.pop(0)
        else:
            break
    return (data["-input"], data["-output"], data["-functions_definition"])
