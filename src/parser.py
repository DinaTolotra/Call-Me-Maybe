def parse(argv: list[str]) -> tuple[str, str, str]:
    fn_def_file = "data/input/functions_definition.json"
    input_file = "data/input/function_calling_tests.json"
    output_file = "data/output/function_calling_results.json"

    i = 1
    while len(argv) < 7 and i < len(argv) - 1:
        if argv[i] == "-functions_definition":
            fn_def_file = argv[i + 1]
            i += 2
        elif argv[i] == "-input":
            input_file = argv[i + 1]
            i += 2
        elif argv[i] == "-output":
            output_file = argv[i + 1]
            i += 2
        break
    return (fn_def_file, input_file, output_file)
