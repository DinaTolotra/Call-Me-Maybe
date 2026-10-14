from src.model import Result


def print_result(result: Result) -> None:
    print(" + Result")
    print(f"   - Request\t {result.prompt}")
    print(f"   - Function\t {result.name}")
    for arg_name, arg_val in result.parameters.items():
        print(f"   - Argument\t {arg_name} = {arg_val}")


def print_arg(fn_def_file: str, input_file: str, output_file: str) -> None:
    print(" + Data")
    print("   - Function definition file:", fn_def_file)
    print("   - Input file:", input_file)
    print("   - Output file:", output_file)
