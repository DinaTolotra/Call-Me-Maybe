import json
from sys import stderr

from src.model import Result


def _format_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def print_error(message: str, error: Exception | None = None) -> None:
    detail = ""
    if error is not None:
        error_text = str(error)
        if len(error_text) > 500:
            error_text = f"{error_text[:497]}..."
        detail = f" [{type(error).__name__}] {_format_value(error_text)}"
    print(f"Error: {message}{detail}", file=stderr)


def print_usage(message: str) -> None:
    print(message, file=stderr)


def print_result(result: Result) -> None:
    print("\nResult")
    print(f"  Request:  {_format_value(result.prompt)}")
    print(f"  Function: {_format_value(result.name)}")
    for arg_name, arg_val in result.parameters.items():
        print(f"  Argument: {_format_value(arg_name)} = {_format_value(arg_val)}")


def print_arg(fn_def_file: str, input_file: str, output_file: str) -> None:
    print("Input / Output")
    print(f"  Definitions: {_format_value(fn_def_file)}")
    print(f"  Requests:    {_format_value(input_file)}")
    print(f"  Results:     {_format_value(output_file)}")


def print_request_duration(index: int, duration: float) -> None:
    print(f"  Request {index}: {duration:.2f}s")


def print_summary(
        succeeded: int,
        skipped: int,
        failed: int,
        inference_duration: float
        ) -> None:
    print("\nSummary")
    print(f"  Succeeded: {succeeded}")
    print(f"  No match:  {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Inference: {inference_duration:.2f}s")


def print_program_duration(duration: float) -> None:
    print(f"Total time: {duration:.2f}s")


def print_output_saved(path: str) -> None:
    print(f"Results saved to {_format_value(path)}")
