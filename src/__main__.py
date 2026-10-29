from time import perf_counter
from sys import argv

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
from src.model import Result
from src.parser import parse, usage
import src.factory as fct
import src.utils as utils
import src.logging as logging

BATCH_SIZE = 3


def get_function_in_batch(
        model: Model,
        request: str,
        template: str,
        fn_batch: list[FnDef]
        ) -> FnDef | None:
    fn_dict = {fn.name + ";": fn for fn in fn_batch}
    grammar = fct.GrammarFactory.create_fn_name_grammar(model, fn_batch)
    factory = fct.FunctionPromptFactory(template, request, fn_batch)
    prompt_token = model.encode(factory.prompt)
    while not grammar.has_ended:
        logit = model.infer(prompt_token + grammar.result)
        proba = utils.filter_proba(utils.softmax(logit), grammar.token)
        token, _ = utils.index_sort(proba, 1)[0]
        grammar.consume(token)
    fn = fn_dict.get(model.decode(grammar.result))
    return fn


def function_name_tournament(
        model: Model,
        request: Prompt,
        fn_list: list[FnDef],
        template: Template
        ) -> FnDef | None:
    result = list(fn_list)
    while len(result) > 1:
        temp_result: list[FnDef] = []
        for fn_batch in utils.batched(result, BATCH_SIZE):
            fn = get_function_in_batch(
                model, request.prompt,
                template.value, fn_batch,
            )
            if fn is not None:
                temp_result.append(fn)
        result = temp_result

    if len(result) == 1:
        return result[0]
    return None


def extract_argument(
        model: Model,
        request: Prompt,
        fn: FnDef,
        template: Template
        ) -> dict[str, str | float]:
    arg_val: dict[str, str | float] = {}
    val: str | float = ""
    for arg in fn.parameters:
        factory = fct.ArgExtractionPromptFactory(
            template.value, fn_name=fn.name,
            description=fn.description,
            param_name=arg, instruction=request,
            arg_val=arg_val, param_list=fn.parameters,
        )
        if fn.parameters[arg]["type"] == "number":
            grammar = fct.GrammarFactory.create_number_grammar(model)
        elif fn.parameters[arg]["type"] == "boolean":
            grammar = fct.GrammarFactory.create_boolean_grammar(model)
        else:
            grammar = fct.GrammarFactory.create_string_grammar(model)
        prompt_token = model.encode(factory.prompt)
        while not grammar.has_ended:
            logit = model.infer(prompt_token + grammar.result)
            proba = utils.filter_proba(utils.softmax(logit), grammar.token)
            token, _ = utils.index_sort(proba, 1)[0]
            grammar.consume(token)
        val = utils.clean(model.decode(grammar.result))
        if fn.parameters[arg]["type"] == "number":
            val = float(val)
        arg_val[arg] = val
    return arg_val


def run_request(
        model: Model,
        request: Prompt,
        fn_list: list[FnDef],
        templates: dict[str, Template]
        ) -> Result:
    result = Result(
        prompt=request.prompt,
        parameters={},
        name=None,
    )
    fn = function_name_tournament(
        model, request, fn_list,
        templates["function_name"]
    )
    if fn is None:
        return result
    result.name = fn.name
    result.parameters = extract_argument(
        model, request, fn,
        templates["argument_value"]
    )
    return result


def main() -> int:
    try:
        fn_def_file, input_file, output_file = parse(argv)
    except ValueError as error:
        logging.print_error("Invalid command line", error)
        logging.print_usage(usage())
        return 2

    logging.print_arg(fn_def_file, input_file, output_file)
    if not utils.check_input_file(fn_def_file, input_file):
        logging.print_error("Function definitions or request input is unreadable")
        return 2
    if not utils.check_output_file(output_file, (fn_def_file, input_file)):
        logging.print_error("Output path is not writable or conflicts with an input")
        return 2

    try:
        request_list = Prompt.parse_json(input_file)
        fn_list = FnDef.parse_json(fn_def_file)
        template_dict = Template.parse_json("data/template/prompt.json")
    except (OSError, ValueError) as error:
        logging.print_error("Could not read or validate JSON data", error)
        return 2

    if not {"function_name", "argument_value"}.issubset(template_dict):
        logging.print_error("Prompt template is missing required entries")
        return 2
    if not fn_list or not request_list:
        logging.print_error("Function definitions and requests must not be empty")
        return 2

    try:
        model = Model()
    except Exception as error:
        logging.print_error("Could not initialize the model", error)
        return 1

    result_list: list[Result] = []
    total_duration = 0.0
    skipped = 0
    failed = 0
    for index, request in enumerate(request_list, start=1):
        start = perf_counter()
        try:
            result = run_request(model, request, fn_list, template_dict)
            elapsed = perf_counter() - start
            total_duration += elapsed
            logging.print_request_duration(index, elapsed)
            if result.name is None:
                skipped += 1
                continue
            logging.print_result(result)
            result_list.append(result)
        except Exception as error:
            elapsed = perf_counter() - start
            total_duration += elapsed
            failed += 1
            logging.print_error(f"Request {index} failed", error)

    logging.print_summary(
        len(result_list), skipped, failed, total_duration
    )
    try:
        Result.save_list(result_list, output_file)
    except (OSError, ValueError) as error:
        logging.print_error("Could not save result file", error)
        return 1
    logging.print_output_saved(output_file)
    return 1 if failed else 0


if __name__ == "__main__":
    start = perf_counter()
    exit_code = main()
    duration = perf_counter() - start
    logging.print_program_duration(duration)
    raise SystemExit(exit_code)
