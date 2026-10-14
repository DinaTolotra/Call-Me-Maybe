from time import perf_counter

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
from src.model import Result
import src.factory as fct
import src.utils as utils

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


def print_result(result: Result) -> None:
    print(" + Result")
    print(f"   - Request\t {result.prompt}")
    print(f"   - Function\t {result.name}")
    for arg_name, arg_val in result.parameters.items():
        print(f"   - Argument\t {arg_name} = {arg_val}")


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


def main() -> None:
    model = Model()
    request_list = Prompt.parse_json("data/input/function_calling_tests.json")
    fn_list = FnDef.parse_json("data/input/functions_definition.json")
    template_dict = Template.parse_json("data/template/prompt.json")
    output_file = "data/output/function_calling_results.json"
    result_list: list[Result] = []

    total_duration = 0.0
    for request in request_list:
        start = perf_counter()
        result = run_request(model, request, fn_list, template_dict)
        elapsed = perf_counter() - start
        total_duration += elapsed

        if result.name is None:
            continue

        print_result(result)
        result_list.append(result)
        print(f"   - Duration\t {elapsed:.2f}s")
    print(f" + Total {total_duration:.2f}s")
    Result.save_list(result_list, output_file)


if __name__ == "__main__":
    main()
