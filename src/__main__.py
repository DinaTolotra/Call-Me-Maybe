from time import perf_counter

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
from src.model import Result
import src.factory as fct
import src.utils as utils

BATCH_SIZE = 3


def get_function_in_batch(model: Model, request: str, prompt_template: str, fn_batch: list[FnDef]):
    fn_dict = {fn.name + ";": fn for fn in fn_batch}
    grammar = fct.GrammarFactory.create_fn_name_grammar(model, fn_batch)
    prompt = prompt_template.format(
        instruction = request,
        function_list = "\n".join([
            f"{fn.name}; {fn.description}" for fn in fn_batch
        ])
    )
    prompt_token = model.encode(prompt)
    proba_list = []
    while not grammar.has_ended:
        logit = model.infer(prompt_token + grammar.result)
        proba = utils.filter_proba(utils.softmax(logit), grammar.possible_token)
        token, _ = utils.index_sort(proba, 1)[0]
        grammar.consume(token)
        proba_list.append(_)
    fn = fn_dict.get(model.decode(grammar.result))
    return fn


def function_name_tournament(model: Model, request: Prompt, fn_list: list[FnDef], template: Template) -> FnDef | None:
    result = fn_list
    while len(result) > 1:
        temp_result = []
        for fn_batch in utils.batched(result, BATCH_SIZE):
            fn = get_function_in_batch(
            model,
            request.prompt,
            template.value,
            fn_batch
        )
            if fn:
                temp_result.append(fn)
        result = temp_result

    fn = None
    if len(result) == 1:
        fn = result[0]
    return fn


def extract_argument(model: Model, request: Prompt, template: Template, fn: FnDef) -> dict[str, str]:
    arg_val = dict[str, str]()
    for arg in fn.parameters:
        factory = fct.ArgExtractionPromptFactory(
            template.value, fn_name=fn.name, description=fn.description,
            param_name=arg, instruction=request,
            arg_val=arg_val, param_list=fn.parameters
        )
        if fn.parameters[arg]["type"] == "number":
            grammar = fct.GrammarFactory.create_number_grammar(model)
        else:
            grammar = fct.GrammarFactory.create_string_grammar(model)
        prompt_token = model.encode(factory.prompt)
        while not grammar.has_ended:
            logit = model.infer(prompt_token + grammar.result)
            proba = utils.filter_proba(utils.softmax(logit), grammar.possible_token)
            token, _ = utils.index_sort(proba, 1)[0]
            grammar.consume(token)
        arg_val[arg] = model.decode(grammar.result).removesuffix(";")
    return arg_val


def print_result(result: Result) -> None:
    print(f" + Result")
    print(f"   - Request: {request.prompt}")
    print(f"   - Function: {result.name}")
    for arg_name, arg_val in result.parameters.items():
        print(f"   - Argument\t{arg_name} = {arg_val}")


model = Model()
request_list = Prompt.parse_json("data/input/function_calling_tests.json")
fn_list = FnDef.parse_json("data/input/functions_definition.json")
template_dict = Template.parse_json("data/template/prompt.json")

duration = 0
for request in request_list:
    start = perf_counter()
    result = Result(
        prompt=request.prompt,
        parameters={},
        name=None
    )
    fn = function_name_tournament(model, request, fn_list, template_dict["function_name"])
    if fn is None:
        continue
    result.name = fn.name
    result.parameters = extract_argument(model, request, template_dict["argument_value"], fn)
    end = perf_counter()
    duration += end - start
    print_result(result)
    print(f"   - Duration {duration:.2f}s")
print(f" + Total {duration:.2f}s")
