from math import ceil
from functools import reduce

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
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


def function_name_tournament(model: Model, request: Prompt, fn_list: list[FnDef], template_dict: dict[str, Template]) -> FnDef | None:
    result = fn_list
    while len(result) > 1:
        temp_result = []
        for fn_batch in utils.batched(result, BATCH_SIZE):
            fn = get_function_in_batch(
            model,
            request.prompt,
            template_dict["function_name"].value,
            fn_batch
        )
            if fn:
                temp_result.append(fn)
        result = temp_result

    fn = None
    if len(result) == 1:
        fn = result[0]
    return fn


def extract_argument(model: Model, request: Prompt, template_dict: dict[str, Template], fn: FnDef) -> dict[str, str]:
    arg_val = dict[str, str]()
    for arg in fn.parameters:
        prompt = template_dict["argument_value"].value.format(
            fn_name = fn.name, fn_description = fn.description,
            parameter_list = "\n".join([
                f"'{p}' type {fn.parameters[p]["type"]}" for p in fn.parameters
            ]),
            instruction = request.prompt,
            previous_arguments = "\n".join([
                f"'{a}' = {arg_val[a]}" for a in arg_val
            ]),
            param_name = arg
        )
        if fn.parameters[arg]["type"] == "number":
            grammar = fct.GrammarFactory.create_number_grammar(model)
        else:
            grammar = fct.GrammarFactory.create_string_grammar(model)
        prompt_token = model.encode(prompt)
        while not grammar.has_ended:
            logit = model.infer(prompt_token + grammar.result)
            proba = utils.filter_proba(utils.softmax(logit), grammar.possible_token)
            token, _ = utils.index_sort(proba, 1)[0]
            # print(f"   - Sample: {model.decode([token])} {_}")
            grammar.consume(token)
        arg_val[arg] = model.decode(grammar.result)
    return arg_val
        


model = Model()
request_list = Prompt.parse_json("data/input/function_calling_tests.json")
fn_list = FnDef.parse_json("data/input/functions_definition.json")
template_dict = Template.parse_json("data/template/prompt.json")


for request in request_list:
    fn = function_name_tournament(model, request, fn_list, template_dict)
    if fn:
        arg = extract_argument(model, request, template_dict, fn)
        print(" + Result:")
        print(f"   - Request: {request}")
        print(f"   - Function: {fn.name}")
        for arg_name, arg_val in arg.items():
            print(f"   - Argument\t{arg_name} = {arg_val}")