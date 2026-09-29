from math import ceil
from functools import reduce

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
import src.factory as fct
import src.utils as utils
import src.fsm as fsm

BATCH_SIZE = 2
REQUEST_INDEX = 2


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
        print(f"   - Sample: {model.decode([token])} {_}")
        grammar.consume(token)
        proba_list.append(_)
    fn = fn_dict.get(model.decode(grammar.result))
    return fn


model = Model()
request_list = Prompt.parse_json("data/input/function_calling_tests.json")
fn_list = FnDef.parse_json("data/input/functions_definition.json")
template_dict = Template.parse_json("data/template/prompt.json")

result = fn_list
while len(result) > 1:
    temp_result = []
    for fn_batch in utils.batched(result, BATCH_SIZE):
        fn = get_function_in_batch(
            model,
            request_list[REQUEST_INDEX].prompt,
            template_dict["function_name"].value,
            fn_batch
        )
        if fn:
            temp_result.append(fn)
    result = temp_result

fn = None
if len(result) == 1:
    print(f" + Result: {result[0].name}")
    fn = result[0]
elif len(result) == 0:
    print(" + Result: None")
else:
    print(" + Result: ???")

if fn:
    arg_val = dict[str, str]()
    for arg in fn.parameters:
        prompt = template_dict["argument_value"].value.format(
            fn_name = fn.name, fn_description = fn.description,
            parameter_list = "\n".join([
                f"'{p}' type {fn.parameters[p]["type"]}" for p in fn.parameters
            ]),
            instruction = request_list[REQUEST_INDEX].prompt,
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
        proba_list = []
        while not grammar.has_ended:
            logit = model.infer(prompt_token + grammar.result)
            proba = utils.filter_proba(utils.softmax(logit), grammar.possible_token)
            token, _ = utils.index_sort(proba, 1)[0]
            print(f"   - Sample: {model.decode([token])} {_}")
            grammar.consume(token)
            proba_list.append(_)
        print(f" + Result: {model.decode(grammar.result)}")