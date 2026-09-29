from math import ceil
from functools import reduce

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
import src.factory as fct
import src.utils as utils
import src.fsm as fsm

BATCH_SIZE = 3
REQUEST_INDEX = 1


def get_function_in_batch(model: Model, request: str, prompt_template: str, fn_batch: list[FnDef]):
    fn_dict = {fn.name + ";": fn for fn in fn_batch}
    state = {
        "generating": fsm.State("generating"),
        "stoping": fsm.State("stoping")
    }
    grammar = fct.GrammarFactory.create(model, fn_batch)
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


print(" + Initialising data...")
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
if len(result) == 1:
    print(f" + Result: {result[0].name}")
elif len(result) == 0:
    print(" + Result: None")
else:
    print(" + Result: ???")
