from math import ceil
from functools import reduce

from src.llm_model import Model
from src.model import Template
from src.model import Prompt
from src.model import FnDef
import src.utils as utils
import src.fsm as fsm

BATCH_SIZE = 2
REQUEST_INDEX = 11


def get_function_in_batch(model: Model, request: str, prompt_template: str, fn_batch):
    fn_dict = dict[str, FnDef]()
    fn_id_token_set = {*model.encode(f"0")}
    for i in range(1, len(fn_batch) + 1):
        fn_dict[f"{i};"] = fn_batch[i - 1]
        fn_id_token_set = fn_id_token_set.union(model.encode(f"{i}"))
    state = {
        "generating": fsm.State("generating"),
        "stoping": fsm.State("stoping")
    }
    table = fsm.TransitionTable()
    grammar = fsm.Grammar(table, state["generating"])
    table.add_transition(
        state["generating"],
        state["generating"],
        fn_id_token_set
    )
    table.add_transition(
        state["generating"],
        state["stoping"],
        set(model.encode(";"))
    )
    prompt = prompt_template.format(
        instruction = request,
        function_list = "\n".join([
            f"F{i}; {str(fn)}" for i, fn in enumerate(fn_batch, 1)
        ])
    )
    prompt_token = model.encode(prompt)
    proba_list = []
    while grammar.state != state["stoping"]:
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
