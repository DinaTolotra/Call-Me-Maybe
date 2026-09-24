from math import ceil
from functools import reduce

from src.llm_model import Model
from src.model import FnDef
import src.utils as utils
import src.fsm as fsm

BATCH_SIZE = 3

print(" + Initialising data...")
model = Model()
fn_list = FnDef.parse_json("data/input/functions_definition.json")
prompt = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
prompt_fmt = """\
Select the function that can perform the requested operation.

Functions:
{function_list}
F0; None of the function above

Request:
{instruction}

Answer:
"""


def select_function_in_batch(prompt: str, fn_list: list[FnDef]) -> tuple[FnDef, float]:
    fn_list_str = "\n".join(
        [f"F{i + 1}; {fn.description}" for i, fn in enumerate(fn_list)]
    )
    prompt = prompt_fmt.format(
        function_list = fn_list_str,
        instruction = prompt
    )

    print("  -\x1b[2m Initializing token...\x1b[0m")
    prompt_token = model.encode(prompt)
    fn_id_token = list[int]()
    ending_token = list[int]()
    for i in range(len(fn_list)):
        fn_id_token.extend(model.encode(f"F{i + 1}"))
    ending_token.extend(model.encode(";"))

    print("  -\x1b[2m Initializing FSM...\x1b[0m")
    table = fsm.TransitionTable()
    generating = fsm.State("generating")
    stoping = fsm.State("stoping")
    table.add_transition(
        generating,
        generating,
        set(fn_id_token)
    )
    table.add_transition(
        generating,
        stoping,
        set(ending_token)
    )

    print(" + Launching inference...")
    grammar = fsm.Grammar(table, generating)
    seq_prob = []
    while grammar.state != stoping:
        proba = utils.softmax(model.infer(prompt_token + grammar.result))
        sample = utils.index_sort(utils.filter_proba(proba, grammar.possible_token), 1)
        seq_prob.append(sample[0][1])
        grammar.consume(sample[0][0])
    id = model.decode(grammar.result).removeprefix("F")
    id = id.removesuffix(";")
    index = int(id) - 1
    return (
        fn_list[index],
        reduce(lambda a, b: a * b, seq_prob)
    )

for batch in utils.batched(fn_list, BATCH_SIZE):
    fn, proba = select_function_in_batch(prompt, batch)
    print(f" - {fn.name} ({proba:.8})")
