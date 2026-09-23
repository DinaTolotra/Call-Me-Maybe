from math import ceil
from functools import reduce

from src.llm_model import Model
from src.utils import softmax, filter_proba, index_sort
from src.model import FnDef
import src.fsm as fsm

print(" + Initialising data...")
model = Model()
fn_list = FnDef.parse_json("data/input/functions_definition.json")
prompt_fmt = """\
Select the function that can perform the requested operation.

Functions:
{function_list}
F0; None of the function above

Request:
{instruction}

Answer:
"""


def generate_sequence(model: Model, fn_list: list[FnDef]) -> tuple[str, float]:
    prompt = prompt_fmt.format(
        function_list = "\n".join([f"F{i + 1}; {fn.description}" for i, fn in enumerate(fn_list)]),
        instruction = "Replace the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"
    )

    print("  -\x1b[2m Initializing token...\x1b[0m")
    prompt_token = model.encode(prompt)
    fn_name_token = list[int]()
    ending_token = list[int]()
    for i in range(len(fn_list)):
        fn_name_token.extend(model.encode(f"F{i + 1}"))
    ending_token.extend(model.encode(";"))

    print("  -\x1b[2m Initializing FSM...\x1b[0m")
    table = fsm.TransitionTable()
    generating = fsm.State("generating")
    stoping = fsm.State("stoping")
    table.add_transition(
        generating,
        generating,
        set(fn_name_token)
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
        proba = softmax(model.infer(prompt_token + grammar.result))
        sample = index_sort(filter_proba(proba, grammar.possible_token), 1)
        seq_prob.append(sample[0][1])
        grammar.consume(sample[0][0])
    id = model.decode(grammar.result).removeprefix("F")
    id = id.removesuffix(";")
    return (
        fn_list[int(id) - 1].name,
        reduce(lambda a, b: a * b, seq_prob)
    )

for i in range(ceil(len(fn_list) / 3)):
    beg, end = (i * 3, min((i + 1) * 3, len(fn_list)))
    seq, proba = generate_sequence(model, fn_list[beg:end])
    print(f" - {seq} ({proba:.8})")
