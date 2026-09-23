from string import digits

from src.llm_model import Model
from src.utils import softmax, filter_proba, index_sort
from src.model import FnDef
import src.fsm as fsm

print(" + Initialising data...")
model = Model()
fn_list = FnDef.parse_json("data/input/functions_definition.json")
prompt = """\

instruction:
{instruction}

function list:
{function_list}

output example:
fn_print_hello;

corresponding function:

"""
prompt = prompt.format(
	function_list = "\n".join([str(fn) for fn in fn_list]),
	instruction = "Calculate the square root of 144"
)

print("  - Initializing token...")
prompt_token = model.encode(prompt)
fn_name_token = list[int]()
ending_token = list[int]()
for fn in fn_list:
	token = model.encode(fn.name)
	fn_name_token.extend(token)
ending_token.extend(model.encode(";"))

print("  - Initializing FSM...")
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
while grammar.state != stoping:
	proba = softmax(model.infer(prompt_token + grammar.result))
	sample = index_sort(filter_proba(proba, grammar.possible_token), 1)
	print(f"  - Sample: [{model.decode([sample[0][0]])}]")
	grammar.consume(sample[0][0])
print(model.decode(grammar.result))
