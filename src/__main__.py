from .llm_model import Model
from .utils import softmax
from .utils import index_sort
from .utils import filter_proba
from .model import FnDef, Prompt
from .tree import Node, print_tree


def infer(
        model: Model,
        prompt_token: list[int],
        allowed_token: set[int]
        ) -> tuple[int, float]:
    proba = softmax(model.infer(prompt_token))
    filtered_proba = filter_proba(proba, set(allowed_token))
    sample = index_sort(filtered_proba, 1)
    return sample[0]


def generate_path(root: Node, data: list[int]) -> None:
    if len(data) == 0:
        return
    token = data[0]

    node = root[token]
    if not root.has(token):
        node = root.add_child(token)
    generate_path(node, data[1:])


def generate_descision_tree(
        model: Model,
        choice: list[str]
        ) -> Node:
    root = Node(-1)
    for opt in choice:
        opt_token = model.encode(opt)
        generate_path(root, opt_token)
    return root


model = Model()

print(" + Data initialisation...")
prompt = "How a programmer greet? Hello "
prompt_token = model.encode(prompt)
allowed_word = ["world", "everybody", "everyone", "fucker"]
descision_tree = generate_descision_tree(model, allowed_word)

print(f"  - Prompt: [{prompt}] = {prompt_token}")

print(" + Running the model...")
node = descision_tree
result = list[int]()
while node.get_children_count() > 0:
    allowed_token = node.get_token_option()
    token, proba = infer(model, prompt_token + result, allowed_token)
    result.append(token)
    node = node[token]

print(f"  - Result: [{prompt}{model.decode(result)}]")
