from .model import Model
from .utils import softmax
from .utils import index_sort

model: Model = Model()
prompt = ""
output = ""

while True:
	input_token = model.encode(prompt + output)
	logits = model.infer(input_token)
	output_token = index_sort(softmax(logits), 1)
	decoded = model.decode([output_token[0][0]])
	if all(c not in [" ", "\n"] for c in decoded):
		output += decoded
	else:
		break
print(prompt + output)
