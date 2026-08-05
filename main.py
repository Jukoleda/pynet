# from tokenizer import Tokenizer
# from nn.embedding import Embedding

# with open("dataset.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# tokenizer = Tokenizer()

# tokenizer.fit(text)

# embbeding = Embedding (
#     vocab_size=len(tokenizer.word_to_id),
#     embedding_size=16
# )

# tokens = tokenizer.encode("hola david")

# print(tokens)

# for token in tokens:
#     print(tokenizer.id_to_word[token],
#           embbeding.get(token))

# print(tokenizer.decode(tokens))





# from neuron import Neuron

# brain = Neuron()

# learning_rate = 0.01

# for epoch in range(20):
#     error = brain.train(
#         value=10,
#         expected=7,
#         learning_rate=learning_rate
#     )

#     prediction = brain.predict(10)

#     print(
#         epoch,
#         "peso: ", round(brain.weight, 4),
#         "prediccion: ", round(prediction, 4),
#         "error: ", round(error, 4)
#     )

# print(brain.weight)
# print(brain.predict(10))


# from nn.sequential import Sequential

# module = Sequential([4,5,6,4])

# print(module.forward([0,0,1,1]))



from core.tensor import Tensor
from autograd.add import Add
from autograd.multiply import Multiply
from autograd.engine import AutogradEngine

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

c = Add().forward(a, b)

d = Multiply().forward(c, b)

AutogradEngine().print_graph(d)

