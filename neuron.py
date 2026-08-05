import random

class Neuron:
    def __init__(self, inputs_size):
        self.weights = []

        for _ in range(inputs_size):
            self.weights.append(random.uniform(-1, 1))

        self.bias = random.uniform(-1, 1)


    def forward(self, inputs):

        if len(inputs) != len(self.weights):
            raise ValueError("Inputs size mismatch")

        self.last_inputs = inputs
        output = self.bias

        for value, weight in zip(inputs, self.weights):
            output += value * weight

        # for i in range(len(inputs)):
        #     output += inputs[i] * self.weights[i]

        self.last_output = output

        return output