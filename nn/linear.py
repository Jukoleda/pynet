import random

from legacy.linalg.vector import dot

from core.module import Module
from core.parameter import Parameter

class Linear(Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.weight_matrix = []
        self.bias = []
        self.last_output = None
        self.last_input = None
        self.input_size = input_size
        self.output_size = output_size

        self.initialize_parameters()


    def forward(self, vector):
        if len(vector) != self.input_size:
            raise ValueError("Input size mismatch")

        self.last_input = vector

        outputs = []

        for neuron_index, row in enumerate(self.weight_matrix.value):
            value = dot(vector, row) + self.bias.value[neuron_index]
            outputs.append(value)

        self.last_output = outputs
        return outputs

    def parameters(self):
        return [self.weight_matrix, self.bias]

    def initialize_parameters(self):
        weights = []
        bias = []
        for _ in range(self.output_size):
            row = []
            bias.append(random.uniform(-0.5, 0.5))

            for _ in range(self.input_size):
                row.append(random.uniform(-0.5, 0.5))

            weights.append(row)


        self.weight_matrix = Parameter(weights)
        self.bias = Parameter(bias)

  