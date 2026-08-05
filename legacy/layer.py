from neuron import Neuron

class Layer:

    def __init__(self, input_size, output_size):
        self.neurons = []

        for _ in range(output_size):
            self.neurons.append(
                Neuron(input_size)
            )

    def forward(self, input):
        output = []

        for neuron in self.neurons:
            output.append(
                neuron.forward(input)
            )

        return output

    def backward(self, gradient, learning_rate):
        