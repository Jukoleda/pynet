

class Trainer:

    def __init__(self, epochs):
        self.epochs = epochs

    def train(self, model, dataset, optimizer):

        for epoch in range(self.epochs):
            for inputs, expected in dataset:
                prediction = model.forward(inputs)

                loss = model.loss(
                    prediction,
                    expected
                )

                model.backward(loss)

                optimizer.step()
