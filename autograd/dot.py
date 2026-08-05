from autograd.operation import Operation

class Dot(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self.validate_shape(a, b)

        self.save_for_backward(a, b)

        data = 0
        for x,w in zip(a.data, b.data):
            data += x * w
                
        return self.create_tensor(data, a, b)