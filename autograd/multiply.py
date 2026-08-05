from autograd.operation import Operation

class Multiply(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        self.validate_shape(a, b)

        self.save_for_backward(a, b)

        data = []
                
        for a_, b_ in zip(a.data, b.data):
            data.append(a_*b_)
                
        return self.create_tensor(data, a, b)
    