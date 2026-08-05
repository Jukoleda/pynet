from operation import Operation
from core.tensor import Tensor

class Add(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Shape mismatch")
        
        output = []
        
        for a_, b_ in zip(a.data, b.data):
            output.append(a_+b_)
        output = Tensor(output)

        output.parents = (a, b)
        output.requires_grad = (a.requires_grad or b.requires_grad)
        return output