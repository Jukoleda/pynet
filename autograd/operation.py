from core.tensor import Tensor

class Operation:
    def __init__(self):
        self.saved_tensors = ()

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def validate_shape(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Shape mismatch")

    def create_tensor(self, data, *parents):
        result = Tensor(data)
        result.grad_fn = self
        result.parents = parents
        result.requires_grad = any( p.requires_grad for p in parents)
        return result