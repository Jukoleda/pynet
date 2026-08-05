
class Tensor:

    def __init__(self, data):
        self.data = data
        self.grad = None

        self.parents = ()
        self.grad_fn = None

        self.shape = self.compute_shape(data)
        self.requires_grad = True

    def __add__(self, other):
        from autograd.add import Add
        return Add.forward(self, other)

    def compute_shape(self, data):
        if not isinstance(data, (list, tuple)):
            return ()

        if len(data) == 0:
            return (0,)

        return (len(data),) + self.compute_shape(data[0])