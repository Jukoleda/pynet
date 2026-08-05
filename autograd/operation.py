class Operation:

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError