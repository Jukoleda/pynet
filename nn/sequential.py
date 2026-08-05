from nn.linear import Linear
from core.module import Module

class Sequential(Module):

    def __init__(self, sizes):
        super().__init__()
        self.modules = []

        for i in range(len(sizes) - 1):
            self.modules.append(
                Linear(sizes[i], sizes[i+1])
            )

    def forward(self, x):

        for module in self.modules:
            x = module.forward(x)

        return x

    def parameters(self):
        params = []

        for module in self.modules:
            params.extend(
                module.parameters()
            )
        return params