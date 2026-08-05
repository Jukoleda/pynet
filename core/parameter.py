class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = 0
        self.requires_grad = True