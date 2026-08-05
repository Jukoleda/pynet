class Module:
    def __call__(self, input):
        raise NotImplementedError

    def parameters(self):
        return []