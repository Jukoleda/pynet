class AutogradEngine:
    def backward(self, tensor):
        pass

    def _backward(self, tensor):
        pass


    def print_graph(self, tensor):
        self._print_graph(tensor, "", True)

    def _print_graph(self, tensor, prefix, is_last):

        connector = "└── " if is_last else "├── "

        if tensor.grad_fn is None:
            print(prefix + connector + "Tensor")
        else:
            print(
                prefix +
                connector +
                f"Tensor <- {tensor.grad_fn.__class__.__name__}"
            )

        new_prefix = prefix + ("    " if is_last else "│   ")

        for i, parent in enumerate(tensor.parents):
            self._print_graph(
                parent,
                new_prefix,
                i == len(tensor.parents) - 1
            )