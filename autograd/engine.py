class AutogradEngine:
    def backward(self, tensor):
        pass

    def _backward(self, tensor):
        pass


    def print_graph(self, tensor, show_data=False):
        self._print_graph(tensor, "", is_last=True, show_data=show_data)

    def _print_graph(self, tensor, prefix, is_last, show_data):

        connector = "└── " if is_last else "├── "

        if tensor.grad_fn is None:
            print(
                prefix +
                connector +
                f"Tensor#{tensor.id} shape={tensor.shape}" +
                (f" data={tensor.data}" if show_data else "")
            )
        else:
            print(
                prefix +
                connector +
                f"Tensor#{tensor.id} shape={tensor.shape}" + 
                (f" data={tensor.data}" if show_data else "") +
                f" <- {tensor.grad_fn.__class__.__name__}"
            )

        new_prefix = prefix + ("    " if is_last else "│   ")

        for i, parent in enumerate(tensor.parents):
            self._print_graph(
                parent,
                new_prefix,
                i == len(tensor.parents) - 1,
                show_data
            )