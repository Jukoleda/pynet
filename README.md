# pynet

Open source framework to build AI models — written from scratch in pure Python, with **zero dependencies**.

pynet is an educational deep learning framework: no NumPy, no C extensions, no magic. Every tensor,
every gradient and every layer is plain Python code you can read top to bottom. The goal is to
understand *how* frameworks like PyTorch actually work by rebuilding one piece by piece.

> **Status: work in progress.** The autograd graph is built correctly during the forward pass;
> the backward pass is the feature currently under construction.
> See the [changelog](CHANGELOG.md) for what has landed so far.

## Requirements

- Python 3.10+ (no third-party packages)

## Quick start

```bash
git clone https://github.com/jukoleda/pynet.git
cd pynet
python main.py
```

Building a computation graph:

```python
from core.tensor import Tensor
from autograd.add import Add
from autograd.multiply import Multiply
from autograd.engine import AutogradEngine

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

c = Add().forward(a, b)        # [5, 7, 9]
d = Multiply().forward(c, b)   # [20, 35, 54]

AutogradEngine().print_graph(d, show_data=True)
```

Output:

```
└── Tensor#3 shape=(3,) data=[20, 35, 54] <- Multiply
    ├── Tensor#2 shape=(3,) data=[5, 7, 9] <- Add
    │   ├── Tensor#0 shape=(3,) data=[1, 2, 3]
    │   └── Tensor#1 shape=(3,) data=[4, 5, 6]
    └── Tensor#1 shape=(3,) data=[4, 5, 6]
```

Each result tensor remembers the operation that produced it (`grad_fn`) and the tensors it came
from (`parents`), which is exactly the graph the backward pass will walk.

Every tensor carries a unique `id`, so a tensor used more than once is recognisable as the *same*
node: `Tensor#1` above appears twice, as a parent of both `Add` and `Multiply`. That is precisely
the case where the backward pass will have to **accumulate** gradients instead of overwriting them.
Pass `show_data=False` (the default) for a compact tree without the values.

> On Windows, `print_graph` uses box-drawing characters that the default `cp1252` console cannot
> encode. Run it with `PYTHONIOENCODING=utf-8 python main.py` (or `chcp 65001`) to see the tree.

## Concepts

| Concept | File | Description |
| --- | --- | --- |
| `Tensor` | [core/tensor.py](core/tensor.py) | Holds nested-list data, infers its `shape`, gets a unique `id`, and tracks `grad`, `grad_fn` and `parents`. |
| `Operation` | [autograd/operation.py](autograd/operation.py) | Base class for every differentiable op: `forward`, `backward`, `save_for_backward`, `validate_shape` and `create_tensor` (which wires the graph). |
| `AutogradEngine` | [autograd/engine.py](autograd/engine.py) | Traverses the graph. `print_graph(tensor, show_data=False)` renders it as a tree; `backward` is not implemented yet. |
| `Module` | [core/module.py](core/module.py) | Base class for layers, exposes `parameters()`. |
| `Parameter` | [core/parameter.py](core/parameter.py) | A trainable value with its own `grad` and `requires_grad`. |

### Available operations

| Operation | File | Forward |
| --- | --- | --- |
| `Add` | [autograd/add.py](autograd/add.py) | Element-wise sum |
| `Multiply` | [autograd/multiply.py](autograd/multiply.py) | Element-wise product |
| `Dot` | [autograd/dot.py](autograd/dot.py) | Dot product (vector → scalar) |

### Layers and utilities

| Component | File | Description |
| --- | --- | --- |
| `Linear` | [nn/linear.py](nn/linear.py) | Fully connected layer with randomly initialised weights and bias. |
| `Sequential` | [nn/sequential.py](nn/sequential.py) | Stacks `Linear` layers from a list of sizes, e.g. `Sequential([4, 5, 6, 4])`. |
| `Embedding` | [nn/embedding.py](nn/embedding.py) | Lookup table mapping token ids to dense vectors. |
| `Tokenizer` | [tokenizer.py](tokenizer.py) | Word-level tokenizer with `fit` / `encode` / `decode`. |
| `Trainer` | [trainer.py](trainer.py) | Training loop skeleton: forward → loss → backward → optimizer step. |

## Project structure

```
pynet/
├── core/            # Tensor, Parameter, Module
├── autograd/        # Operation base class, ops and the autograd engine
├── nn/              # Layers: linear, sequential, embedding
├── optim/           # Optimizers (planned)
├── losses/          # Loss functions (planned)
├── models/          # Complete models (planned)
├── legacy/          # Earlier neuron/layer/vector implementations, kept for reference
├── tokenizer.py
├── trainer.py
├── dataset.txt      # Small toy corpus
└── main.py          # Playground / entry point
```

The `nn` layers and the `Parameter` class come from the pre-autograd stage of the project and still
operate on plain Python lists. Migrating them to `Tensor` is part of the roadmap below.

## Roadmap

What is already done lives in the [changelog](CHANGELOG.md). What is left:

- [ ] Backward pass (`AutogradEngine.backward` + `backward` on each operation)
- [ ] Operator overloading on `Tensor` (`+`, `*`, `@`)
- [ ] `MatMul` and activations (`ReLU`, `Sigmoid`)
- [ ] Loss functions: MSE, cross entropy
- [ ] Optimizers: `Optimizer` base class and SGD
- [ ] Port `Linear` / `Sequential` to `Tensor` and `Parameter` with gradients
- [ ] Train an end-to-end model with `Trainer`

## License

[MIT](LICENSE) © Jukoleda
