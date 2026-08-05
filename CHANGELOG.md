# Changelog

All notable changes to pynet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing has been released yet. The first version, `0.1.0`, will be cut once the
backward pass can train a model end to end.

### Added

- `Tensor`: nested-list data with shape inference, a unique `id` from a
  class-level counter, and the graph metadata `grad`, `grad_fn` and `parents`.
- `Operation` base class with the wiring shared by every op: `save_for_backward`,
  `validate_shape` and `create_tensor`, which sets `grad_fn` and `parents` and
  propagates `requires_grad` onto the result.
- Forward pass for `Add` (element-wise sum), `Multiply` (element-wise product)
  and `Dot` (dot product).
- `AutogradEngine.print_graph`, which renders the computation graph as a tree.
  Nodes print as `Tensor#id shape=...`, and the `show_data` flag appends the
  tensor values.
- `.gitignore` covering `__pycache__`, build artifacts, virtualenvs and editor
  directories.
- README with the project overview, concepts, structure and roadmap.

### Changed

- `Add` now builds its result through the `Operation` helpers instead of wiring
  the output tensor by hand.
- Moved `linalg/` to `legacy/linalg/`, next to the other pre-autograd code, and
  updated the import in `nn.linear`.
- `main.py` is now a playground for the autograd graph.

### Fixed

- Circular import between `Tensor` and `Add`, by importing `Add` lazily inside
  `Tensor.__add__`.

### Removed

- `__pycache__` directories that had been committed to version control.

[Unreleased]: https://github.com/jukoleda/pynet/commits/main
