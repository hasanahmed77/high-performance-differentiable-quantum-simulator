# Examples

Runnable scripts on the PennyLane device `ketjet.qubit`. Each one checks its own result, and CI runs all of them on
every pull request, so they stay working.

```bash
pip install "ketjet[pennylane]"
```

| Script | What it shows |
|---|---|
| [`bell.py`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/examples/bell.py) | A Bell state: exact probabilities, the ⟨Z⊗Z⟩ correlation, and seeded, reproducible shots |
| [`qft.py`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/examples/qft.py) | PennyLane's `QFT` template, decomposed by the device, matching the analytic formula to 1e-12 |
| [`vqe.py`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/examples/vqe.py) | A VQE for H₂ with a `DoubleExcitation` ansatz and adjoint gradients, reaching the exact ground-state energy |

For a walk-through of the same chemistry with ketjet's own API, see
[The ground state of H₂](vqe-h2.md).

Run one with, for example:

```bash
python examples/vqe.py
```

```text
Hartree-Fock energy: -1.11668215 Ha
step 10: -1.13726124 Ha
...
VQE energy:          -1.13726973 Ha
exact (FCI) energy:  -1.13726973 Ha  (error 1.6e-15)
```
