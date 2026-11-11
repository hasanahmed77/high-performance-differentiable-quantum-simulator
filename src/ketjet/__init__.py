"""ketjet: a high-performance differentiable quantum statevector simulator.

The public API grows sprint by sprint; see ``plan.md`` in the repository for the roadmap.
"""

from ketjet.circuit import Circuit
from ketjet.statevector import MAX_QUBITS, state_nbytes, zero_state

__version__ = "0.1.0"

__all__ = ["MAX_QUBITS", "Circuit", "__version__", "state_nbytes", "zero_state"]
